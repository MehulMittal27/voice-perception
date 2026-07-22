"""Audio decoding and PCM helpers."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import av
import numpy as np
from av.audio.resampler import AudioResampler

from voice_perception import config

logger = logging.getLogger(__name__)


class AudioDecodeError(RuntimeError):
    """Raised when PyAV cannot decode a chunk into audio frames."""


@dataclass
class StreamingAudioDecoder:
    """Decode MediaRecorder chunks that may be standalone or fragmented."""

    max_buffer_bytes: int = config.MAX_AUDIO_BUFFER_BYTES
    _buffer: bytearray = field(default_factory=bytearray)
    _stream_samples_returned: int = 0
    _chunks_seen: int = 0
    _mode: str = "unknown"
    last_format: str = "unknown"

    def decode(self, raw_bytes: bytes) -> np.ndarray:
        if not raw_bytes:
            return empty_pcm()
        self._chunks_seen += 1
        self._append(raw_bytes)
        if self._mode == "chunk":
            chunk_pcm = self._safe_decode(raw_bytes)
            if chunk_pcm.size:
                return chunk_pcm
        stream_pcm = self._safe_decode(bytes(self._buffer))
        if stream_pcm.size > self._stream_samples_returned:
            new_pcm = stream_pcm[self._stream_samples_returned :]
            self._stream_samples_returned = stream_pcm.size
            if self._chunks_seen > 1:
                self._mode = "stream"
            return new_pcm.astype(np.float32, copy=False)
        chunk_pcm = self._safe_decode(raw_bytes)
        if chunk_pcm.size:
            self._mode = "chunk"
            return chunk_pcm
        logger.debug("event=audio_buffered bytes=%d", len(self._buffer))
        return empty_pcm()

    def _append(self, raw_bytes: bytes) -> None:
        self._buffer.extend(raw_bytes)
        if len(self._buffer) > self.max_buffer_bytes:
            logger.warning("event=audio_buffer_reset bytes=%d", len(self._buffer))
            self._buffer = bytearray(raw_bytes)
            self._stream_samples_returned = 0
            self._mode = "unknown"

    def _safe_decode(self, raw_bytes: bytes) -> np.ndarray:
        try:
            pcm, format_name = _decode_container(raw_bytes)
            self.last_format = format_name
            return pcm
        except Exception as exc:
            logger.debug("event=audio_decode_retryable error=%s", exc)
            return empty_pcm()


def decode_chunk(raw_bytes: bytes) -> np.ndarray:
    """Decode one raw container chunk into 16 kHz mono float32 PCM."""

    try:
        pcm, _ = _decode_container(raw_bytes)
        return pcm
    except Exception as exc:
        logger.debug("event=audio_decode_failed error=%s", exc)
        return empty_pcm()


def compute_silence_ratio(pcm_16khz_mono: np.ndarray) -> float:
    """Return the fraction of short RMS frames below the silence threshold."""

    pcm = to_float32_mono(pcm_16khz_mono)
    if pcm.size == 0:
        return 1.0
    frame_len = max(1, int(config.SAMPLE_RATE * config.SILENCE_FRAME_MS / 1000))
    silent = 0
    total = 0
    for start in range(0, pcm.size, frame_len):
        frame = pcm[start : start + frame_len]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))))
        silent += int(rms < config.SILENCE_RMS_THRESHOLD)
        total += 1
    return silent / total if total else 1.0


def load_wav_16khz_mono(path: str | Path) -> np.ndarray:
    """Load an audio file with soundfile and resample to 16 kHz mono."""

    import soundfile as sf

    data, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    pcm = to_float32_mono(np.asarray(data))
    return resample_to_16khz_mono(pcm, int(sample_rate))


def resample_to_16khz_mono(data: np.ndarray, source_rate: int) -> np.ndarray:
    """Convert arbitrary mono or stereo PCM to 16 kHz mono float32."""

    pcm = to_float32_mono(data)
    if pcm.size == 0 or source_rate == config.SAMPLE_RATE:
        return pcm
    duration = pcm.size / float(source_rate)
    target_len = max(1, int(round(duration * config.SAMPLE_RATE)))
    source_x = np.linspace(0.0, pcm.size - 1, num=pcm.size, dtype=np.float64)
    target_x = np.linspace(0.0, pcm.size - 1, num=target_len, dtype=np.float64)
    resampled = np.interp(target_x, source_x, pcm).astype(np.float32)
    return np.ascontiguousarray(np.clip(resampled, -1.0, 1.0))


def to_float32_mono(data: np.ndarray) -> np.ndarray:
    """Normalize array shape and dtype to mono float32 PCM."""

    arr = np.asarray(data)
    if arr.size == 0:
        return empty_pcm()
    if arr.ndim > 1:
        arr = _mix_to_mono(arr)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float32) / max(abs(info.min), info.max)
    else:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(np.clip(arr.reshape(-1), -1.0, 1.0))


def empty_pcm() -> np.ndarray:
    return np.empty(0, dtype=np.float32)


def _decode_container(raw_bytes: bytes) -> tuple[np.ndarray, str]:
    if not raw_bytes:
        return empty_pcm(), "empty"
    container = av.open(io.BytesIO(raw_bytes), mode="r")
    format_name = container.format.name if container.format else "unknown"
    resampler = AudioResampler(format="flt", layout="mono", rate=config.SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    try:
        for frame in container.decode(audio=0):
            for resampled in _iter_frames(resampler.resample(frame)):
                chunks.append(to_float32_mono(resampled.to_ndarray()))
    except Exception as exc:
        if not chunks:
            raise AudioDecodeError(str(exc)) from exc
        logger.debug("event=audio_decode_partial error=%s", exc)
    finally:
        container.close()
    if not chunks:
        return empty_pcm(), format_name
    return np.concatenate(chunks).astype(np.float32, copy=False), format_name


def _iter_frames(frame_or_frames: object) -> Iterable[av.AudioFrame]:
    if frame_or_frames is None:
        return []
    if isinstance(frame_or_frames, list):
        return frame_or_frames
    return [frame_or_frames]


def _mix_to_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        return arr.mean(axis=0)
    return arr.mean(axis=1)
