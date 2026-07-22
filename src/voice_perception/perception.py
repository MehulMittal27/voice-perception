"""SenseVoice-Small ONNX wrapper for paralinguistic perception."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voice_perception import config
from voice_perception.audio import (
    SpeechActivity,
    analyze_speech_activity,
    load_wav_16khz_mono,
    to_float32_mono,
)

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"<\|([^|<>]+)\|>")
EMOTION_LABELS = set(config.EMOTION_STRESS.keys())
EVENT_ALIASES = {
    "SPEECH": "Speech",
    "BREATH": "Breath",
    "COUGH": "Cough",
    "CRY": "Cry",
    "LAUGHTER": "Laughter",
    "LAUGH": "Laughter",
    "APPLAUSE": "Applause",
    "BGM": "BGM",
    "MUSIC": "Music",
    "NOISE": "Noise",
    "SNEEZE": "Sneeze",
}
CONTROL_TOKENS = {"ZH", "EN", "YUE", "JA", "KO", "NOSPEECH", "WITHITN", "WOITN"}


class VoicePerception:
    """Load SenseVoice once and analyze 16 kHz mono PCM chunks."""

    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = model_dir or config.SENSEVOICE_MODEL_DIR
        self.resolved_model_dir: Path | None = None
        self._lock = threading.Lock()
        self._input_mode: str | None = None
        start = time.perf_counter()
        self.model = self._load_model()
        self.load_ms = int((time.perf_counter() - start) * 1000)
        logger.info("event=model_loaded model_dir=%s load_ms=%d", self.model_dir, self.load_ms)

    def analyze(self, pcm_16khz_mono: np.ndarray) -> dict[str, Any]:
        pcm = to_float32_mono(pcm_16khz_mono)
        activity = analyze_speech_activity(pcm)
        if pcm.size < config.MIN_INFERENCE_SAMPLES:
            return self._empty_result(activity.silence_ratio)
        if not activity.has_speech:
            return self._no_speech_result(activity)
        start = time.perf_counter()
        with self._lock:
            raw_output = self._infer(pcm)
        inference_ms = int((time.perf_counter() - start) * 1000)
        parsed = parse_sensevoice_output(raw_output)
        parsed["silence_ratio"] = activity.silence_ratio
        parsed["inference_ms"] = inference_ms
        parsed["no_speech"] = False
        return parsed

    def _load_model(self) -> Any:
        try:
            from funasr_onnx import SenseVoiceSmall

            model_path = self._resolve_model_dir()
            self._ensure_sentencepiece_model(model_path)
            self.resolved_model_dir = model_path
            logger.info("event=model_config language=%s", config.SENSEVOICE_LANGUAGE)
            return SenseVoiceSmall(
                str(model_path),
                batch_size=1,
                device_id=config.SENSEVOICE_DEVICE_ID,
                quantize=config.SENSEVOICE_QUANTIZE,
                intra_op_num_threads=config.SENSEVOICE_THREADS,
            )
        except TypeError as exc:
            if "exceptions must derive" in str(exc):
                raise RuntimeError(
                    "funasr-onnx could not load the SenseVoice model. Use "
                    "iic/SenseVoiceSmall-onnx or set SENSEVOICE_MODEL_DIR to a "
                    "local ONNX model path; PyTorch repos require a funasr export step."
                ) from exc
            raise

    def _resolve_model_dir(self) -> Path:
        model_path = Path(self.model_dir)
        if model_path.exists():
            return model_path
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install modelscope to download remote SenseVoice models.") from exc
        cache_dir = str(Path(config.SENSEVOICE_CACHE_DIR))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return Path(snapshot_download(self.model_dir, cache_dir=cache_dir))

    def _ensure_sentencepiece_model(self, model_path: Path) -> None:
        bpe_path = model_path / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        if bpe_path.exists():
            return
        try:
            from modelscope.hub.file_download import model_file_download
        except ImportError as exc:
            raise RuntimeError("Install modelscope to fetch the SenseVoice tokenizer model.") from exc
        source = model_file_download(
            "iic/SenseVoiceSmall",
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
            cache_dir=str(Path(config.SENSEVOICE_CACHE_DIR)),
        )
        if not source:
            raise RuntimeError("Could not download SenseVoice SentencePiece tokenizer model.")
        shutil.copyfile(source, bpe_path)

    def _infer(self, pcm: np.ndarray) -> Any:
        if self._input_mode == "file":
            return self._infer_file(pcm)
        try:
            output = self._infer_numpy(pcm)
            self._input_mode = "numpy"
            return output
        except Exception as exc:
            logger.info("event=model_numpy_input_failed fallback=file error=%s", exc)
            self._input_mode = "file"
            return self._infer_file(pcm)

    def _infer_numpy(self, pcm: np.ndarray) -> Any:
        return self.model(
            np.ascontiguousarray(pcm, dtype=np.float32),
            language=config.SENSEVOICE_LANGUAGE,
            textnorm="withitn",
        )

    def _infer_file(self, pcm: np.ndarray) -> Any:
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = handle.name
            sf.write(temp_path, pcm, config.SAMPLE_RATE, subtype="PCM_16")
            return self.model(
                temp_path,
                language=config.SENSEVOICE_LANGUAGE,
                textnorm="withitn",
            )
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    @staticmethod
    def _empty_result(silence_ratio: float) -> dict[str, Any]:
        return {
            "transcript": "",
            "emotion": "NEUTRAL",
            "emotion_confidence": 0.0,
            "raw_emotion": None,
            "events": [],
            "silence_ratio": silence_ratio,
            "inference_ms": 0,
            "no_speech": True,
        }

    @staticmethod
    def _no_speech_result(activity: SpeechActivity) -> dict[str, Any]:
        return {
            "transcript": "",
            "emotion": "NEUTRAL",
            "emotion_confidence": 0.0,
            "raw_emotion": None,
            "events": [],
            "silence_ratio": activity.silence_ratio,
            "inference_ms": 0,
            "no_speech": True,
        }


def parse_sensevoice_output(raw_output: Any) -> dict[str, Any]:
    record = _first_record(raw_output)
    raw_text, direct_confidence = _extract_text_and_confidence(record)
    tokens = TOKEN_RE.findall(raw_text)
    emotion, token_has_emotion, raw_emotion = _extract_emotion(tokens)
    events = _extract_events(tokens)
    transcript = _strip_tokens(raw_text)
    _log_unknown_emotion(raw_emotion, raw_text)
    if direct_confidence is None:
        # funasr-onnx 0.4.1 returns decoded tokens, not per-label probabilities.
        # Presence of an explicit emotion token is the confidence proxy.
        confidence = 1.0 if token_has_emotion else 0.0
    else:
        confidence = _clip(direct_confidence)
    return {
        "transcript": transcript,
        "emotion": emotion,
        "emotion_confidence": confidence,
        "raw_emotion": raw_emotion,
        "events": events,
        "silence_ratio": 0.0,
        "inference_ms": 0,
    }


def _log_unknown_emotion(raw_emotion: str | None, raw_text: str) -> None:
    if raw_emotion is None or raw_emotion in EMOTION_LABELS:
        return
    logger.info(
        "event=model_unknown_emotion raw_emotion=%s output_chars=%d",
        raw_emotion,
        len(raw_text),
    )


def _first_record(raw_output: Any) -> Any:
    if isinstance(raw_output, (list, tuple)):
        return raw_output[0] if raw_output else ""
    return raw_output


def _extract_text_and_confidence(record: Any) -> tuple[str, float | None]:
    if isinstance(record, dict):
        text = str(record.get("text") or record.get("transcript") or record.get("result") or "")
        return text, _find_confidence(record)
    return str(record), None


def _find_confidence(record: dict[str, Any]) -> float | None:
    for key in ("emotion_confidence", "confidence", "score"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_emotion(tokens: list[str]) -> tuple[str, bool, str | None]:
    raw_emotion: str | None = None
    for token in tokens:
        label = token.strip().upper()
        if label in EMOTION_LABELS:
            return label, True, label
        if label.startswith("EMO_"):
            raw_emotion = label
    return "NEUTRAL", False, raw_emotion


def _extract_events(tokens: list[str]) -> list[str]:
    events: list[str] = []
    for token in tokens:
        event = _normalize_event(token)
        if event and event not in events:
            events.append(event)
    return events


def _normalize_event(token: str) -> str | None:
    label = token.strip().upper()
    if label in EMOTION_LABELS or label in CONTROL_TOKENS:
        return None
    return EVENT_ALIASES.get(label)


def _strip_tokens(raw_text: str) -> str:
    without_tokens = TOKEN_RE.sub(" ", raw_text)
    compact = re.sub(r"\s+", " ", without_tokens).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", compact)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SenseVoice perception on a WAV file.")
    parser.add_argument("wav_file", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pcm = load_wav_16khz_mono(args.wav_file)
    perception = VoicePerception()
    result = perception.analyze(pcm)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
