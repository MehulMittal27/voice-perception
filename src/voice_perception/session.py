"""Per-session state management."""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from voice_perception import config
from voice_perception.audio import StreamingAudioDecoder, empty_pcm, to_float32_mono
from voice_perception.fusion import HesitationScorer
from voice_perception.signals import analyze_acoustic_context, default_acoustic_analysis

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RollingAudioBuffer:
    """Keep recent live PCM and decide when there is enough context."""

    max_samples: int = max(config.LIVE_ROLLING_SAMPLES, config.LIVE_MIN_CONTEXT_SAMPLES)
    min_samples: int = config.LIVE_MIN_CONTEXT_SAMPLES
    hop_samples: int = config.LIVE_INFERENCE_HOP_SAMPLES
    _samples: np.ndarray = field(default_factory=empty_pcm)
    _samples_since_inference: int = 0

    def append(self, pcm_16khz_mono: np.ndarray) -> np.ndarray | None:
        chunk = to_float32_mono(pcm_16khz_mono)
        if chunk.size == 0:
            return None
        combined = np.concatenate((self._samples, chunk))
        self._samples = _trim_samples(combined, self.max_samples)
        self._samples_since_inference += chunk.size
        if not self._ready_for_inference():
            return None
        self._samples_since_inference = 0
        return self._samples.copy()

    @property
    def samples(self) -> int:
        return int(self._samples.size)

    def snapshot(self) -> np.ndarray:
        return self._samples.copy()

    def _ready_for_inference(self) -> bool:
        if self._samples.size < self.min_samples:
            return False
        return self._samples_since_inference >= self.hop_samples


@dataclass
class SessionState:
    session_id: str
    scorer: HesitationScorer = field(default_factory=HesitationScorer)
    transcript_text: str = ""
    updated_at: datetime = field(default_factory=utc_now)
    last_accessed_at: datetime = field(default_factory=utc_now)
    perception_result: dict[str, Any] = field(default_factory=dict)
    hesitation_score: float = 0.0
    chunks_processed: int = 0
    decoder: StreamingAudioDecoder = field(default_factory=StreamingAudioDecoder)
    audio_buffer: RollingAudioBuffer = field(default_factory=RollingAudioBuffer)
    acoustic_result: dict[str, Any] = field(default_factory=default_acoustic_analysis)

    def ingest_audio(self, pcm_16khz_mono: np.ndarray) -> np.ndarray | None:
        self.last_accessed_at = utc_now()
        window = self.audio_buffer.append(pcm_16khz_mono)
        self._update_acoustic_from_buffer()
        return window

    def update(self, result: dict[str, Any]) -> None:
        self.perception_result = dict(result)
        self._store_acoustic(result)
        self.hesitation_score = self.scorer.update(result)
        self.chunks_processed += 1
        self.updated_at = utc_now()
        self.last_accessed_at = self.updated_at
        self._replace_transcript(str(result.get("transcript", "")))

    def mark_no_speech(self, result: dict[str, Any]) -> None:
        self.perception_result = dict(result)
        self._store_acoustic(result)
        self.scorer = HesitationScorer()
        self.hesitation_score = 0.0
        self.updated_at = utc_now()
        self.last_accessed_at = self.updated_at
        self._replace_transcript("")

    def to_response(self) -> dict[str, Any]:
        self.last_accessed_at = utc_now()
        result = self.perception_result or _default_perception_result()
        response = {
            "session_id": self.session_id,
            "updated_at": _iso_z(self.updated_at),
            "transcript_partial": self.transcript_partial,
            "emotion": result.get("emotion", "NEUTRAL"),
            "emotion_confidence": float(result.get("emotion_confidence", 0.0)),
            "events": list(result.get("events", [])),
            "hesitation_score": self.hesitation_score,
            "chunks_processed": self.chunks_processed,
            "no_speech": bool(result.get("no_speech", False)),
        }
        response.update(_additive_response_fields(result, self.acoustic_result, self.updated_at))
        return response

    @property
    def transcript_partial(self) -> str:
        return self.transcript_text[-config.TRANSCRIPT_MAX_CHARS :]

    def _replace_transcript(self, transcript: str) -> None:
        self.transcript_text = transcript.strip()[-config.TRANSCRIPT_MAX_CHARS :]

    def _update_acoustic_from_buffer(self) -> None:
        context = self.audio_buffer.snapshot()
        if context.size:
            self.acoustic_result = analyze_acoustic_context(context)

    def _store_acoustic(self, result: dict[str, Any]) -> None:
        if "voice_state" in result and "signals" in result:
            self.acoustic_result = _extract_acoustic_fields(result)


def _additive_response_fields(
    result: dict[str, Any], acoustic: dict[str, Any], updated_at: datetime
) -> dict[str, Any]:
    fields = _extract_acoustic_fields(acoustic)
    voice_state = dict(fields.get("voice_state", {}))
    if voice_state.get("updated_at") is None:
        voice_state["updated_at"] = _iso_z(updated_at)
    fields["voice_state"] = voice_state
    for key in _MODEL_DEBUG_KEYS:
        if key in result:
            fields[key] = result[key]
    if "capabilities" not in fields:
        fields["capabilities"] = _default_capabilities()
    return fields


def _extract_acoustic_fields(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "voice_state": dict(source.get("voice_state", {})),
        "signals": dict(source.get("signals", {})),
        "signal_events": list(source.get("signal_events", [])),
        "score_drivers": dict(source.get("score_drivers", {})),
        "debug_features": dict(source.get("debug_features", {})),
    }


_MODEL_DEBUG_KEYS = (
    "emotion_source",
    "raw_emotion",
    "raw_ser_label",
    "sensevoice_emotion",
    "sensevoice_emotion_confidence",
    "sensevoice_raw_emotion",
    "ser",
    "capabilities",
)


def _default_capabilities() -> dict[str, Any]:
    return {
        "emotion_labels_supported": config.SER_ENABLED,
        "emotion_probabilities_calibrated": False,
        "event_labels_supported": True,
        "transcript_supported": True,
        "voice_state_supported": True,
        "score_drivers_supported": True,
        "ser_license_caveat": config.SER_LICENSE_CAVEAT if config.SER_ENABLED else None,
    }


class SessionManager:
    """Singleton holder for active voice perception sessions."""

    _instance: "SessionManager | None" = None
    _initialized = False

    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self.__class__._initialized:
            return
        self.sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()
        self.__class__._initialized = True

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self.sessions[session_id] = SessionState(session_id=session_id)
        logger.info("event=session_created session_id=%s", session_id)
        return session_id

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            state = self.sessions.get(session_id)
            if state:
                state.last_accessed_at = utc_now()
            return state

    def update(self, session_id: str, result: dict[str, Any]) -> SessionState:
        with self._lock:
            state = self.sessions[session_id]
            state.update(result)
        logger.info(
            "event=session_updated session_id=%s chunks=%d hesitation=%.3f",
            session_id,
            state.chunks_processed,
            state.hesitation_score,
        )
        return state

    def mark_no_speech(self, session_id: str, result: dict[str, Any]) -> SessionState:
        with self._lock:
            state = self.sessions[session_id]
            state.mark_no_speech(result)
        logger.info("event=session_no_speech session_id=%s", session_id)
        return state

    def end(self, session_id: str) -> bool:
        with self._lock:
            removed = self.sessions.pop(session_id, None) is not None
        logger.info("event=session_ended session_id=%s removed=%s", session_id, removed)
        return removed

    def expire_sessions(self) -> int:
        cutoff = utc_now() - timedelta(seconds=config.SESSION_TTL_SECONDS)
        with self._lock:
            expired = [sid for sid, state in self.sessions.items() if state.last_accessed_at < cutoff]
            for session_id in expired:
                self.sessions.pop(session_id, None)
        if expired:
            logger.info("event=sessions_expired count=%d", len(expired))
        return len(expired)

    async def expire_sessions_loop(self) -> None:
        while True:
            await asyncio.sleep(config.SESSION_SWEEP_SECONDS)
            self.expire_sessions()


def _trim_samples(samples: np.ndarray, max_samples: int) -> np.ndarray:
    if samples.size <= max_samples:
        return np.ascontiguousarray(samples, dtype=np.float32)
    return np.ascontiguousarray(samples[-max_samples:], dtype=np.float32)


def _default_perception_result() -> dict[str, Any]:
    return {
        "transcript": "",
        "emotion": "NEUTRAL",
        "emotion_confidence": 0.0,
        "events": [],
        "silence_ratio": 0.0,
        "inference_ms": 0,
        "no_speech": True,
    }


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
