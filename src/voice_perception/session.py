"""Per-session state management."""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from voice_perception import config
from voice_perception.audio import StreamingAudioDecoder
from voice_perception.fusion import HesitationScorer

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SessionState:
    session_id: str
    scorer: HesitationScorer = field(default_factory=HesitationScorer)
    transcript_buffer: list[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=utc_now)
    last_accessed_at: datetime = field(default_factory=utc_now)
    perception_result: dict[str, Any] = field(default_factory=dict)
    hesitation_score: float = 0.0
    chunks_processed: int = 0
    decoder: StreamingAudioDecoder = field(default_factory=StreamingAudioDecoder)

    def update(self, result: dict[str, Any]) -> None:
        self.perception_result = dict(result)
        self.hesitation_score = self.scorer.update(result)
        self.chunks_processed += 1
        self.updated_at = utc_now()
        self.last_accessed_at = self.updated_at
        self._append_transcript(str(result.get("transcript", "")))

    def to_response(self) -> dict[str, Any]:
        self.last_accessed_at = utc_now()
        result = self.perception_result or _default_perception_result()
        return {
            "session_id": self.session_id,
            "updated_at": _iso_z(self.updated_at),
            "transcript_partial": self.transcript_partial,
            "emotion": result.get("emotion", "NEUTRAL"),
            "emotion_confidence": float(result.get("emotion_confidence", 0.0)),
            "events": list(result.get("events", [])),
            "hesitation_score": self.hesitation_score,
            "chunks_processed": self.chunks_processed,
        }

    @property
    def transcript_partial(self) -> str:
        joined = " ".join(part for part in self.transcript_buffer if part).strip()
        return joined[-config.TRANSCRIPT_MAX_CHARS :]

    def _append_transcript(self, transcript: str) -> None:
        if not transcript.strip():
            return
        self.transcript_buffer.append(transcript.strip())
        if len(self.transcript_partial) >= config.TRANSCRIPT_MAX_CHARS:
            self.transcript_buffer = [self.transcript_partial]


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


def _default_perception_result() -> dict[str, Any]:
    return {
        "transcript": "",
        "emotion": "NEUTRAL",
        "emotion_confidence": 0.0,
        "events": [],
        "silence_ratio": 0.0,
        "inference_ms": 0,
    }


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
