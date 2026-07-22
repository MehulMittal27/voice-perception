"""Deterministic live emotion stabilization for rolling SER windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from voice_perception import config

NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class EmotionDecision:
    """Displayed live emotion plus raw model metadata."""

    label: str
    confidence: float
    raw_label: str
    raw_confidence: float
    decision: str
    pending_label: str | None
    pending_count: int
    neutral_count: int

    def to_debug(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "raw_label": self.raw_label,
            "raw_confidence": self.raw_confidence,
            "decision": self.decision,
            "pending_label": self.pending_label,
            "pending_count": self.pending_count,
            "neutral_count": self.neutral_count,
            "hold_seconds": config.LIVE_EMOTION_HOLD_SECONDS,
            "switch_updates": config.LIVE_EMOTION_NEW_LABEL_UPDATES,
            "neutral_updates": config.LIVE_EMOTION_NEUTRAL_UPDATES,
            "source": "server_session_stabilizer",
        }


@dataclass
class LiveEmotionStabilizer:
    """Prevent live Emotion2Vec display flicker across rolling windows."""

    stable_label: str = NEUTRAL
    stable_confidence: float = 0.0
    pending_label: str | None = None
    pending_count: int = 0
    neutral_count: int = 0
    last_strong_non_neutral_at: datetime | None = None

    def apply(self, result: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
        output = dict(result)
        decision = self.update(
            str(output.get("emotion", NEUTRAL)),
            float(output.get("emotion_confidence", 0.0)),
            bool(output.get("no_speech", False)),
            now,
        )
        self._attach_debug(output, decision)
        output["emotion"] = decision.label
        output["emotion_confidence"] = decision.confidence
        output["emotion_source"] = _stable_source(str(output.get("emotion_source", "unknown")))
        _refresh_score_driver(output, decision.label, decision.confidence)
        return output

    def update(
        self,
        label: str,
        confidence: float,
        no_speech: bool = False,
        now: datetime | None = None,
    ) -> EmotionDecision:
        now = now or datetime.now(timezone.utc)
        raw_label = _normalize_label(label)
        raw_confidence = _clip(confidence)
        if no_speech:
            return self.reset(raw_label=raw_label, raw_confidence=raw_confidence)
        decision = self._update_speech(raw_label, raw_confidence, now)
        return self._decision(raw_label, raw_confidence, decision)

    def reset(self, raw_label: str = NEUTRAL, raw_confidence: float = 0.0) -> EmotionDecision:
        self.stable_label = NEUTRAL
        self.stable_confidence = 0.0
        self.pending_label = None
        self.pending_count = 0
        self.neutral_count = 0
        self.last_strong_non_neutral_at = None
        return self._decision(raw_label, raw_confidence, "reset_no_speech")

    def _update_speech(self, raw_label: str, raw_confidence: float, now: datetime) -> str:
        if raw_label == self.stable_label:
            return self._confirm_current(raw_label, raw_confidence, now)
        if raw_label == NEUTRAL:
            return self._handle_neutral(raw_confidence, now)
        return self._handle_non_neutral(raw_label, raw_confidence, now)

    def _confirm_current(self, raw_label: str, raw_confidence: float, now: datetime) -> str:
        self.pending_label = None
        self.pending_count = 0
        self.neutral_count = 0
        self.stable_confidence = raw_confidence
        if raw_label != NEUTRAL and raw_confidence >= config.LIVE_EMOTION_STRONG_CONFIDENCE:
            self.last_strong_non_neutral_at = now
        return "same_label"

    def _handle_neutral(self, raw_confidence: float, now: datetime) -> str:
        self.pending_label = None
        self.pending_count = 0
        self.neutral_count = self.neutral_count + 1 if _high_neutral(raw_confidence) else 0
        if self.stable_label == NEUTRAL:
            self.stable_confidence = raw_confidence
            return "same_neutral"
        if self.neutral_count >= config.LIVE_EMOTION_NEUTRAL_UPDATES:
            self._switch(NEUTRAL, raw_confidence, now)
            return "switch_repeated_neutral"
        if self._holding_strong(now):
            return "hold_non_neutral"
        return "awaiting_repeated_neutral"

    def _handle_non_neutral(self, raw_label: str, raw_confidence: float, now: datetime) -> str:
        self.neutral_count = 0
        if self._should_switch_immediately(raw_label, raw_confidence):
            self._switch(raw_label, raw_confidence, now)
            return "switch_immediate"
        self._track_pending(raw_label)
        if self.pending_count >= config.LIVE_EMOTION_NEW_LABEL_UPDATES:
            self._switch(raw_label, raw_confidence, now)
            return "switch_sustained"
        return "awaiting_sustained_label"

    def _should_switch_immediately(self, raw_label: str, raw_confidence: float) -> bool:
        if self.stable_label == NEUTRAL:
            return raw_confidence >= config.LIVE_EMOTION_STRONG_CONFIDENCE
        margin = raw_confidence - self.stable_confidence
        return (
            raw_label != self.stable_label
            and raw_confidence >= config.LIVE_EMOTION_IMMEDIATE_SWITCH_CONFIDENCE
            and margin >= config.LIVE_EMOTION_IMMEDIATE_SWITCH_MARGIN
        )

    def _track_pending(self, raw_label: str) -> None:
        if self.pending_label == raw_label:
            self.pending_count += 1
            return
        self.pending_label = raw_label
        self.pending_count = 1

    def _switch(self, label: str, confidence: float, now: datetime) -> None:
        self.stable_label = label
        self.stable_confidence = confidence
        self.pending_label = None
        self.pending_count = 0
        self.neutral_count = 0
        if label == NEUTRAL:
            self.last_strong_non_neutral_at = None
        elif confidence >= config.LIVE_EMOTION_STRONG_CONFIDENCE:
            self.last_strong_non_neutral_at = now

    def _holding_strong(self, now: datetime) -> bool:
        if self.stable_label == NEUTRAL or self.last_strong_non_neutral_at is None:
            return False
        elapsed = (now - self.last_strong_non_neutral_at).total_seconds()
        return elapsed <= config.LIVE_EMOTION_HOLD_SECONDS

    def _decision(self, raw_label: str, raw_confidence: float, decision: str) -> EmotionDecision:
        return EmotionDecision(
            label=self.stable_label,
            confidence=self.stable_confidence,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            decision=decision,
            pending_label=self.pending_label,
            pending_count=self.pending_count,
            neutral_count=self.neutral_count,
        )

    @staticmethod
    def _attach_debug(output: dict[str, Any], decision: EmotionDecision) -> None:
        output["live_raw_emotion"] = decision.raw_label
        output["live_raw_emotion_confidence"] = decision.raw_confidence
        output["live_stabilized_emotion"] = decision.to_debug()
        if isinstance(output.get("ser"), dict):
            ser = dict(output["ser"])
            ser["display_label"] = decision.label
            ser["display_confidence"] = decision.confidence
            ser["stabilized"] = True
            output["ser"] = ser


def _normalize_label(label: str) -> str:
    normalized = str(label or NEUTRAL).strip().upper()
    return normalized if normalized in config.EMOTION_STRESS else NEUTRAL


def _stable_source(source: str) -> str:
    if source == "emotion2vec":
        return "emotion2vec_stabilized"
    if source.endswith("_stabilized"):
        return source
    return source


def _refresh_score_driver(output: dict[str, Any], label: str, confidence: float) -> None:
    drivers = output.get("score_drivers")
    if not isinstance(drivers, dict):
        return
    drivers["raw_emotion_token"] = drivers.get("emotion_token", 0.0)
    drivers["emotion_token"] = config.EMOTION_STRESS.get(label, 0.0) * confidence


def _high_neutral(confidence: float) -> bool:
    return confidence >= config.LIVE_EMOTION_NEUTRAL_CONFIDENCE


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
