"""Hesitation score fusion logic."""

from __future__ import annotations

from typing import Any, Iterable

from voice_perception import config


def clip(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def compute_hesitation_score(perception_result: dict[str, Any]) -> float:
    emotion = str(perception_result.get("emotion", "NEUTRAL")).upper()
    confidence = clip(float(perception_result.get("emotion_confidence", 0.0)))
    events = perception_result.get("events", [])
    silence_ratio = clip(float(perception_result.get("silence_ratio", 0.0)))

    emotion_stress = config.EMOTION_STRESS.get(emotion, config.EMOTION_STRESS["NEUTRAL"])
    emotion_stress *= confidence
    event_stress = _event_stress(events if isinstance(events, list) else [])
    silence_stress = silence_ratio * config.SILENCE_STRESS_MULTIPLIER

    return clip(
        config.HESITATION_EMOTION_WEIGHT * emotion_stress
        + config.HESITATION_EVENT_WEIGHT * event_stress
        + config.HESITATION_SILENCE_WEIGHT * silence_stress
    )


class HesitationScorer:
    """Smooth hesitation scores with an exponential moving average."""

    def __init__(self, alpha: float = config.HESITATION_EMA_ALPHA) -> None:
        self.alpha = clip(alpha)
        self._score: float | None = None

    @property
    def score(self) -> float:
        return 0.0 if self._score is None else self._score

    def update(self, perception_result: dict[str, Any]) -> float:
        raw_score = compute_hesitation_score(perception_result)
        if self._score is None:
            self._score = raw_score
        else:
            self._score = self.alpha * raw_score + (1.0 - self.alpha) * self._score
        self._score = clip(self._score)
        return self._score


def _event_stress(events: Iterable[Any]) -> float:
    normalized = {str(event).strip().lower() for event in events}
    total = 0.0
    for event, weight in config.EVENT_STRESS.items():
        if event.lower() in normalized:
            total += weight
    return clip(total, 0.0, config.EVENT_STRESS_MAX)
