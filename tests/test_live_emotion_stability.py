from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_perception.emotion_stability import LiveEmotionStabilizer
from voice_perception.session import SessionState


class LiveEmotionStabilityTests(unittest.TestCase):
    def test_happy_followed_by_brief_neutral_keeps_displayed_happy(self) -> None:
        stabilizer = LiveEmotionStabilizer()
        now = _time()

        first = stabilizer.update("HAPPY", 0.82, now=now)
        second = stabilizer.update("NEUTRAL", 0.91, now=now + timedelta(seconds=1))

        self.assertEqual(first.label, "HAPPY")
        self.assertEqual(second.label, "HAPPY")
        self.assertEqual(second.raw_label, "NEUTRAL")
        self.assertEqual(second.decision, "hold_non_neutral")

    def test_repeated_high_confidence_neutral_switches_display_to_neutral(self) -> None:
        stabilizer = LiveEmotionStabilizer()
        now = _time()

        stabilizer.update("HAPPY", 0.82, now=now)
        held = stabilizer.update("NEUTRAL", 0.91, now=now + timedelta(seconds=1))
        switched = stabilizer.update("NEUTRAL", 0.94, now=now + timedelta(seconds=2))

        self.assertEqual(held.label, "HAPPY")
        self.assertEqual(switched.label, "NEUTRAL")
        self.assertEqual(switched.confidence, 0.94)
        self.assertEqual(switched.decision, "switch_repeated_neutral")

    def test_no_speech_resets_to_idle_neutral_safely(self) -> None:
        stabilizer = LiveEmotionStabilizer()
        now = _time()

        stabilizer.update("HAPPY", 0.82, now=now)
        reset = stabilizer.update("HAPPY", 0.80, no_speech=True, now=now + timedelta(seconds=1))

        self.assertEqual(reset.label, "NEUTRAL")
        self.assertEqual(reset.confidence, 0.0)
        self.assertEqual(reset.decision, "reset_no_speech")
        self.assertIsNone(reset.pending_label)

    def test_sustained_new_emotion_switches_after_required_updates(self) -> None:
        stabilizer = LiveEmotionStabilizer()
        now = _time()

        stabilizer.update("HAPPY", 0.82, now=now)
        first_sad = stabilizer.update("SAD", 0.78, now=now + timedelta(seconds=1))
        second_sad = stabilizer.update("SAD", 0.80, now=now + timedelta(seconds=2))

        self.assertEqual(first_sad.label, "HAPPY")
        self.assertEqual(first_sad.pending_label, "SAD")
        self.assertEqual(second_sad.label, "SAD")
        self.assertEqual(second_sad.decision, "switch_sustained")

    def test_session_response_uses_stable_primary_and_exposes_raw_live_fields(self) -> None:
        state = SessionState(session_id="test")

        state.update(_result("HAPPY", 0.82))
        state.update(_result("NEUTRAL", 0.91))
        response = state.to_response()

        self.assertEqual(response["emotion"], "HAPPY")
        self.assertEqual(response["live_raw_emotion"], "NEUTRAL")
        self.assertEqual(response["live_raw_emotion_confidence"], 0.91)
        self.assertEqual(response["live_stabilized_emotion"]["label"], "HAPPY")
        self.assertEqual(response["ser"]["display_label"], "HAPPY")


def _result(label: str, confidence: float) -> dict[str, Any]:
    return {
        "transcript": "hello",
        "emotion": label,
        "emotion_confidence": confidence,
        "emotion_source": "emotion2vec",
        "events": ["Speech"],
        "silence_ratio": 0.0,
        "inference_ms": 0,
        "no_speech": False,
        "ser": {
            "enabled": True,
            "skipped": False,
            "label": label,
            "confidence": confidence,
            "raw_label": label.lower(),
            "license_caveat": "hackathon_demo_license_pending",
        },
        "voice_state": {},
        "signals": {},
        "signal_events": [],
        "score_drivers": {"emotion_token": 0.0},
        "debug_features": {},
    }


def _time() -> datetime:
    return datetime(2026, 7, 22, tzinfo=timezone.utc)


if __name__ == "__main__":
    unittest.main()
