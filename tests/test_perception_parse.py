from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_perception import config
from voice_perception.perception import VoicePerception, parse_sensevoice_output


class HallucinatingModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args: object, **_kwargs: object) -> list[str]:
        self.calls += 1
        return ["<|en|><|NEUTRAL|><|Speech|><|withitn|>I."]


class PerceptionParseTests(unittest.TestCase):
    def test_unknown_emotion_token_is_diagnosable(self) -> None:
        result = parse_sensevoice_output(
            ["<|en|><|EMO_UNKNOWN|><|Speech|><|withitn|>I need a moment."]
        )

        self.assertEqual(result["emotion"], "NEUTRAL")
        self.assertEqual(result["emotion_confidence"], 0.0)
        self.assertEqual(result["raw_emotion"], "EMO_UNKNOWN")
        self.assertEqual(result["events"], ["Speech"])

    def test_silent_audio_guard_prevents_hallucinated_speech(self) -> None:
        model = HallucinatingModel()
        perception = _perception_with_model(model)
        silence = np.zeros(config.SAMPLE_RATE * 2, dtype=np.float32)

        result = perception.analyze(silence)

        self.assertEqual(model.calls, 0)
        self.assertTrue(result["no_speech"])
        self.assertEqual(result["transcript"], "")
        self.assertEqual(result["events"], [])
        self.assertNotEqual(result["transcript"], "I.")

    def test_non_silent_audio_still_reaches_model(self) -> None:
        model = HallucinatingModel()
        perception = _perception_with_model(model)
        tone = np.ones(config.SAMPLE_RATE, dtype=np.float32) * 0.02

        result = perception.analyze(tone)

        self.assertEqual(model.calls, 1)
        self.assertEqual(result["transcript"], "I.")
        self.assertEqual(result["events"], ["Speech"])
        self.assertFalse(result["no_speech"])


def _perception_with_model(model: HallucinatingModel) -> VoicePerception:
    perception = VoicePerception.__new__(VoicePerception)
    perception.model = model
    perception.model_dir = "test"
    perception.resolved_model_dir = None
    perception._lock = threading.Lock()
    perception._input_mode = "numpy"
    perception.load_ms = 0
    return perception


if __name__ == "__main__":
    unittest.main()
