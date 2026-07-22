from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_perception import config
from voice_perception.emotion import (
    Emotion2VecClassifier,
    map_emotion2vec_label,
    parse_emotion2vec_output,
)


class FakeSerModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **_kwargs: object) -> list[dict[str, object]]:
        self.calls += 1
        return [{"labels": ["neutral", "fearful"], "scores": [0.10, 0.82]}]


class Emotion2VecTests(unittest.TestCase):
    def test_label_mapping_to_public_emotions(self) -> None:
        cases = {
            "angry": "ANGRY",
            "disgusted": "DISGUSTED",
            "fearful": "FEARFUL",
            "happy": "HAPPY",
            "neutral": "NEUTRAL",
            "sad": "SAD",
            "surprised": "SURPRISED",
            "生气/angry": "ANGRY",
        }
        for raw_label, expected in cases.items():
            with self.subTest(raw_label=raw_label):
                self.assertEqual(map_emotion2vec_label(raw_label, 0.77), (expected, 0.77))

    def test_unknown_and_other_map_to_neutral_zero_confidence(self) -> None:
        for raw_label in ("other", "unknown", "<unk>", "not-a-real-label"):
            with self.subTest(raw_label=raw_label):
                self.assertEqual(map_emotion2vec_label(raw_label, 0.99), ("NEUTRAL", 0.0))

    def test_parser_selects_top_scored_label(self) -> None:
        parsed = parse_emotion2vec_output(
            [{"labels": ["neutral", "fearful", "sad"], "scores": [0.1, 0.7, 0.2]}]
        )

        self.assertEqual(parsed["label"], "FEARFUL")
        self.assertAlmostEqual(parsed["confidence"], 0.7)
        self.assertEqual(parsed["raw_label"], "fearful")

    def test_no_speech_guard_does_not_call_ser_model(self) -> None:
        model = FakeSerModel()
        classifier = Emotion2VecClassifier(model_dir="test", enabled=True, preload=False)
        classifier._model = model
        silence = np.zeros(config.SAMPLE_RATE * 2, dtype=np.float32)

        result = classifier.analyze(silence)

        self.assertEqual(model.calls, 0)
        self.assertTrue(result["skipped"])
        self.assertEqual(result["reason"], "no_speech")
        self.assertEqual(result["label"], "NEUTRAL")
        self.assertEqual(result["confidence"], 0.0)

    def test_speech_context_calls_ser_model(self) -> None:
        model = FakeSerModel()
        classifier = Emotion2VecClassifier(model_dir="test", enabled=True, preload=False)
        classifier._model = model
        speech_like = np.ones(config.SAMPLE_RATE * 2, dtype=np.float32) * 0.02

        result = classifier.analyze(speech_like)

        self.assertEqual(model.calls, 1)
        self.assertEqual(result["label"], "FEARFUL")
        self.assertAlmostEqual(result["confidence"], 0.82)


if __name__ == "__main__":
    unittest.main()
