from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_perception.fusion import HesitationScorer, compute_hesitation_score


class FusionTests(unittest.TestCase):
    def test_calm_baseline_under_015(self) -> None:
        result = {
            "emotion": "NEUTRAL",
            "emotion_confidence": 1.0,
            "events": [],
            "silence_ratio": 0.0,
        }
        self.assertLess(compute_hesitation_score(result), 0.15)

    def test_fearful_breath_silence_over_06(self) -> None:
        result = {
            "emotion": "FEARFUL",
            "emotion_confidence": 1.0,
            "events": ["Breath"],
            "silence_ratio": 0.8,
        }
        self.assertGreater(compute_hesitation_score(result), 0.6)

    def test_ema_smooths_spike_between_calm_chunks(self) -> None:
        scorer = HesitationScorer()
        calm = {
            "emotion": "NEUTRAL",
            "emotion_confidence": 1.0,
            "events": [],
            "silence_ratio": 0.0,
        }
        spike = {
            "emotion": "FEARFUL",
            "emotion_confidence": 1.0,
            "events": ["Breath"],
            "silence_ratio": 0.8,
        }
        first = scorer.update(calm)
        smoothed_spike = scorer.update(spike)
        recovered = scorer.update(calm)
        raw_spike = compute_hesitation_score(spike)

        self.assertLess(first, 0.15)
        self.assertLess(smoothed_spike, raw_spike)
        self.assertGreater(smoothed_spike, first)
        self.assertGreater(recovered, first)
        self.assertLess(recovered, smoothed_spike)


if __name__ == "__main__":
    unittest.main()
