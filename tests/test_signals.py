from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_perception import config
from voice_perception.audio import load_wav_16khz_mono
from voice_perception.signals import analyze_acoustic_context


class AcousticSignalTests(unittest.TestCase):
    def test_silence_returns_no_speech_and_zero_stress(self) -> None:
        result = analyze_acoustic_context(np.zeros(config.SAMPLE_RATE * 2, dtype=np.float32))

        self.assertEqual(result["voice_state"]["label"], "no_speech")
        self.assertTrue(result["signals"]["no_speech"])
        self.assertEqual(result["signals"]["stress"], 0.0)
        self.assertEqual(result["signals"]["hesitation"], 0.0)
        self.assertEqual(result["signals"]["speech_activity"], 0.0)

    def test_calm_and_anxious_fixtures_are_speech(self) -> None:
        calm = analyze_acoustic_context(load_wav_16khz_mono(ROOT / "tests/fixtures/calm.wav"))
        anxious = analyze_acoustic_context(load_wav_16khz_mono(ROOT / "tests/fixtures/anxious.wav"))

        self.assertFalse(calm["signals"]["no_speech"])
        self.assertFalse(anxious["signals"]["no_speech"])
        self.assertGreater(calm["signals"]["speech_activity"], 0.40)
        self.assertGreater(anxious["signals"]["speech_activity"], 0.25)

    def test_anxious_fixture_has_more_pause_driven_hesitation_than_calm(self) -> None:
        calm = analyze_acoustic_context(load_wav_16khz_mono(ROOT / "tests/fixtures/calm.wav"))
        anxious = analyze_acoustic_context(load_wav_16khz_mono(ROOT / "tests/fixtures/anxious.wav"))

        self.assertGreater(anxious["debug_features"]["pause_ratio"], calm["debug_features"]["pause_ratio"])
        self.assertGreater(
            anxious["signals"]["hesitation"],
            calm["signals"]["hesitation"] + 0.10,
        )
        self.assertGreater(
            anxious["score_drivers"]["pause_silence"],
            calm["score_drivers"]["pause_silence"],
        )


if __name__ == "__main__":
    unittest.main()
