from __future__ import annotations

import sys
import threading
import unittest
from typing import Any
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


class FakeGermanTranscriber:
    def __init__(self) -> None:
        self.calls = 0
        self.languages: list[str] = []

    def analyze(self, _pcm: np.ndarray, language: str = "de") -> dict[str, Any]:
        self.calls += 1
        self.languages.append(language)
        return {
            "transcript": "Guten Tag",
            "transcript_source": "faster_whisper",
            "transcript_backend": "faster_whisper",
            "transcript_model": "test-whisper-base",
            "transcript_language": "de",
            "transcript_latency_ms": 12,
            "transcript_skipped": False,
            "transcript_skip_reason": None,
        }


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
        self.assertEqual(result["transcript_source"], "sensevoice")
        self.assertEqual(result["events"], ["Speech"])
        self.assertFalse(result["no_speech"])

    def test_german_language_uses_faster_whisper_transcript_source(self) -> None:
        model = HallucinatingModel()
        german_transcriber = FakeGermanTranscriber()
        perception = _perception_with_model(model, german_transcriber)
        tone = np.ones(config.SAMPLE_RATE * 2, dtype=np.float32) * 0.02

        result = perception.analyze(tone, language="de")

        self.assertEqual(model.calls, 1)
        self.assertEqual(german_transcriber.calls, 1)
        self.assertEqual(german_transcriber.languages, ["de"])
        self.assertEqual(result["transcript"], "Guten Tag")
        self.assertEqual(result["sensevoice_transcript"], "I.")
        self.assertEqual(result["transcript_source"], "faster_whisper")
        self.assertEqual(result["transcript_language"], "de")
        self.assertEqual(result["events"], ["Speech"])

    def test_default_language_keeps_sensevoice_transcript_path(self) -> None:
        model = HallucinatingModel()
        german_transcriber = FakeGermanTranscriber()
        perception = _perception_with_model(model, german_transcriber)
        tone = np.ones(config.SAMPLE_RATE * 2, dtype=np.float32) * 0.02

        result = perception.analyze(tone, language="auto")

        self.assertEqual(model.calls, 1)
        self.assertEqual(german_transcriber.calls, 0)
        self.assertEqual(result["transcript"], "I.")
        self.assertEqual(result["transcript_source"], "sensevoice")

    def test_silent_german_audio_suppresses_faster_whisper(self) -> None:
        model = HallucinatingModel()
        german_transcriber = FakeGermanTranscriber()
        perception = _perception_with_model(model, german_transcriber)
        silence = np.zeros(config.SAMPLE_RATE * 2, dtype=np.float32)

        result = perception.analyze(silence, language="de")

        self.assertEqual(model.calls, 0)
        self.assertEqual(german_transcriber.calls, 0)
        self.assertTrue(result["no_speech"])
        self.assertEqual(result["transcript"], "")
        self.assertEqual(result["transcript_source"], "none")
        self.assertEqual(result["transcript_skip_reason"], "no_speech")


def _perception_with_model(
    model: HallucinatingModel,
    german_transcriber: FakeGermanTranscriber | None = None,
) -> VoicePerception:
    perception = VoicePerception.__new__(VoicePerception)
    perception.model = model
    perception.model_dir = "test"
    perception.resolved_model_dir = None
    perception._lock = threading.Lock()
    perception._input_mode = "numpy"
    perception.german_transcriber = german_transcriber
    perception.load_ms = 0
    return perception


if __name__ == "__main__":
    unittest.main()
