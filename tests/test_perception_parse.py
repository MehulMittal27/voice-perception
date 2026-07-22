from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_perception.perception import parse_sensevoice_output


class PerceptionParseTests(unittest.TestCase):
    def test_unknown_emotion_token_is_diagnosable(self) -> None:
        result = parse_sensevoice_output(
            ["<|en|><|EMO_UNKNOWN|><|Speech|><|withitn|>I need a moment."]
        )

        self.assertEqual(result["emotion"], "NEUTRAL")
        self.assertEqual(result["emotion_confidence"], 0.0)
        self.assertEqual(result["raw_emotion"], "EMO_UNKNOWN")
        self.assertEqual(result["events"], ["Speech"])


if __name__ == "__main__":
    unittest.main()
