from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StaticUiCopyTests(unittest.TestCase):
    def test_ui_does_not_present_acoustic_voice_state_as_product_label(self) -> None:
        html = (ROOT / "static/index.html").read_text(encoding="utf-8")

        self.assertIn("Acoustic guardrails", html)
        self.assertIn("Do not use this as an emotion label", html)
        self.assertIn("Emotion2Vec+ is the primary exact-emotion model", html)
        self.assertNotIn(">Voice state<", html)
        self.assertNotIn("acoustic voice state", html)

    def test_ui_exposes_transcript_language_dropdown(self) -> None:
        html = (ROOT / "static/index.html").read_text(encoding="utf-8")

        self.assertIn('id="languageSelect"', html)
        self.assertIn('<option value="auto">Auto</option>', html)
        self.assertIn('<option value="en">English</option>', html)
        self.assertIn('<option value="de">German</option>', html)
        self.assertIn("formData.append('language', selectedLanguage())", html)


if __name__ == "__main__":
    unittest.main()
