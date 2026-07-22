from __future__ import annotations

import importlib
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_perception import config


class RuntimeDependencyTests(unittest.TestCase):
    def test_requirements_match_pyproject_dependencies(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = list(pyproject["project"]["dependencies"])

        self.assertEqual(_requirements_lines(), dependencies)

    def test_german_asr_imports_available_when_enabled(self) -> None:
        if not config.GERMAN_ASR_ENABLED:
            self.skipTest("German ASR is disabled")

        self.assertIsNotNone(importlib.import_module("faster_whisper"))
        self.assertIsNotNone(importlib.import_module("ctranslate2"))


def _requirements_lines() -> list[str]:
    lines: list[str] = []
    for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


if __name__ == "__main__":
    unittest.main()
