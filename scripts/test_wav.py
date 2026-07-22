#!/usr/bin/env python3
"""Run the real perception and fusion pipeline against WAV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_perception import config
from voice_perception.audio import load_wav_16khz_mono
from voice_perception.fusion import HesitationScorer
from voice_perception.perception import VoicePerception
from voice_perception.session import RollingAudioBuffer


def iter_chunks(pcm, chunk_samples: int = config.SAMPLE_RATE) -> Iterable[tuple[int, object]]:
    for index, start in enumerate(range(0, len(pcm), chunk_samples), start=1):
        yield index, pcm[start : start + chunk_samples]


def analyze_file(path: Path, perception: VoicePerception) -> float:
    pcm = load_wav_16khz_mono(path)
    scorer = HesitationScorer()
    buffer = RollingAudioBuffer()
    final_score = 0.0
    print(f"file: {path}")
    print("chunk emotion    speech    events                 silence infer_ms score")
    print("----- ---------- ------------ ---------------------- ------- -------- -----")
    for chunk_index, chunk in iter_chunks(pcm):
        window = buffer.append(chunk)
        if window is None:
            print_buffered_row(chunk_index, buffer.samples)
            continue
        result = perception.analyze(window)
        final_score = scorer.update(result)
        print_row(chunk_index, result, final_score)
    print()
    return final_score


def print_row(chunk_index: int, result: dict[str, object], score: float) -> None:
    events = ",".join(result.get("events", [])) or "-"  # type: ignore[arg-type]
    signals = result.get("signals", {})
    label = "no_speech" if signals.get("no_speech") else "speech"
    print(
        f"{chunk_index:>5} {result.get('emotion', 'NEUTRAL'):<10} "
        f"{label:<12.12} {events:<22.22} {result.get('silence_ratio', 0.0):>7.2f} "
        f"{result.get('inference_ms', 0):>8} {score:>5.2f}"
    )


def print_buffered_row(chunk_index: int, buffered_samples: int) -> None:
    buffered_seconds = buffered_samples / config.SAMPLE_RATE
    print(f"{chunk_index:>5} {'BUFFERING':<10} {'-':<12} {'-':<22} {buffered_seconds:>7.2f} {'-':>8} {'-':>5}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WAV files in 1 second chunks.")
    parser.add_argument("wav_path", nargs="?", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("CALM_WAV", "ANXIOUS_WAV"))
    args = parser.parse_args()
    if not args.wav_path and not args.compare:
        parser.error("provide a WAV path or --compare CALM_WAV ANXIOUS_WAV")
    perception = VoicePerception()
    if args.compare:
        calm_score = analyze_file(args.compare[0], perception)
        anxious_score = analyze_file(args.compare[1], perception)
        if anxious_score <= calm_score:
            raise SystemExit("expected anxious score to be greater than calm score")
        print(f"compare ok: anxious {anxious_score:.2f} > calm {calm_score:.2f}")
        return
    analyze_file(args.wav_path, perception)


if __name__ == "__main__":
    main()
