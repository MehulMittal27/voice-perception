from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_perception import config
from voice_perception import main
from voice_perception.session import SessionState


class FakeDecoder:
    last_format = "test/raw"

    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = list(chunks)

    def decode(self, raw_bytes: bytes) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return self._chunks.pop(0)


class FakePerception:
    def __init__(self, no_speech: bool = False) -> None:
        self.window_sizes: list[int] = []
        self.no_speech = no_speech

    def analyze(self, pcm: np.ndarray) -> dict[str, Any]:
        self.window_sizes.append(int(pcm.size))
        if self.no_speech:
            return _no_speech_result()
        return {
            "transcript": f"rolling window {pcm.size}",
            "emotion": "NEUTRAL",
            "emotion_confidence": 1.0,
            "events": ["Speech"],
            "silence_ratio": 0.0,
            "inference_ms": 0,
            "no_speech": False,
        }


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send_json(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


class LiveBufferingTests(unittest.TestCase):
    def test_session_waits_for_context_and_replaces_transcript(self) -> None:
        state = SessionState(session_id="test")
        one_second = np.ones(config.SAMPLE_RATE, dtype=np.float32) * 0.01

        self.assertIsNone(state.ingest_audio(one_second))
        first_window = state.ingest_audio(one_second)
        self.assertIsNotNone(first_window)
        self.assertGreaterEqual(first_window.size, config.LIVE_MIN_CONTEXT_SAMPLES)

        state.update(_result("I am not sure"))
        next_window = state.ingest_audio(one_second)
        self.assertIsNotNone(next_window)
        state.update(_result("I am not sure wait please"))

        self.assertEqual(state.transcript_partial, "I am not sure wait please")
        self.assertEqual(state.transcript_partial.count("I am not sure"), 1)

    def test_session_keeps_only_rolling_audio_context(self) -> None:
        state = SessionState(session_id="test")
        one_second = np.ones(config.SAMPLE_RATE, dtype=np.float32) * 0.01

        for _ in range(8):
            state.ingest_audio(one_second)

        self.assertLessEqual(state.audio_buffer.samples, config.LIVE_ROLLING_SAMPLES)


class MainChunkPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_processing_analyzes_rolling_window(self) -> None:
        session_id = main.manager.create_session()
        state = main.manager.get(session_id)
        assert state is not None
        chunks = [np.ones(config.SAMPLE_RATE, dtype=np.float32) for _ in range(3)]
        state.decoder = FakeDecoder(chunks)  # type: ignore[assignment]
        fake_perception = FakePerception()
        main.app.state.perception = fake_perception
        websocket = FakeWebSocket()

        try:
            await main._process_audio_bytes(websocket, session_id, b"one")
            await main._process_audio_bytes(websocket, session_id, b"two")
            await main._process_audio_bytes(websocket, session_id, b"three")
        finally:
            main.manager.end(session_id)

        self.assertEqual(
            fake_perception.window_sizes,
            [config.SAMPLE_RATE * 2, config.SAMPLE_RATE * 3],
        )
        self.assertTrue(websocket.messages[0].get("buffered"))
        self.assertEqual(websocket.messages[-1].get("chunk_processed"), True)

    async def test_websocket_no_speech_result_clears_live_state(self) -> None:
        session_id = main.manager.create_session()
        state = main.manager.get(session_id)
        assert state is not None
        chunks = [np.zeros(config.SAMPLE_RATE, dtype=np.float32) for _ in range(2)]
        state.decoder = FakeDecoder(chunks)  # type: ignore[assignment]
        main.app.state.perception = FakePerception(no_speech=True)
        websocket = FakeWebSocket()

        try:
            await main._process_audio_bytes(websocket, session_id, b"one")
            await main._process_audio_bytes(websocket, session_id, b"two")
            response = state.to_response()
        finally:
            main.manager.end(session_id)

        self.assertTrue(response["no_speech"])
        self.assertEqual(response["transcript_partial"], "")
        self.assertEqual(response["events"], [])
        self.assertEqual(response["hesitation_score"], 0.0)
        self.assertEqual(response["chunks_processed"], 0)
        self.assertTrue(websocket.messages[-1].get("no_speech"))


def _result(transcript: str) -> dict[str, Any]:
    return {
        "transcript": transcript,
        "emotion": "NEUTRAL",
        "emotion_confidence": 1.0,
        "events": ["Speech"],
        "silence_ratio": 0.0,
        "inference_ms": 0,
        "no_speech": False,
    }


def _no_speech_result() -> dict[str, Any]:
    return {
        "transcript": "",
        "emotion": "NEUTRAL",
        "emotion_confidence": 0.0,
        "events": [],
        "silence_ratio": 1.0,
        "inference_ms": 0,
        "no_speech": True,
    }


if __name__ == "__main__":
    unittest.main()
