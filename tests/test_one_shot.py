from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_perception import config, main
from voice_perception.perception import VoicePerception


class HallucinatingModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args: object, **_kwargs: object) -> list[str]:
        self.calls += 1
        return ["<|en|><|NEUTRAL|><|Speech|><|withitn|>I."]


class FakePerception:
    def __init__(self) -> None:
        self.languages: list[str] = []

    def analyze(self, pcm: np.ndarray, language: str | None = None) -> dict[str, Any]:
        self.languages.append(language or "auto")
        return {
            "transcript": "I need a moment",
            "emotion": "FEARFUL",
            "emotion_confidence": 1.0,
            "events": ["Breath"],
            "silence_ratio": 0.5,
            "inference_ms": 7,
        }


class OneShotEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.original_decode_chunk = main.decode_chunk
        self.original_perception = getattr(main.app.state, "perception", None)
        main.app.state.perception = FakePerception()

    async def asyncTearDown(self) -> None:
        main.decode_chunk = self.original_decode_chunk
        if self.original_perception is None:
            delattr(main.app.state, "perception")
        else:
            main.app.state.perception = self.original_perception

    async def test_classify_endpoint_returns_one_shot_fields(self) -> None:
        main.decode_chunk = _fake_decode(config.SAMPLE_RATE)

        status, payload = await _post_multipart_file(
            b"fake audio", "recording.webm", "audio/webm;codecs=opus"
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["transcript"], "I need a moment")
        self.assertEqual(payload["transcript_partial"], "I need a moment")
        self.assertEqual(payload["emotion"], "FEARFUL")
        self.assertEqual(payload["emotion_confidence"], 1.0)
        self.assertEqual(payload["events"], ["Breath"])
        self.assertGreater(payload["hesitation_score"], 0.5)
        self.assertEqual(payload["audio_samples"], config.SAMPLE_RATE)
        self.assertEqual(payload["classification_mode"], "one_shot")
        self.assertIn("voice_state", payload)
        self.assertIn("acoustic_debug", payload)
        self.assertEqual(payload["acoustic_debug"]["voice_state"], payload["voice_state"])
        self.assertIn("signals", payload)
        self.assertIn("capabilities", payload)
        self.assertTrue(payload["capabilities"]["voice_state_debug_only"])
        self.assertFalse(payload["capabilities"]["acoustic_emotion_labels_supported"])

    async def test_classify_endpoint_rejects_empty_upload(self) -> None:
        status, _ = await _post_multipart_file(b"", "empty.webm", "audio/webm;codecs=opus")

        self.assertEqual(status, 400)

    async def test_classify_endpoint_rejects_undecodable_upload(self) -> None:
        main.decode_chunk = _fake_decode(0)

        status, _ = await _post_multipart_file(
            b"not audio", "broken.webm", "audio/webm;codecs=opus"
        )

        self.assertEqual(status, 422)

    async def test_classify_endpoint_returns_no_speech_for_silent_audio(self) -> None:
        model = HallucinatingModel()
        main.app.state.perception = _perception_with_model(model)
        main.decode_chunk = _fake_decode(config.SAMPLE_RATE * 2, amplitude=0.0)

        status, payload = await _post_multipart_file(
            b"silence", "silence.webm", "audio/webm;codecs=opus"
        )

        self.assertEqual(status, 200)
        self.assertEqual(model.calls, 0)
        self.assertTrue(payload["no_speech"])
        self.assertEqual(payload["transcript"], "")
        self.assertEqual(payload["events"], [])
        self.assertNotEqual(payload["transcript"], "I.")
        self.assertEqual(payload["hesitation_score"], 0.0)
        self.assertEqual(payload["voice_state"]["label"], "no_speech")
        self.assertEqual(payload["acoustic_debug"]["voice_state"]["label"], "no_speech")
        self.assertEqual(payload["signals"]["stress"], 0.0)

    async def test_one_shot_processing_uses_uploaded_audio_once(self) -> None:
        main.decode_chunk = _fake_decode(config.SAMPLE_RATE * 2)

        payload = await main._classify_audio_bytes(b"fake audio", "audio/webm")

        self.assertEqual(payload["transcript"], "I need a moment")
        self.assertEqual(payload["language"], "auto")
        self.assertEqual(payload["inference_ms"], 7)
        self.assertEqual(payload["audio_samples"], config.SAMPLE_RATE * 2)

    async def test_classify_endpoint_accepts_language_form_field(self) -> None:
        fake_perception = FakePerception()
        main.app.state.perception = fake_perception
        main.decode_chunk = _fake_decode(config.SAMPLE_RATE)

        status, payload = await _post_multipart_file(
            b"fake audio",
            "recording.webm",
            "audio/webm;codecs=opus",
            language="de",
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["language"], "de")
        self.assertEqual(fake_perception.languages, ["de"])


def _fake_decode(sample_count: int, amplitude: float = 0.02) -> Callable[[bytes], np.ndarray]:
    def decode(raw_bytes: bytes) -> np.ndarray:
        return np.ones(sample_count, dtype=np.float32) * amplitude

    return decode


def _perception_with_model(model: HallucinatingModel) -> VoicePerception:
    perception = VoicePerception.__new__(VoicePerception)
    perception.model = model
    perception.model_dir = "test"
    perception.resolved_model_dir = None
    perception._lock = threading.Lock()
    perception._input_mode = "numpy"
    perception.load_ms = 0
    return perception


async def _post_multipart_file(
    content: bytes,
    filename: str,
    content_type: str,
    language: str | None = None,
) -> tuple[int, dict[str, Any]]:
    boundary = "voiceperceptiontest"
    body = _multipart_body(boundary, content, filename, content_type, language)
    events: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        events.append(message)

    await main.app(_scope(boundary, len(body)), receive, send)
    status = _response_status(events)
    payload = json.loads(_response_body(events).decode("utf-8"))
    return status, payload


def _multipart_body(
    boundary: str,
    content: bytes,
    filename: str,
    content_type: str,
    language: str | None = None,
) -> bytes:
    file_part = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + content + b"\r\n"
    language_part = b""
    if language is not None:
        language_part = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="language"\r\n\r\n'
            f"{language}\r\n"
        ).encode("utf-8")
    suffix = f"--{boundary}--\r\n".encode("utf-8")
    return file_part + language_part + suffix


def _scope(boundary: str, body_length: int) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/classify",
        "raw_path": b"/classify",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-length", str(body_length).encode("ascii")),
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode("ascii")),
        ],
    }


def _response_status(events: list[dict[str, Any]]) -> int:
    starts = [event for event in events if event["type"] == "http.response.start"]
    return int(starts[0]["status"])


def _response_body(events: list[dict[str, Any]]) -> bytes:
    bodies = [event.get("body", b"") for event in events if event["type"] == "http.response.body"]
    return b"".join(bodies)


if __name__ == "__main__":
    unittest.main()
