"""FastAPI app for the Voice Perception Service."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from voice_perception import config
from voice_perception.audio import decode_chunk
from voice_perception.fusion import compute_hesitation_score
from voice_perception.perception import VoicePerception
from voice_perception.session import SessionManager
from voice_perception.signals import analyze_acoustic_context
from voice_perception.transcription import normalize_language_selection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    start = time.perf_counter()
    logger.info("event=startup_model_load model_dir=%s", config.SENSEVOICE_MODEL_DIR)
    app.state.perception = VoicePerception()
    load_ms = int((time.perf_counter() - start) * 1000)
    app.state.expiry_task = asyncio.create_task(manager.expire_sessions_loop())
    logger.info("event=startup_ready model_loaded=true model_load_ms=%d", load_ms)
    try:
        yield
    finally:
        app.state.expiry_task.cancel()
        await _cancel_task(app.state.expiry_task)
        logger.info("event=shutdown_complete")


app = FastAPI(title="Voice Perception Service", version="0.1.0", lifespan=lifespan)

# Hackathon-only: allow any local demo page or partner service to query state.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/session/start")
async def start_session(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, str]:
    language = normalize_language_selection((payload or {}).get("language"))
    return {"session_id": manager.create_session(language=language)}


@app.websocket("/audio/{session_id}")
async def audio_socket(websocket: WebSocket, session_id: str) -> None:
    state = manager.get(session_id)
    if state is None:
        await websocket.close(code=1008, reason="unknown session")
        return
    await websocket.accept()
    logger.info("event=websocket_connected session_id=%s", session_id)
    try:
        await _receive_audio_loop(websocket, session_id)
    except WebSocketDisconnect:
        logger.info("event=websocket_disconnected session_id=%s", session_id)


@app.get("/state/{session_id}")
async def get_state(session_id: str) -> dict[str, Any]:
    state = manager.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")
    return state.to_response()


@app.post("/session/{session_id}/end")
async def end_session(session_id: str) -> dict[str, bool]:
    manager.end(session_id)
    return {"ok": True}


@app.post("/classify")
async def classify_audio(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> dict[str, Any]:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="audio upload is empty")
    return await _classify_audio_bytes(raw_bytes, file.content_type or "unknown", language)


@app.get("/health")
async def health() -> dict[str, Any]:
    perception = getattr(app.state, "perception", None)
    ser = getattr(perception, "ser", None)
    german_asr = getattr(perception, "german_transcriber", None)
    return {
        "status": "ok",
        "model_loaded": perception is not None,
        "ser_enabled": config.SER_ENABLED,
        "ser_loaded": bool(getattr(ser, "loaded", False)),
        "german_asr_enabled": config.GERMAN_ASR_ENABLED,
        "german_asr_loaded": bool(getattr(german_asr, "loaded", False)),
    }


async def _classify_audio_bytes(
    raw_bytes: bytes,
    content_type: str = "unknown",
    language: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    selected_language = normalize_language_selection(language)
    pcm = decode_chunk(raw_bytes)
    if pcm.size == 0:
        raise HTTPException(status_code=422, detail="audio upload could not be decoded")
    result = await asyncio.to_thread(app.state.perception.analyze, pcm, selected_language)
    acoustic = result if "signals" in result else analyze_acoustic_context(pcm)
    latency_ms = int((time.perf_counter() - started) * 1000)
    response = _one_shot_response(result, latency_ms, int(pcm.size), acoustic, selected_language)
    logger.info(
        "event=one_shot_classified content_type=%s bytes=%d samples=%d "
        "latency_ms=%d inference_ms=%d emotion=%s hesitation=%.3f",
        content_type,
        len(raw_bytes),
        pcm.size,
        latency_ms,
        result.get("inference_ms", 0),
        response["emotion"],
        response["hesitation_score"],
    )
    return response


def _one_shot_response(
    result: dict[str, Any],
    latency_ms: int,
    audio_samples: int,
    acoustic: dict[str, Any],
    language: str,
) -> dict[str, Any]:
    transcript = str(result.get("transcript", ""))
    response = {
        "transcript": transcript,
        "transcript_partial": transcript,
        "language": language,
        "emotion": result.get("emotion", "NEUTRAL"),
        "emotion_confidence": float(result.get("emotion_confidence", 0.0)),
        "events": list(result.get("events", [])),
        "hesitation_score": 0.0 if result.get("no_speech") else compute_hesitation_score(result),
        "silence_ratio": float(result.get("silence_ratio", 0.0)),
        "no_speech": bool(result.get("no_speech", False)),
        "inference_ms": int(result.get("inference_ms", 0)),
        "latency_ms": latency_ms,
        "audio_samples": audio_samples,
        "classification_mode": "one_shot",
    }
    response.update(_additive_one_shot_fields(result, acoustic))
    return response


def _additive_one_shot_fields(result: dict[str, Any], acoustic: dict[str, Any]) -> dict[str, Any]:
    voice_state = dict(acoustic.get("voice_state", {}))
    fields = {
        "voice_state": voice_state,
        "acoustic_debug": {"voice_state": voice_state},
        "signals": dict(acoustic.get("signals", {})),
        "signal_events": list(acoustic.get("signal_events", [])),
        "score_drivers": dict(acoustic.get("score_drivers", {})),
        "debug_features": dict(acoustic.get("debug_features", {})),
    }
    for key in _MODEL_DEBUG_KEYS:
        if key in result:
            fields[key] = result[key]
    if "capabilities" not in fields:
        fields["capabilities"] = _default_capabilities()
    return fields


def _ack_signal_fields(state_response: dict[str, Any]) -> dict[str, Any]:
    return {
        "voice_state": state_response.get("voice_state"),
        "acoustic_debug": state_response.get("acoustic_debug"),
        "signals": state_response.get("signals"),
        "signal_events": state_response.get("signal_events", []),
    }


_MODEL_DEBUG_KEYS = (
    "emotion_source",
    "raw_emotion",
    "raw_ser_label",
    "sensevoice_emotion",
    "sensevoice_emotion_confidence",
    "sensevoice_raw_emotion",
    "ser",
    "transcript_source",
    "transcript_backend",
    "transcript_model",
    "transcript_language",
    "transcript_latency_ms",
    "transcript_skipped",
    "transcript_skip_reason",
    "transcript_error",
    "sensevoice_transcript",
    "sensevoice_language",
    "asr_detected_language",
    "asr_detected_language_probability",
    "capabilities",
)


def _default_capabilities() -> dict[str, Any]:
    return {
        "emotion_labels_supported": config.SER_ENABLED,
        "emotion_probabilities_calibrated": False,
        "event_labels_supported": True,
        "transcript_supported": True,
        "german_transcript_supported": config.GERMAN_ASR_ENABLED,
        "german_transcript_model": config.GERMAN_ASR_MODEL if config.GERMAN_ASR_ENABLED else None,
        "voice_state_supported": True,
        "voice_state_debug_only": True,
        "acoustic_emotion_labels_supported": False,
        "score_drivers_supported": True,
        "ser_license_caveat": config.SER_LICENSE_CAVEAT if config.SER_ENABLED else None,
    }


async def _receive_audio_loop(websocket: WebSocket, session_id: str) -> None:
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect()
        raw_bytes = message.get("bytes")
        if raw_bytes is None:
            continue
        await _process_audio_bytes(websocket, session_id, raw_bytes)


async def _process_audio_bytes(websocket: WebSocket, session_id: str, raw_bytes: bytes) -> None:
    started = time.perf_counter()
    state = manager.get(session_id)
    if state is None:
        await websocket.close(code=1008, reason="session ended")
        return
    pcm = state.decoder.decode(raw_bytes)
    window = state.ingest_audio(pcm)
    if window is None or window.size < config.MIN_INFERENCE_SAMPLES:
        await _send_ack(websocket, started, buffered=True, state=state)
        return
    result = await asyncio.to_thread(app.state.perception.analyze, window, state.language)
    if result.get("no_speech"):
        updated_state = manager.mark_no_speech(session_id, result)
        await _send_ack(websocket, started, buffered=False, no_speech=True, state=updated_state)
        return
    updated_state = manager.update(session_id, result)
    latency_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "event=chunk_processed session_id=%s bytes=%d chunk_samples=%d "
        "window_samples=%d container=%s latency_ms=%d inference_ms=%d "
        "emotion=%s hesitation=%.3f",
        session_id,
        len(raw_bytes),
        pcm.size,
        window.size,
        state.decoder.last_format,
        latency_ms,
        result.get("inference_ms", 0),
        result.get("emotion", "NEUTRAL"),
        state.hesitation_score,
    )
    await _send_ack(websocket, started, buffered=False, state=updated_state)


async def _send_ack(
    websocket: WebSocket,
    started: float,
    buffered: bool,
    no_speech: bool = False,
    state: Any | None = None,
) -> None:
    latency_ms = int((time.perf_counter() - started) * 1000)
    message: dict[str, Any] = {
        "chunk_processed": True,
        "latency_ms": latency_ms,
        "buffered": buffered,
        "no_speech": no_speech,
    }
    if state is not None:
        signal_fields = _ack_signal_fields(state.to_response())
        message.update(signal_fields)
        signals = signal_fields.get("signals") or {}
        message["no_speech"] = no_speech or bool(signals.get("no_speech", False))
    await websocket.send_json(message)


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        return


_static_dir = Path(__file__).resolve().parents[2] / "static"
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
