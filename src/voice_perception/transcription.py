"""Transcript backends and language routing helpers."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from voice_perception import config
from voice_perception.audio import to_float32_mono

logger = logging.getLogger(__name__)

GERMAN_LANGUAGE = "de"
ENGLISH_LANGUAGE = "en"
AUTO_LANGUAGE = "auto"
SENSEVOICE_SUPPORTED_LANGUAGES = {"zh", "en", "yue", "ja", "ko"}
GERMAN_ALIASES = {"de", "de-de", "de_at", "de-at", "de_ch", "de-ch", "german", "deutsch"}
ENGLISH_ALIASES = {"en", "en-us", "en-gb", "english"}
AUTO_ALIASES = {"", "auto", "default", "detect"}


class FasterWhisperTranscriber:
    """Lazy CTranslate2 Whisper adapter for German transcript."""

    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool | None = None,
        preload: bool | None = None,
    ) -> None:
        self.model_name = model_name or config.GERMAN_ASR_MODEL
        self.enabled = config.GERMAN_ASR_ENABLED if enabled is None else enabled
        self._model: Any | None = None
        self._lock = threading.Lock()
        self.load_ms = 0
        if self.enabled and (config.GERMAN_ASR_PRELOAD if preload is None else preload):
            self._ensure_model()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def analyze(self, pcm_16khz_mono: np.ndarray, language: str = GERMAN_LANGUAGE) -> dict[str, Any]:
        pcm = to_float32_mono(pcm_16khz_mono)
        selected_language = normalize_language_selection(language)
        if not self.enabled:
            return _transcript_status(selected_language, skipped=True, reason="disabled")
        if pcm.size < config.GERMAN_ASR_MIN_CONTEXT_SAMPLES:
            return _transcript_status(selected_language, skipped=True, reason="min_context")
        started = time.perf_counter()
        try:
            text, info = self._transcribe(pcm, selected_language)
        except Exception as exc:
            logger.exception("event=german_asr_failed model=%s error=%s", self.model_name, exc)
            result = _transcript_status(selected_language, skipped=True, reason="error")
            result["transcript_error"] = str(exc)
            result["transcript_latency_ms"] = _elapsed_ms(started)
            return result
        result = _transcript_status(selected_language, skipped=False, reason=None)
        result.update(
            {
                "transcript": text,
                "transcript_latency_ms": _elapsed_ms(started),
                "asr_detected_language": getattr(info, "language", None),
                "asr_detected_language_probability": getattr(info, "language_probability", None),
            }
        )
        logger.info(
            "event=german_asr_transcribed model=%s chars=%d latency_ms=%d",
            self.model_name,
            len(text),
            int(result["transcript_latency_ms"]),
        )
        return result

    def _transcribe(self, pcm: np.ndarray, language: str) -> tuple[str, Any]:
        with self._lock:
            model = self._ensure_model()
            whisper_language = GERMAN_LANGUAGE if language == GERMAN_LANGUAGE else None
            segments, info = model.transcribe(
                np.ascontiguousarray(pcm, dtype=np.float32),
                language=whisper_language,
                beam_size=config.GERMAN_ASR_BEAM_SIZE,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = "".join(segment.text for segment in segments).strip()
        return text, info

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        started = time.perf_counter()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper to enable German ASR.") from exc
        self._model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type=config.GERMAN_ASR_COMPUTE_TYPE,
            cpu_threads=config.GERMAN_ASR_THREADS,
        )
        self.load_ms = _elapsed_ms(started)
        logger.info("event=german_asr_model_loaded model=%s load_ms=%d", self.model_name, self.load_ms)
        return self._model


def normalize_language_selection(value: str | None) -> str:
    """Normalize user or API language selections to the small public set."""

    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized in GERMAN_ALIASES:
        return GERMAN_LANGUAGE
    if normalized in ENGLISH_ALIASES:
        return ENGLISH_LANGUAGE
    if normalized in AUTO_ALIASES:
        return AUTO_LANGUAGE
    if normalized in SENSEVOICE_SUPPORTED_LANGUAGES:
        return normalized
    return AUTO_LANGUAGE


def is_german_language(language: str | None) -> bool:
    return normalize_language_selection(language) == GERMAN_LANGUAGE


def sensevoice_language_for(language: str | None) -> str:
    """Return a SenseVoice language code without sending unsupported German."""

    selected = normalize_language_selection(language)
    if selected in SENSEVOICE_SUPPORTED_LANGUAGES:
        return selected
    configured = normalize_language_selection(config.SENSEVOICE_LANGUAGE)
    if configured in SENSEVOICE_SUPPORTED_LANGUAGES:
        return configured
    return AUTO_LANGUAGE


def transcript_not_called(language: str | None, reason: str) -> dict[str, Any]:
    selected = normalize_language_selection(language)
    backend = "faster_whisper" if selected == GERMAN_LANGUAGE else "sensevoice"
    model = config.GERMAN_ASR_MODEL if selected == GERMAN_LANGUAGE else config.SENSEVOICE_MODEL_DIR
    return {
        "transcript_source": "none",
        "transcript_backend": backend,
        "transcript_model": model,
        "transcript_language": selected,
        "transcript_latency_ms": 0,
        "transcript_skipped": True,
        "transcript_skip_reason": reason,
    }


def sensevoice_transcript_fields(
    transcript: str,
    language: str | None,
    latency_ms: int,
    model_dir: str,
) -> dict[str, Any]:
    selected = normalize_language_selection(language)
    return {
        "transcript_source": "sensevoice",
        "transcript_backend": "sensevoice",
        "transcript_model": model_dir,
        "transcript_language": selected,
        "transcript_latency_ms": int(latency_ms),
        "transcript_skipped": False,
        "transcript_skip_reason": None,
        "sensevoice_transcript": transcript,
        "sensevoice_language": sensevoice_language_for(selected),
    }


def _transcript_status(language: str, skipped: bool, reason: str | None) -> dict[str, Any]:
    return {
        "transcript": "",
        "transcript_source": "none" if skipped else "faster_whisper",
        "transcript_backend": "faster_whisper",
        "transcript_model": config.GERMAN_ASR_MODEL,
        "transcript_language": language,
        "transcript_latency_ms": 0,
        "transcript_skipped": skipped,
        "transcript_skip_reason": reason,
    }


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
