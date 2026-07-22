"""Emotion2Vec+ speech emotion recognition adapter.

Emotion2Vec+ is used as an experimental hackathon SER lane while license
terms are verified for any broader shipping use. It is always gated by the
existing no-speech guard so silence does not emit emotion labels.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import numpy as np

from voice_perception import config
from voice_perception.audio import SpeechActivity, analyze_speech_activity, to_float32_mono

logger = logging.getLogger(__name__)

SER_LABEL_MAP = {
    "ANGRY": "ANGRY",
    "ANGER": "ANGRY",
    "生气": "ANGRY",
    "DISGUSTED": "DISGUSTED",
    "DISGUST": "DISGUSTED",
    "厌恶": "DISGUSTED",
    "FEARFUL": "FEARFUL",
    "FEAR": "FEARFUL",
    "AFRAID": "FEARFUL",
    "恐惧": "FEARFUL",
    "HAPPY": "HAPPY",
    "HAPPINESS": "HAPPY",
    "JOY": "HAPPY",
    "开心": "HAPPY",
    "NEUTRAL": "NEUTRAL",
    "中立": "NEUTRAL",
    "SAD": "SAD",
    "SADNESS": "SAD",
    "难过": "SAD",
    "SURPRISED": "SURPRISED",
    "SURPRISE": "SURPRISED",
    "吃惊": "SURPRISED",
}
UNKNOWN_SER_LABELS = {
    "OTHER",
    "OTHERS",
    "UNKNOWN",
    "UNK",
    "<UNK>",
    "其他",
    "其它",
    "未知",
    "",
}


class Emotion2VecClassifier:
    """Lazy or eager FunASR Emotion2Vec+ base classifier."""

    def __init__(
        self,
        model_dir: str | None = None,
        enabled: bool | None = None,
        preload: bool | None = None,
    ) -> None:
        self.model_dir = model_dir or config.SER_MODEL_DIR
        self.enabled = config.SER_ENABLED if enabled is None else enabled
        self._model: Any | None = None
        self._lock = threading.Lock()
        self.load_ms = 0
        if self.enabled and (config.SER_PRELOAD if preload is None else preload):
            self._ensure_model()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def analyze(
        self,
        pcm_16khz_mono: np.ndarray,
        activity: SpeechActivity | None = None,
    ) -> dict[str, Any]:
        pcm = to_float32_mono(pcm_16khz_mono)
        if not self.enabled:
            return _ser_status(enabled=False, skipped=True, reason="disabled")
        activity = activity or analyze_speech_activity(pcm)
        if pcm.size < config.SER_MIN_CONTEXT_SAMPLES:
            return _ser_status(enabled=True, skipped=True, reason="min_context")
        if not activity.has_speech:
            return _ser_status(enabled=True, skipped=True, reason="no_speech")
        return self._analyze_speech(pcm)

    def _analyze_speech(self, pcm: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            with self._lock:
                model = self._ensure_model()
                raw_output = model.generate(
                    input=np.ascontiguousarray(pcm, dtype=np.float32),
                    granularity="utterance",
                    extract_embedding=False,
                )
        except Exception as exc:
            logger.exception("event=ser_inference_failed model=%s error=%s", self.model_dir, exc)
            result = _ser_status(enabled=True, skipped=False, reason="error")
            result["error"] = str(exc)
            result["latency_ms"] = _elapsed_ms(started)
            return result
        result = parse_emotion2vec_output(raw_output)
        result.update(
            {
                "enabled": True,
                "skipped": False,
                "reason": None,
                "model": self.model_dir,
                "latency_ms": _elapsed_ms(started),
                "window_seconds": round(pcm.size / float(config.SAMPLE_RATE), 3),
                "experimental": True,
                "license_caveat": config.SER_LICENSE_CAVEAT,
            }
        )
        _log_mapped_result(result)
        return result

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        started = time.perf_counter()
        _configure_torch_threads()
        os.environ.setdefault("MODELSCOPE_CACHE", config.SER_CACHE_DIR)
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError("Install funasr and torchaudio to enable Emotion2Vec+ SER.") from exc
        try:
            self._model = AutoModel(
                model=self.model_dir,
                disable_update=True,
                disable_pbar=True,
                ncpu=config.SER_THREADS,
            )
        except TypeError:
            self._model = AutoModel(model=self.model_dir, disable_update=True)
        self.load_ms = _elapsed_ms(started)
        logger.info(
            "event=ser_model_loaded model=%s load_ms=%d license_caveat=%s",
            self.model_dir,
            self.load_ms,
            config.SER_LICENSE_CAVEAT,
        )
        return self._model


def parse_emotion2vec_output(raw_output: Any) -> dict[str, Any]:
    """Parse FunASR Emotion2Vec output into the app emotion contract."""

    record = _first_record(raw_output)
    raw_label, raw_confidence, raw_scores = _extract_top_label(record)
    emotion, confidence = map_emotion2vec_label(raw_label, raw_confidence)
    return {
        "label": emotion,
        "confidence": confidence,
        "raw_label": raw_label,
        "raw_confidence": raw_confidence,
        "scores": raw_scores,
        "raw_output_type": type(raw_output).__name__,
    }


def map_emotion2vec_label(raw_label: Any, confidence: float | None = None) -> tuple[str, float]:
    """Map Emotion2Vec labels to the public uppercase emotion labels."""

    for normalized in _label_variants(raw_label):
        if normalized in UNKNOWN_SER_LABELS:
            return "NEUTRAL", 0.0
        mapped = SER_LABEL_MAP.get(normalized)
        if mapped is not None:
            return mapped, _clip(1.0 if confidence is None else confidence)
    logger.info("event=ser_unknown_label raw_label=%s", raw_label)
    return "NEUTRAL", 0.0


def _extract_top_label(record: Any) -> tuple[str | None, float | None, dict[str, float]]:
    if isinstance(record, dict):
        label, confidence, scores = _extract_from_dict(record)
        return label, confidence, scores
    if isinstance(record, str):
        return record, None, {}
    return None, None, {}


def _extract_from_dict(record: dict[str, Any]) -> tuple[str | None, float | None, dict[str, float]]:
    labels = _as_list(record.get("labels") or record.get("label") or record.get("emotion"))
    scores = _float_list(record.get("scores") or record.get("score") or record.get("confidence"))
    if labels and scores and len(labels) == len(scores):
        best = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        return str(labels[best]), float(scores[best]), _score_map(labels, scores)
    if labels:
        return str(labels[0]), scores[0] if scores else None, _score_map(labels, scores)
    return None, scores[0] if scores else None, {}


def _score_map(labels: list[Any], scores: list[float]) -> dict[str, float]:
    if not labels or not scores:
        return {}
    count = min(len(labels), len(scores))
    return {str(labels[index]): float(scores[index]) for index in range(count)}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _float_list(value: Any) -> list[float]:
    output: list[float] = []
    for item in _as_list(value):
        try:
            output.append(float(item))
        except (TypeError, ValueError):
            continue
    return output


def _first_record(raw_output: Any) -> Any:
    if isinstance(raw_output, (list, tuple)):
        return raw_output[0] if raw_output else {}
    return raw_output


def _label_variants(raw_label: Any) -> list[str]:
    label = _normalize_label(raw_label)
    if not label:
        return [""]
    pieces = [label]
    for separator in ("/", "|", ","):
        if separator in label:
            pieces.extend(part.strip() for part in label.split(separator))
    pieces.extend(piece.replace(" ", "_") for piece in list(pieces))
    return [piece.upper() for piece in pieces]


def _normalize_label(raw_label: Any) -> str:
    if raw_label is None:
        return ""
    label = str(raw_label).strip()
    if label.startswith("<") and label.endswith(">"):
        label = label.strip("<>")
    return label


def _ser_status(enabled: bool, skipped: bool, reason: str | None) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "skipped": skipped,
        "reason": reason,
        "label": "NEUTRAL",
        "confidence": 0.0,
        "raw_label": None,
        "raw_confidence": None,
        "scores": {},
        "model": config.SER_MODEL_DIR,
        "latency_ms": 0,
        "window_seconds": 0.0,
        "experimental": True,
        "license_caveat": config.SER_LICENSE_CAVEAT,
    }


def _configure_torch_threads() -> None:
    try:
        import torch

        torch.set_num_threads(max(1, config.SER_THREADS))
        torch.set_num_interop_threads(1)
    except Exception as exc:
        logger.debug("event=ser_torch_thread_config_skipped error=%s", exc)


def _log_mapped_result(result: dict[str, Any]) -> None:
    logger.info(
        "event=ser_classified model=%s label=%s confidence=%.3f raw_label=%s latency_ms=%d",
        result.get("model"),
        result.get("label"),
        float(result.get("confidence") or 0.0),
        result.get("raw_label"),
        int(result.get("latency_ms") or 0),
    )


def _clip(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
