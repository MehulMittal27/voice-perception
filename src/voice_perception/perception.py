"""SenseVoice, Emotion2Vec+, and acoustic perception pipeline."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voice_perception import config
from voice_perception.audio import (
    SpeechActivity,
    analyze_speech_activity,
    load_wav_16khz_mono,
    to_float32_mono,
)
from voice_perception.emotion import Emotion2VecClassifier
from voice_perception.signals import analyze_acoustic_context
from voice_perception.transcription import (
    FasterWhisperTranscriber,
    is_german_language,
    normalize_language_selection,
    sensevoice_language_for,
    sensevoice_transcript_fields,
    transcript_not_called,
)

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"<\|([^|<>]+)\|>")
EMOTION_LABELS = set(config.EMOTION_STRESS.keys())
EVENT_ALIASES = {
    "SPEECH": "Speech",
    "BREATH": "Breath",
    "COUGH": "Cough",
    "CRY": "Cry",
    "LAUGHTER": "Laughter",
    "LAUGH": "Laughter",
    "APPLAUSE": "Applause",
    "BGM": "BGM",
    "MUSIC": "Music",
    "NOISE": "Noise",
    "SNEEZE": "Sneeze",
}
CONTROL_TOKENS = {"ZH", "EN", "YUE", "JA", "KO", "NOSPEECH", "WITHITN", "WOITN"}


class VoicePerception:
    """Load SenseVoice once and analyze 16 kHz mono PCM chunks."""

    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = model_dir or config.SENSEVOICE_MODEL_DIR
        self.resolved_model_dir: Path | None = None
        self._lock = threading.Lock()
        self._input_mode: str | None = None
        start = time.perf_counter()
        self.model = self._load_model()
        self.ser = Emotion2VecClassifier(preload=config.SER_PRELOAD)
        self.german_transcriber = FasterWhisperTranscriber(preload=config.GERMAN_ASR_PRELOAD)
        self.load_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "event=model_loaded model_dir=%s ser_enabled=%s load_ms=%d",
            self.model_dir,
            self.ser.enabled,
            self.load_ms,
        )

    def analyze(self, pcm_16khz_mono: np.ndarray, language: str | None = None) -> dict[str, Any]:
        selected_language = normalize_language_selection(language)
        pcm = to_float32_mono(pcm_16khz_mono)
        activity = analyze_speech_activity(pcm)
        acoustic = analyze_acoustic_context(pcm)
        if pcm.size < config.MIN_INFERENCE_SAMPLES:
            return _attach_additive_fields(
                self._empty_result(activity.silence_ratio, selected_language, "min_context"),
                acoustic,
                _ser_not_called("min_context"),
            )
        if not activity.has_speech:
            return _attach_additive_fields(
                self._no_speech_result(activity, selected_language),
                acoustic,
                _ser_not_called("no_speech"),
            )
        start = time.perf_counter()
        sensevoice_language = sensevoice_language_for(selected_language)
        with self._lock:
            raw_output = self._infer(pcm, sensevoice_language)
        inference_ms = int((time.perf_counter() - start) * 1000)
        parsed = parse_sensevoice_output(raw_output)
        parsed["silence_ratio"] = activity.silence_ratio
        parsed["inference_ms"] = inference_ms
        parsed["no_speech"] = False
        self._apply_transcript_lane(parsed, pcm, selected_language, sensevoice_language)
        ser_result = self._analyze_ser(pcm, activity)
        return _attach_additive_fields(parsed, acoustic, ser_result)

    def _apply_transcript_lane(
        self,
        parsed: dict[str, Any],
        pcm: np.ndarray,
        language: str,
        sensevoice_language: str,
    ) -> None:
        sensevoice_transcript = str(parsed.get("transcript", ""))
        parsed.update(
            sensevoice_transcript_fields(
                sensevoice_transcript,
                language,
                int(parsed.get("inference_ms", 0)),
                self.model_dir,
            )
        )
        parsed["sensevoice_language"] = sensevoice_language
        if not is_german_language(language):
            return
        transcriber = getattr(self, "german_transcriber", None)
        if transcriber is None:
            german_result = transcript_not_called(language, "not_configured")
            german_result["transcript"] = ""
        else:
            german_result = transcriber.analyze(pcm, language="de")
        parsed["transcript"] = str(german_result.get("transcript", ""))
        parsed.update(german_result)
        parsed["sensevoice_transcript"] = sensevoice_transcript
        parsed["sensevoice_language"] = sensevoice_language

    def _analyze_ser(self, pcm: np.ndarray, activity: SpeechActivity) -> dict[str, Any]:
        ser = getattr(self, "ser", None)
        if ser is None:
            return _ser_not_called("not_configured")
        return ser.analyze(pcm, activity)

    def _load_model(self) -> Any:
        try:
            from funasr_onnx import SenseVoiceSmall

            model_path = self._resolve_model_dir()
            self._ensure_sentencepiece_model(model_path)
            self.resolved_model_dir = model_path
            logger.info("event=model_config language=%s", config.SENSEVOICE_LANGUAGE)
            return SenseVoiceSmall(
                str(model_path),
                batch_size=1,
                device_id=config.SENSEVOICE_DEVICE_ID,
                quantize=config.SENSEVOICE_QUANTIZE,
                intra_op_num_threads=config.SENSEVOICE_THREADS,
            )
        except TypeError as exc:
            if "exceptions must derive" in str(exc):
                raise RuntimeError(
                    "funasr-onnx could not load the SenseVoice model. Use "
                    "iic/SenseVoiceSmall-onnx or set SENSEVOICE_MODEL_DIR to a "
                    "local ONNX model path; PyTorch repos require a funasr export step."
                ) from exc
            raise

    def _resolve_model_dir(self) -> Path:
        model_path = Path(self.model_dir)
        if model_path.exists():
            return model_path
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install modelscope to download remote SenseVoice models.") from exc
        cache_dir = str(Path(config.SENSEVOICE_CACHE_DIR))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return Path(snapshot_download(self.model_dir, cache_dir=cache_dir))

    def _ensure_sentencepiece_model(self, model_path: Path) -> None:
        bpe_path = model_path / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        if bpe_path.exists():
            return
        try:
            from modelscope.hub.file_download import model_file_download
        except ImportError as exc:
            raise RuntimeError("Install modelscope to fetch the SenseVoice tokenizer model.") from exc
        source = model_file_download(
            "iic/SenseVoiceSmall",
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
            cache_dir=str(Path(config.SENSEVOICE_CACHE_DIR)),
        )
        if not source:
            raise RuntimeError("Could not download SenseVoice SentencePiece tokenizer model.")
        shutil.copyfile(source, bpe_path)

    def _infer(self, pcm: np.ndarray, language: str) -> Any:
        if self._input_mode == "file":
            return self._infer_file(pcm, language)
        try:
            output = self._infer_numpy(pcm, language)
            self._input_mode = "numpy"
            return output
        except Exception as exc:
            logger.info("event=model_numpy_input_failed fallback=file error=%s", exc)
            self._input_mode = "file"
            return self._infer_file(pcm, language)

    def _infer_numpy(self, pcm: np.ndarray, language: str) -> Any:
        return self.model(
            np.ascontiguousarray(pcm, dtype=np.float32),
            language=language,
            textnorm="withitn",
        )

    def _infer_file(self, pcm: np.ndarray, language: str) -> Any:
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = handle.name
            sf.write(temp_path, pcm, config.SAMPLE_RATE, subtype="PCM_16")
            return self.model(
                temp_path,
                language=language,
                textnorm="withitn",
            )
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    @staticmethod
    def _empty_result(silence_ratio: float, language: str, reason: str) -> dict[str, Any]:
        result = {
            "transcript": "",
            "emotion": "NEUTRAL",
            "emotion_confidence": 0.0,
            "raw_emotion": None,
            "events": [],
            "silence_ratio": silence_ratio,
            "inference_ms": 0,
            "no_speech": True,
        }
        result.update(transcript_not_called(language, reason))
        return result

    @staticmethod
    def _no_speech_result(activity: SpeechActivity, language: str) -> dict[str, Any]:
        result = {
            "transcript": "",
            "emotion": "NEUTRAL",
            "emotion_confidence": 0.0,
            "raw_emotion": None,
            "events": [],
            "silence_ratio": activity.silence_ratio,
            "inference_ms": 0,
            "no_speech": True,
        }
        result.update(transcript_not_called(language, "no_speech"))
        return result


def _attach_additive_fields(
    result: dict[str, Any], acoustic: dict[str, Any], ser_result: dict[str, Any]
) -> dict[str, Any]:
    output = dict(result)
    _preserve_sensevoice_emotion(output)
    _apply_ser_emotion(output, ser_result)
    output.update(acoustic)
    output["acoustic_debug"] = {"voice_state": dict(acoustic.get("voice_state", {}))}
    output["ser"] = ser_result
    output["raw_ser_label"] = ser_result.get("raw_label")
    output["capabilities"] = _capabilities(bool(ser_result.get("enabled")))
    _merge_emotion_driver(output)
    return output


def _preserve_sensevoice_emotion(output: dict[str, Any]) -> None:
    output["sensevoice_emotion"] = output.get("emotion", "NEUTRAL")
    output["sensevoice_emotion_confidence"] = float(output.get("emotion_confidence", 0.0))
    output["sensevoice_raw_emotion"] = output.get("raw_emotion")


def _apply_ser_emotion(output: dict[str, Any], ser_result: dict[str, Any]) -> None:
    output["emotion_source"] = "sensevoice"
    if not ser_result.get("enabled"):
        return
    if ser_result.get("skipped") or ser_result.get("error"):
        output["emotion"] = "NEUTRAL"
        output["emotion_confidence"] = 0.0
        output["emotion_source"] = "none"
        return
    output["emotion"] = ser_result.get("label", "NEUTRAL")
    output["emotion_confidence"] = float(ser_result.get("confidence", 0.0))
    output["emotion_source"] = "emotion2vec"


def _merge_emotion_driver(output: dict[str, Any]) -> None:
    drivers = output.get("score_drivers")
    if not isinstance(drivers, dict):
        return
    emotion = str(output.get("emotion", "NEUTRAL")).upper()
    confidence = float(output.get("emotion_confidence", 0.0))
    drivers["emotion_token"] = config.EMOTION_STRESS.get(emotion, 0.0) * confidence


def _ser_not_called(reason: str) -> dict[str, Any]:
    return {
        "enabled": config.SER_ENABLED,
        "skipped": True,
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


def _capabilities(ser_enabled: bool) -> dict[str, Any]:
    return {
        "emotion_labels_supported": ser_enabled,
        "emotion_probabilities_calibrated": False,
        "event_labels_supported": True,
        "transcript_supported": True,
        "german_transcript_supported": config.GERMAN_ASR_ENABLED,
        "german_transcript_model": config.GERMAN_ASR_MODEL if config.GERMAN_ASR_ENABLED else None,
        "voice_state_supported": True,
        "voice_state_debug_only": True,
        "acoustic_emotion_labels_supported": False,
        "score_drivers_supported": True,
        "ser_license_caveat": config.SER_LICENSE_CAVEAT if ser_enabled else None,
    }


def parse_sensevoice_output(raw_output: Any) -> dict[str, Any]:
    record = _first_record(raw_output)
    raw_text, direct_confidence = _extract_text_and_confidence(record)
    tokens = TOKEN_RE.findall(raw_text)
    emotion, token_has_emotion, raw_emotion = _extract_emotion(tokens)
    events = _extract_events(tokens)
    transcript = _strip_tokens(raw_text)
    _log_unknown_emotion(raw_emotion, raw_text)
    if direct_confidence is None:
        # funasr-onnx 0.4.1 returns decoded tokens, not per-label probabilities.
        # Presence of an explicit emotion token is the confidence proxy.
        confidence = 1.0 if token_has_emotion else 0.0
    else:
        confidence = _clip(direct_confidence)
    return {
        "transcript": transcript,
        "emotion": emotion,
        "emotion_confidence": confidence,
        "raw_emotion": raw_emotion,
        "events": events,
        "silence_ratio": 0.0,
        "inference_ms": 0,
    }


def _log_unknown_emotion(raw_emotion: str | None, raw_text: str) -> None:
    if raw_emotion is None or raw_emotion in EMOTION_LABELS:
        return
    logger.info(
        "event=model_unknown_emotion raw_emotion=%s output_chars=%d",
        raw_emotion,
        len(raw_text),
    )


def _first_record(raw_output: Any) -> Any:
    if isinstance(raw_output, (list, tuple)):
        return raw_output[0] if raw_output else ""
    return raw_output


def _extract_text_and_confidence(record: Any) -> tuple[str, float | None]:
    if isinstance(record, dict):
        text = str(record.get("text") or record.get("transcript") or record.get("result") or "")
        return text, _find_confidence(record)
    return str(record), None


def _find_confidence(record: dict[str, Any]) -> float | None:
    for key in ("emotion_confidence", "confidence", "score"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_emotion(tokens: list[str]) -> tuple[str, bool, str | None]:
    raw_emotion: str | None = None
    for token in tokens:
        label = token.strip().upper()
        if label in EMOTION_LABELS:
            return label, True, label
        if label.startswith("EMO_"):
            raw_emotion = label
    return "NEUTRAL", False, raw_emotion


def _extract_events(tokens: list[str]) -> list[str]:
    events: list[str] = []
    for token in tokens:
        event = _normalize_event(token)
        if event and event not in events:
            events.append(event)
    return events


def _normalize_event(token: str) -> str | None:
    label = token.strip().upper()
    if label in EMOTION_LABELS or label in CONTROL_TOKENS:
        return None
    return EVENT_ALIASES.get(label)


def _strip_tokens(raw_text: str) -> str:
    without_tokens = TOKEN_RE.sub(" ", raw_text)
    compact = re.sub(r"\s+", " ", without_tokens).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", compact)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run voice perception on a WAV file.")
    parser.add_argument("wav_file", type=Path)
    parser.add_argument("--language", default="auto", help="Transcript language: auto, en, or de")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pcm = load_wav_16khz_mono(args.wav_file)
    with contextlib.redirect_stdout(sys.stderr):
        perception = VoicePerception()
        result = perception.analyze(pcm, language=args.language)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
