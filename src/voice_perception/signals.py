"""Fast numpy acoustic voice-state signals.

This lane is deterministic and language-agnostic. It does not claim exact
emotion. It estimates speech activity, arousal, stress, hesitation, and
speaking confidence from energy, pauses, spectral shape, and pitch movement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from voice_perception import config
from voice_perception.audio import analyze_speech_activity, to_float32_mono

EPS = 1e-8


@dataclass(frozen=True)
class AcousticContext:
    features: dict[str, np.ndarray]
    speech_mask: np.ndarray
    f0_hz: np.ndarray
    activity_silence_ratio: float
    duration_s: float


def analyze_acoustic_context(pcm_16khz_mono: np.ndarray) -> dict[str, Any]:
    """Return additive voice-state fields for a PCM rolling context."""

    started = time.perf_counter()
    pcm = to_float32_mono(pcm_16khz_mono)
    if pcm.size == 0:
        result = default_acoustic_analysis()
        result["signals"]["latency_ms"] = _elapsed_ms(started)
        return result
    context = _build_context(pcm)
    aggregates = _aggregate_context(context)
    events = _detect_signal_events(context, aggregates)
    scores = _score_aggregates(aggregates, len(events))
    no_speech = _is_no_speech(context, aggregates)
    result = _format_result(aggregates, scores, events, no_speech)
    result["signals"]["latency_ms"] = _elapsed_ms(started)
    return result


def default_acoustic_analysis() -> dict[str, Any]:
    """Return a stable no-speech voice-state payload."""

    return {
        "voice_state": {
            "label": "no_speech",
            "confidence": 1.0,
            "secondary": [],
            "updated_at": None,
        },
        "signals": {
            "speech_activity": 0.0,
            "speech_confidence": 0.0,
            "snr_db": 0.0,
            "arousal": 0.0,
            "stress": 0.0,
            "hesitation": 0.0,
            "speaking_confidence": 0.0,
            "signal_reliability": 1.0,
            "calibrated": False,
            "latency_ms": 0,
            "no_speech": True,
        },
        "signal_events": [],
        "score_drivers": {
            "pause_silence": 0.0,
            "energy_arousal": 0.0,
            "pitch_instability": 0.0,
            "breath_event": 0.0,
            "emotion_token": 0.0,
        },
        "debug_features": {},
    }


def _build_context(pcm: np.ndarray) -> AcousticContext:
    activity = analyze_speech_activity(pcm)
    features = _frame_features(pcm)
    speech_mask = _speech_mask(features, activity.has_speech)
    f0_hz = _pitch_values(pcm, speech_mask)
    return AcousticContext(
        features=features,
        speech_mask=speech_mask,
        f0_hz=f0_hz,
        activity_silence_ratio=activity.silence_ratio,
        duration_s=pcm.size / float(config.SAMPLE_RATE),
    )


def _frame_features(pcm: np.ndarray) -> dict[str, np.ndarray]:
    frame_len = _ms_to_samples(config.ACOUSTIC_FRAME_MS)
    hop = _ms_to_samples(config.ACOUSTIC_HOP_MS)
    frames = _frames(pcm, frame_len, hop)
    window = np.hanning(frame_len).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / config.SAMPLE_RATE)
    rows = [_single_frame_features(frame, window, freqs) for frame in frames]
    return {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in rows[0]}


def _single_frame_features(
    frame: np.ndarray, window: np.ndarray, freqs: np.ndarray
) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32)) + EPS))
    mag = np.abs(np.fft.rfft(frame * window)) + EPS
    mag_sum = float(np.sum(mag))
    high_mag = float(np.sum(mag[freqs >= 2500.0]))
    flatness = float(np.exp(np.mean(np.log(mag))) / (np.mean(mag) + EPS))
    return {
        "rms": rms,
        "rms_db": _db(rms),
        "peak": float(np.max(np.abs(frame))) if frame.size else 0.0,
        "zcr": _zero_crossing_rate(frame),
        "centroid_hz": float(np.sum(freqs * mag) / (mag_sum + EPS)),
        "high_band_ratio": high_mag / (mag_sum + EPS),
        "flatness": flatness,
    }


def _frames(pcm: np.ndarray, frame_len: int, hop: int) -> list[np.ndarray]:
    if pcm.size < frame_len:
        padded = np.zeros(frame_len, dtype=np.float32)
        padded[: pcm.size] = pcm
        return [padded]
    starts = list(range(0, pcm.size - frame_len + 1, hop))
    if starts[-1] != pcm.size - frame_len:
        starts.append(pcm.size - frame_len)
    return [pcm[start : start + frame_len] for start in starts]


def _speech_mask(features: dict[str, np.ndarray], activity_has_speech: bool) -> np.ndarray:
    rms_db = features["rms_db"]
    noise_floor_db = _noise_floor_db(rms_db)
    energy_speech = rms_db >= max(noise_floor_db + 8.0, -55.0)
    high_band = features["high_band_ratio"]
    flatness = features["flatness"]
    zcr = features["zcr"]
    tone_like = (flatness < 0.015) & (high_band < 0.015)
    voice_like = (((zcr >= 0.01) & (zcr <= 0.30)) | (high_band > 0.03)) & (flatness < 0.78)
    noise_like = (flatness > 0.78) & (high_band > 0.55)
    mask = energy_speech & voice_like & ~noise_like & ~tone_like
    return _fallback_speech_mask(mask, energy_speech, noise_like, tone_like, activity_has_speech)


def _fallback_speech_mask(
    mask: np.ndarray,
    energy_speech: np.ndarray,
    noise_like: np.ndarray,
    tone_like: np.ndarray,
    activity_has_speech: bool,
) -> np.ndarray:
    if _speech_seconds(mask) >= config.ACOUSTIC_MIN_SPEECH_SECONDS:
        return mask
    if not activity_has_speech or bool(np.all(tone_like)):
        return mask
    fallback = energy_speech & ~noise_like & ~tone_like
    return fallback if _speech_seconds(fallback) > _speech_seconds(mask) else mask


def _pitch_values(pcm: np.ndarray, speech_mask: np.ndarray) -> np.ndarray:
    frame_len = _ms_to_samples(config.ACOUSTIC_PITCH_FRAME_MS)
    hop = _ms_to_samples(config.ACOUSTIC_HOP_MS)
    if pcm.size < frame_len or speech_mask.size == 0:
        return np.empty(0, dtype=np.float32)
    values: list[float] = []
    window = np.hanning(frame_len).astype(np.float32)
    for index, start in enumerate(range(0, pcm.size - frame_len + 1, hop)):
        if index >= speech_mask.size or not bool(speech_mask[index]):
            continue
        f0 = _autocorrelation_pitch(pcm[start : start + frame_len], window)
        if f0 is not None:
            values.append(f0)
    return np.asarray(values, dtype=np.float32)


def _autocorrelation_pitch(frame: np.ndarray, window: np.ndarray) -> float | None:
    centered = (frame - float(np.mean(frame))) * window
    energy = float(np.dot(centered, centered))
    if energy <= EPS:
        return None
    corr = np.correlate(centered, centered, mode="full")[frame.size - 1 :]
    min_lag = max(1, int(config.SAMPLE_RATE / 400.0))
    max_lag = min(corr.size - 1, int(config.SAMPLE_RATE / 60.0))
    if max_lag <= min_lag:
        return None
    best_lag = int(np.argmax(corr[min_lag : max_lag + 1]) + min_lag)
    confidence = float(corr[best_lag] / (corr[0] + EPS))
    return config.SAMPLE_RATE / best_lag if confidence >= 0.35 else None


def _aggregate_context(context: AcousticContext) -> dict[str, float]:
    features = context.features
    mask = context.speech_mask
    speech_db = features["rms_db"][mask]
    speech_seconds = _speech_seconds(mask)
    pauses = _pause_durations(mask)
    f0_stats = _f0_stats(context.f0_hz)
    return _base_aggregates(context, speech_db, speech_seconds, pauses) | f0_stats


def _base_aggregates(
    context: AcousticContext,
    speech_db: np.ndarray,
    speech_seconds: float,
    pauses: list[float],
) -> dict[str, float]:
    features = context.features
    mask = context.speech_mask
    median_speech_db = _median_or(speech_db, -90.0)
    noise_floor_db = _noise_floor_db(features["rms_db"])
    return {
        "duration_s": context.duration_s,
        "speech_seconds": speech_seconds,
        "speech_ratio": _safe_ratio(speech_seconds, context.duration_s),
        "silence_ratio": context.activity_silence_ratio,
        "pause_ratio": _safe_ratio(sum(pauses), context.duration_s),
        "long_pause_count": float(sum(1 for value in pauses if value >= 0.30)),
        "micro_pause_count": float(sum(1 for value in pauses if 0.12 <= value < 0.30)),
        "turn_restart_rate": _turn_restart_rate(mask, context.duration_s),
        "median_speech_db": median_speech_db,
        "noise_floor_db": noise_floor_db,
        "snr_db": max(0.0, median_speech_db - noise_floor_db),
        "energy_iqr_db": _iqr_or(speech_db, 0.0),
        "energy_range_db": _range_or(speech_db, 0.0),
        "speech_rate_proxy": _speech_rate_proxy(features["rms_db"], mask),
        "high_band_median": _median_or(features["high_band_ratio"][mask], 0.0),
        "flatness_median": _median_or(features["flatness"][mask], 0.0),
        "peak": float(np.max(features["peak"])) if features["peak"].size else 0.0,
    }


def _f0_stats(f0_hz: np.ndarray) -> dict[str, float]:
    if f0_hz.size < 5:
        return {"f0_median_hz": 0.0, "f0_range_st": 0.0, "f0_std_st": 0.0, "f0_voiced_count": float(f0_hz.size)}
    median = float(np.median(f0_hz))
    semitones = 12.0 * np.log2(np.maximum(f0_hz, EPS) / max(median, EPS))
    return {
        "f0_median_hz": median,
        "f0_range_st": _range_or(semitones, 0.0),
        "f0_std_st": float(np.std(semitones)),
        "f0_voiced_count": float(f0_hz.size),
    }


def _detect_signal_events(context: AcousticContext, aggregates: dict[str, float]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for start, end in _segments(~context.speech_mask):
        event = _breath_like_event(context, aggregates, start, end)
        if event:
            events.append(event)
    return events[-3:]


def _breath_like_event(
    context: AcousticContext, aggregates: dict[str, float], start: int, end: int
) -> dict[str, Any] | None:
    duration_s = (end - start) * _hop_seconds()
    if duration_s < 0.08 or duration_s > 0.70 or not _adjacent_to_speech(context.speech_mask, start, end):
        return None
    high = _median_or(context.features["high_band_ratio"][start:end], 0.0)
    flat = _median_or(context.features["flatness"][start:end], 0.0)
    zcr = _median_or(context.features["zcr"][start:end], 0.0)
    db_value = _median_or(context.features["rms_db"][start:end], -90.0)
    confidence = np.mean([_clip((high - 0.30) / 0.35), _clip((flat - 0.25) / 0.45), _clip((zcr - 0.05) / 0.15)])
    if confidence < 0.50 or db_value < aggregates["noise_floor_db"] + 4.0:
        return None
    age_ms = int(max(0.0, context.duration_s - end * _hop_seconds()) * 1000)
    return {"type": "breath_like", "confidence": float(confidence), "age_ms": age_ms}


def _score_aggregates(aggregates: dict[str, float], breath_event_count: int) -> dict[str, float]:
    arousal_parts = _arousal_parts(aggregates)
    hesitation_parts = _hesitation_parts(aggregates, breath_event_count)
    arousal = _combine_arousal(arousal_parts, aggregates)
    hesitation = _clip(sum(hesitation_parts.values()))
    stress = _stress_score(arousal, arousal_parts, hesitation_parts, hesitation)
    reliability = _signal_reliability(aggregates, arousal_parts, hesitation_parts)
    speaking_confidence = _speaking_confidence(aggregates, hesitation, arousal_parts)
    low_energy = _low_energy_score(aggregates, arousal)
    return {
        **arousal_parts,
        **hesitation_parts,
        "arousal": arousal,
        "hesitation": hesitation,
        "stress": stress,
        "signal_reliability": reliability,
        "speaking_confidence": speaking_confidence,
        "low_energy": low_energy,
    }


def _arousal_parts(aggregates: dict[str, float]) -> dict[str, float]:
    return {
        "energy_high": _high_delta(aggregates["median_speech_db"], config.ACOUSTIC_BASELINE_ENERGY_DB, 2.0, 12.0),
        "energy_movement": _high_delta(aggregates["energy_range_db"], config.ACOUSTIC_BASELINE_ENERGY_IQR_DB + 4.0, 0.0, 14.0),
        "pitch_movement": _high_delta(aggregates["f0_range_st"], config.ACOUSTIC_BASELINE_F0_RANGE_ST, 2.0, 10.0),
        "rate_high": _high_delta(aggregates["speech_rate_proxy"], config.ACOUSTIC_BASELINE_RATE, 0.4, 2.5),
    }


def _hesitation_parts(aggregates: dict[str, float], breath_event_count: int) -> dict[str, float]:
    pause_excess = _high_delta(aggregates["pause_ratio"], config.ACOUSTIC_BASELINE_PAUSE_RATIO, 0.08, 0.35)
    silence_excess = _high_delta(aggregates["silence_ratio"], config.ACOUSTIC_BASELINE_PAUSE_RATIO, 0.08, 0.35)
    return {
        "pause_excess": 0.34 * pause_excess,
        "silence_excess": 0.16 * silence_excess,
        "long_pause_norm": 0.20 * _clip(aggregates["long_pause_count"] / 3.0),
        "micro_pause_norm": 0.12 * _clip(aggregates["micro_pause_count"] / 5.0),
        "restart_norm": 0.12 * _clip(aggregates["turn_restart_rate"] / 1.2),
        "rate_drop": 0.04 * _low_delta(aggregates["speech_rate_proxy"], config.ACOUSTIC_BASELINE_RATE, 0.4, 2.0),
        "breath_norm": 0.10 * _clip(breath_event_count / 2.0),
    }


def _combine_arousal(parts: dict[str, float], aggregates: dict[str, float]) -> float:
    if aggregates["f0_voiced_count"] < 5.0:
        return _clip(0.35 * parts["energy_high"] + 0.40 * parts["energy_movement"] + 0.25 * parts["rate_high"])
    return _clip(0.35 * parts["energy_high"] + 0.25 * parts["energy_movement"] + 0.25 * parts["pitch_movement"] + 0.15 * parts["rate_high"])


def _stress_score(
    arousal: float,
    arousal_parts: dict[str, float],
    hesitation_parts: dict[str, float],
    hesitation: float,
) -> float:
    arousal_high = _high_delta(arousal, 0.45, 0.0, 0.45)
    instability = _clip(0.5 * arousal_parts["energy_movement"] + 0.5 * arousal_parts["pitch_movement"])
    noise_tension = _clip(6.0 * hesitation_parts["breath_norm"])
    pause_tension = min(hesitation, 0.75)
    return _clip(0.40 * arousal_high + 0.25 * instability + 0.20 * noise_tension + 0.15 * pause_tension)


def _signal_reliability(
    aggregates: dict[str, float], arousal_parts: dict[str, float], hesitation_parts: dict[str, float]
) -> float:
    snr_quality = _clip((aggregates["snr_db"] - 6.0) / 18.0)
    speech_coverage = _clip(aggregates["speech_seconds"] / 1.0)
    pitch_quality = _clip(aggregates["f0_voiced_count"] / 20.0)
    active_families = sum(1 for value in [*arousal_parts.values(), *hesitation_parts.values()] if value > 0.03)
    feature_agreement = _clip(active_families / 4.0)
    return _clip(0.30 * snr_quality + 0.20 * speech_coverage + 0.20 * 0.55 + 0.15 * pitch_quality + 0.15 * feature_agreement)


def _speaking_confidence(
    aggregates: dict[str, float], hesitation: float, arousal_parts: dict[str, float]
) -> float:
    steady_energy = 1.0 - _clip(aggregates["energy_range_db"] / 24.0)
    steady_pitch = 1.0 - _clip(aggregates["f0_range_st"] / 18.0) if aggregates["f0_voiced_count"] >= 5.0 else 0.5
    few_pauses = 1.0 - hesitation
    rate_ok = 1.0 - _clip(abs(aggregates["speech_rate_proxy"] - config.ACOUSTIC_BASELINE_RATE) / 2.5)
    coverage_ok = _clip(aggregates["speech_ratio"] / 0.65)
    return _clip(0.30 * few_pauses + 0.20 * steady_energy + 0.15 * steady_pitch + 0.20 * rate_ok + 0.15 * coverage_ok)


def _low_energy_score(aggregates: dict[str, float], arousal: float) -> float:
    return _clip(
        0.45 * _low_delta(aggregates["median_speech_db"], config.ACOUSTIC_BASELINE_ENERGY_DB, 3.0, 14.0)
        + 0.25 * _low_delta(aggregates["speech_rate_proxy"], config.ACOUSTIC_BASELINE_RATE, 0.4, 2.0)
        + 0.30 * (1.0 - arousal)
    )


def _is_no_speech(context: AcousticContext, aggregates: dict[str, float]) -> bool:
    return aggregates["speech_seconds"] < config.ACOUSTIC_MIN_SPEECH_SECONDS or context.activity_silence_ratio >= 0.98


def _format_result(
    aggregates: dict[str, float], scores: dict[str, float], events: list[dict[str, Any]], no_speech: bool
) -> dict[str, Any]:
    if no_speech:
        return _no_speech_result(aggregates)
    label = _voice_label(scores)
    confidence = _voice_label_confidence(label, scores)
    return {
        "voice_state": {"label": label, "confidence": confidence, "secondary": _secondary_labels(label, scores), "updated_at": None},
        "signals": _signals_payload(aggregates, scores, no_speech=False),
        "signal_events": events,
        "score_drivers": _score_drivers(scores),
        "debug_features": _debug_features(aggregates),
    }


def _no_speech_result(aggregates: dict[str, float]) -> dict[str, Any]:
    result = default_acoustic_analysis()
    result["signals"]["silence_ratio"] = float(aggregates.get("silence_ratio", 1.0))
    result["debug_features"] = _debug_features(aggregates)
    return result


def _signals_payload(aggregates: dict[str, float], scores: dict[str, float], no_speech: bool) -> dict[str, Any]:
    return {
        "speech_activity": 0.0 if no_speech else _clip(aggregates["speech_ratio"]),
        "speech_confidence": 0.0 if no_speech else _clip(0.65 * aggregates["speech_ratio"] + 0.35 * ((aggregates["snr_db"] - 6.0) / 18.0)),
        "snr_db": round(float(aggregates["snr_db"]), 2),
        "arousal": 0.0 if no_speech else scores["arousal"],
        "stress": 0.0 if no_speech else scores["stress"],
        "hesitation": 0.0 if no_speech else scores["hesitation"],
        "speaking_confidence": 0.0 if no_speech else scores["speaking_confidence"],
        "signal_reliability": 1.0 if no_speech else scores["signal_reliability"],
        "calibrated": False,
        "latency_ms": 0,
        "no_speech": no_speech,
        "silence_ratio": float(aggregates.get("silence_ratio", 0.0)),
    }


def _voice_label(scores: dict[str, float]) -> str:
    if scores["signal_reliability"] < 0.45:
        return "uncertain"
    if scores["hesitation"] >= 0.33:
        return "hesitant"
    if scores["stress"] >= 0.65 and scores["arousal"] >= 0.65:
        return "agitated"
    if scores["stress"] >= 0.55:
        return "stressed"
    if scores["arousal"] <= 0.25 and scores["low_energy"] >= 0.60:
        return "subdued"
    if scores["speaking_confidence"] >= 0.65 and scores["hesitation"] <= 0.30 and scores["stress"] <= 0.35:
        return "confident"
    return "calm"


def _voice_label_confidence(label: str, scores: dict[str, float]) -> float:
    if label == "uncertain":
        return _clip(1.0 - scores["signal_reliability"])
    if label == "confident":
        basis = scores["speaking_confidence"]
    elif label == "hesitant":
        basis = scores["hesitation"]
    elif label in {"stressed", "agitated"}:
        basis = max(scores["stress"], scores["arousal"])
    elif label == "subdued":
        basis = scores["low_energy"]
    else:
        basis = 1.0 - max(scores["stress"], scores["hesitation"])
    return _clip(0.35 + 0.65 * basis * max(scores["signal_reliability"], 0.50))


def _secondary_labels(label: str, scores: dict[str, float]) -> list[str]:
    labels: list[str] = []
    if scores["stress"] >= 0.45 and label != "stressed":
        labels.append("stressed")
    if scores["hesitation"] >= 0.28 and label != "hesitant":
        labels.append("hesitant")
    if scores["low_energy"] >= 0.55 and label != "subdued":
        labels.append("subdued")
    return labels[:2]


def _score_drivers(scores: dict[str, float]) -> dict[str, float]:
    return {
        "pause_silence": _clip(
            scores["pause_excess"]
            + scores["silence_excess"]
            + scores["long_pause_norm"]
            + scores["micro_pause_norm"]
        ),
        "energy_arousal": _clip(scores["arousal"]),
        "pitch_instability": _clip(scores["pitch_movement"]),
        "breath_event": _clip(scores["breath_norm"] * 10.0),
        "emotion_token": 0.0,
    }


def _debug_features(aggregates: dict[str, float]) -> dict[str, float]:
    if not config.ACOUSTIC_SIGNAL_DEBUG:
        return {}
    keys = [
        "duration_s",
        "speech_seconds",
        "speech_ratio",
        "silence_ratio",
        "pause_ratio",
        "long_pause_count",
        "micro_pause_count",
        "median_speech_db",
        "noise_floor_db",
        "snr_db",
        "energy_range_db",
        "speech_rate_proxy",
        "f0_median_hz",
        "f0_range_st",
        "f0_std_st",
        "high_band_median",
    ]
    return {key: round(float(aggregates.get(key, 0.0)), 4) for key in keys}


def _speech_rate_proxy(rms_db: np.ndarray, mask: np.ndarray) -> float:
    speech_seconds = _speech_seconds(mask)
    if speech_seconds <= EPS or rms_db.size < 3:
        return 0.0
    smooth = np.convolve(rms_db, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    threshold = _median_or(smooth[mask], -90.0) + 1.5
    peaks = _count_spaced_peaks(smooth, mask, threshold, min_spacing=12)
    return peaks / speech_seconds


def _count_spaced_peaks(values: np.ndarray, mask: np.ndarray, threshold: float, min_spacing: int) -> int:
    count = 0
    last_peak = -min_spacing
    for index in range(1, values.size - 1):
        if not mask[index] or values[index] < threshold or index - last_peak < min_spacing:
            continue
        if values[index] >= values[index - 1] and values[index] > values[index + 1]:
            count += 1
            last_peak = index
    return count


def _pause_durations(mask: np.ndarray) -> list[float]:
    speech_segments = _segments(mask)
    if len(speech_segments) < 2:
        return []
    pauses: list[float] = []
    for (_, previous_end), (next_start, _) in zip(speech_segments, speech_segments[1:]):
        gap = (next_start - previous_end) * _hop_seconds()
        if gap > 0.0:
            pauses.append(gap)
    return pauses


def _turn_restart_rate(mask: np.ndarray, duration_s: float) -> float:
    if duration_s <= EPS:
        return 0.0
    short_segments = sum(1 for start, end in _segments(mask) if (end - start) * _hop_seconds() < 0.70)
    return short_segments / min(max(duration_s, EPS), 6.0)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, mask.size))
    return segments


def _adjacent_to_speech(mask: np.ndarray, start: int, end: int) -> bool:
    before = bool(np.any(mask[max(0, start - 150) : start]))
    after = bool(np.any(mask[end : min(mask.size, end + 150)]))
    return before or after


def _noise_floor_db(rms_db: np.ndarray) -> float:
    if rms_db.size == 0:
        return -90.0
    return float(max(np.percentile(rms_db, 10), -90.0))


def _speech_seconds(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) * _hop_seconds()


def _hop_seconds() -> float:
    return config.ACOUSTIC_HOP_MS / 1000.0


def _ms_to_samples(milliseconds: int) -> int:
    return max(1, int(round(config.SAMPLE_RATE * milliseconds / 1000.0)))


def _median_or(values: np.ndarray, default: float) -> float:
    return float(np.median(values)) if values.size else default


def _iqr_or(values: np.ndarray, default: float) -> float:
    if values.size < 2:
        return default
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _range_or(values: np.ndarray, default: float) -> float:
    if values.size < 2:
        return default
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return _clip(numerator / denominator) if denominator > EPS else 0.0


def _high_delta(value: float, baseline: float, deadband: float, span: float) -> float:
    return _clip((value - baseline - deadband) / span)


def _low_delta(value: float, baseline: float, deadband: float, span: float) -> float:
    return _clip((baseline - value - deadband) / span)


def _zero_crossing_rate(frame: np.ndarray) -> float:
    if frame.size < 2:
        return 0.0
    return float(np.count_nonzero(np.diff(np.signbit(frame))) / (frame.size - 1))


def _db(value: float) -> float:
    return 20.0 * float(np.log10(max(value, EPS)))


def _clip(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
