# AGENTS.md — Voice Perception Service

## What this is
A standalone microservice that listens to live audio and emits a real-time
paralinguistic state: emotion, audio events, and a composite hesitation score.

Built for a hackathon integration where a separate voice-agent service
(ElevenLabs Conversational AI + Claude) will query this service's state
inside its own LLM callback to make dialogue adaptive.

This service does NOT know about ElevenLabs or Claude. It only exposes:
- WebSocket audio ingestion
- HTTP state retrieval

## Design principles
1. **Single responsibility** — perception only. No dialogue, no TTS, no LLM.
2. **Language-agnostic** — must work on any spoken language input, since the
   downstream use case is multilingual (Ukrainian, Arabic, German, English).
   SenseVoice ASR only covers 5 languages, but emotion/event detection is
   paralinguistic and works cross-lingually. We rely on the paralinguistic
   signals, not the transcript.
3. **Testable in isolation** — must be runnable and demonstrable without any
   other service. Ship a browser test page that shows live state.
4. **Fast enough on a laptop CPU** — target <400ms inference per 1s audio
   chunk on a modern laptop with quantized ONNX. If it doesn't hit this on
   the target machine, we fall back to a cloud GPU endpoint.

## Tech stack (locked)
- Python 3.11
- FastAPI + uvicorn (HTTP + WebSocket)
- **SenseVoice-Small** via `funasr-onnx` (quantized ONNX runtime)
  - Upstream: https://github.com/FunAudioLLM/SenseVoice
  - Model card: https://huggingface.co/FunAudioLLM/SenseVoiceSmall
  - License: check the repo (Apache 2.0 at time of writing)
  - Why: single model returns ASR + emotion + audio events in one forward pass;
    quantized ONNX runs on CPU in ~100-300ms per 1s chunk
- PyAV (av) for decoding webm/opus audio from browser
- numpy, soundfile for audio processing
- No frontend framework - a single index.html with vanilla JS for the test UI

## Repo structure
```
voice-perception/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── voice_perception/
│       ├── __init__.py
│       ├── main.py           # FastAPI app + routes
│       ├── perception.py     # SenseVoice model wrapper
│       ├── session.py        # per-session state manager
│       ├── audio.py          # webm/opus decoding to PCM
│       ├── fusion.py         # composite hesitation score logic
│       └── config.py         # env-var configuration
├── static/
│   └── index.html            # browser test UI
├── scripts/
│   └── test_wav.py           # CLI test against a WAV file
└── tests/
    └── test_fusion.py        # unit tests for scoring logic
```

## API contract (this is the integration surface — do not change without updating consumers)

### POST /session/start
Request: `{}` (empty)
Response: `{ "session_id": "<uuid>" }`

### WebSocket /audio/{session_id}
Client sends binary frames of webm/opus audio (MediaRecorder default output).
Recommended chunk interval: 1000ms.
Server acknowledges each chunk with a small JSON message:
`{ "chunk_processed": true, "latency_ms": 234 }`

### GET /state/{session_id}
Response:
```json
{
  "session_id": "<uuid>",
  "updated_at": "2026-07-22T22:14:03.123Z",
  "transcript_partial": "guten tag ich habe",
  "emotion": "FEARFUL",
  "emotion_confidence": 0.72,
  "events": ["Breath"],
  "hesitation_score": 0.68,
  "chunks_processed": 7
}
```

If no state yet: return defaults (emotion NEUTRAL, hesitation 0.0, empty events).

### POST /session/{session_id}/end
Cleans up the session. Response: `{ "ok": true }`

### GET /health
Response: `{ "status": "ok", "model_loaded": true }`

## Hesitation score fusion (this is the "secret sauce" — get it right)

Inputs per chunk:
- SenseVoice emotion label + confidence
- SenseVoice audio events (Breath, Cough, Cry, etc.)
- Silence ratio in the chunk (from VAD or simple energy threshold)
- Time since last non-silent chunk

Scoring (all clipped to [0, 1]):

```
emotion_stress = {
  "FEARFUL": 0.9,
  "SAD": 0.6,
  "ANGRY": 0.5,
  "DISGUSTED": 0.4,
  "SURPRISED": 0.3,
  "NEUTRAL": 0.1,
  "HAPPY": 0.0
}[label] * confidence

event_stress = 0.3 if "Breath" in events else 0.0
              + 0.4 if "Cough" in events else 0.0
              + 0.5 if "Cry" in events else 0.0
              (clip to 0.8)

silence_stress = min(silence_ratio, 1.0) * 0.5

hesitation_score = clip(
  0.5 * emotion_stress
  + 0.3 * event_stress
  + 0.2 * silence_stress,
  0.0, 1.0
)
```

Smooth across chunks with exponential moving average, alpha = 0.4, so a single
noisy chunk doesn't spike the score.

Keep the weights in config.py so they can be tuned during the demo.

## Known unknowns to verify at build time
1. Verified against `funasr-onnx==0.4.1`: `SenseVoiceSmall.__call__` accepts
   numpy arrays and file paths. The wrapper in `src/voice_perception/perception.py`
   tries numpy first, then falls back to a NamedTemporaryFile WAV if needed.
2. Remote model download uses ModelScope. Use `iic/SenseVoiceSmall-onnx` for
   the quantized ONNX runtime model; `iic/SenseVoiceSmall` is a PyTorch repo and
   triggers an ONNX export path that requires `funasr`. If `modelscope` is
   unavailable, install it or set `SENSEVOICE_MODEL_DIR` to a local model
   directory containing `model_quant.onnx`, `config.yaml`, `am.mvn`, and the
   SentencePiece model. The ONNX repo currently ships `tokens.json` but not the
   `.bpe.model` file, so the wrapper fetches that one tokenizer file from the
   PyTorch repo before constructing `SenseVoiceSmall`.
3. Verified output shape for `SenseVoiceSmall.__call__` is `list[str]` with
   inline tokens such as `<|FEARFUL|>`, `<|Breath|>`, language, and textnorm
   markers. Parse them with regex and strip them from the transcript.
4. `funasr-onnx==0.4.1` imports `jieba` and `torch` at runtime without
   declaring them, so keep those pins in `requirements.txt` unless the wrapper
   or upstream package changes.
5. Browser MediaRecorder produces `audio/webm;codecs=opus` on Chrome/Firefox
   but `audio/mp4` on Safari. PyAV handles both, but log the incoming
   container format for debugging.
6. Live WebSocket inference intentionally buffers a rolling PCM window
   (`LIVE_ROLLING_CONTEXT_SECONDS`, default 5s) and waits for about 2s of
   context before the first inference. Do not revert to per-slice SenseVoice
   calls; short chunks can cause auto-language flips and garbage transcripts.
   For English-only demos, set `SENSEVOICE_LANGUAGE=en`.

## Testing rules
- Every component must be runnable in isolation. `python -m voice_perception.perception` should be able to run inference on a bundled test WAV.
- The browser UI at `/` must work end-to-end without any code changes: click, speak, see state updating.
- Provide two test WAVs in `tests/fixtures/` — one calm, one anxious — and a script that runs both through the pipeline and asserts different hesitation scores.

## Non-goals (do not build these)
- User accounts, auth, or persistence
- Multi-language ASR beyond what SenseVoice ships with
- TTS or dialogue generation
- Cloud deployment scripts

## Coding conventions
- Type hints everywhere
- Async where FastAPI expects it, sync elsewhere
- No global mutable state except the SessionManager singleton
- Log at INFO for lifecycle events, DEBUG for per-chunk noise
- Keep functions under 40 lines; if longer, factor

## Attribution

This service uses SenseVoice-Small by Alibaba's FunAudioLLM team.

Repo: https://github.com/FunAudioLLM/SenseVoice

If we ship this beyond the hackathon, verify license compatibility and add proper attribution in the README and any user-facing demo.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
