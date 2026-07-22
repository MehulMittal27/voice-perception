# Voice Perception Service

Real-time FastAPI microservice for paralinguistic voice state.
It ingests microphone audio over WebSocket and exposes live emotion, audio events, transcript, and hesitation score.
Use the bundled browser page to test calm vs anxious speech without any other service.

## Requirements

- Python 3.11
- uv (`brew install uv` if missing)
- About 500 MB free disk for the SenseVoice model cache
- Browser microphone access

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv run uvicorn voice_perception.main:app --reload
```

Open http://localhost:8000 and click Start for live mode. First startup can take 1 to 3 minutes while SenseVoice downloads and loads.

The page also includes a separate record then submit area for one-shot testing. Click Record, speak for a few seconds, click Stop, then click Submit to upload the completed MediaRecorder audio blob to `POST /classify` and render a single classification result.

Both live and one-shot paths apply a configurable no-speech guard before SenseVoice inference. Audio below `NO_SPEECH_RMS_THRESHOLD`, `NO_SPEECH_MIN_NON_SILENT_RATIO`, or `NO_SPEECH_MIN_NON_SILENT_SECONDS` returns an idle no-speech state instead of a hallucinated transcript or `Speech` event.

Live microphone inference uses a rolling PCM context window so SenseVoice does not transcribe each 1 second MediaRecorder slice in isolation.
For an English-only demo, run with `SENSEVOICE_LANGUAGE=en`; the default remains `auto` for multilingual use.

## API summary

See `AGENTS.md` for the full API contract and architecture.

- `POST /session/start` returns a `session_id`
- `WebSocket /audio/{session_id}` accepts MediaRecorder binary audio chunks
- `GET /state/{session_id}` returns live transcript, emotion, events, hesitation score, and chunk count
- `POST /classify` accepts a multipart `file` audio upload and returns one-shot transcript, emotion, events, hesitation score, latency, and audio sample count
- `POST /session/{session_id}/end` cleans up a session
- `GET /health` reports service health and model load state

## Curl integration example

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/session/start | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
curl -s http://localhost:8000/state/$SESSION_ID
curl -s -X POST http://localhost:8000/session/$SESSION_ID/end
```

## One-shot browser test cases

The model responds to paralinguistic audio cues and SenseVoice emotion tokens, not guaranteed sentiment of the words alone. SenseVoice-Small emotion tokens may be limited or unstable on short microphone demos. If this machine mostly emits `NEUTRAL` or unknown emotion tokens, treat no-speech filtering, transcript, audio events, and hesitation from breath, cough, cry, or silence as the reliable signals.

For the clearest comparison, repeat a neutral phrase such as "I need a moment to think" with different delivery:

- Calm: steady volume and pace, relaxed tone.
- Anxious or hesitant: uneven pace, uncertain tone, filler sounds, slight tremble.
- Breathy pause: speak a phrase, pause with an audible breath, then continue.
- Sad or low energy: lower pitch, softer voice, slower pacing.
- Angry or agitated: sharper attack, louder voice, faster clipped phrasing.

Empirical fixture check on this worktree with `uv run python scripts/test_wav.py tests/fixtures/calm.wav` and `tests/fixtures/anxious.wav`: all fixture chunks emitted `NEUTRAL` plus `Speech`, so these bundled files do not prove broad emotion-token coverage on this machine.

## CLI WAV test

```bash
uv run python -m voice_perception.perception tests/fixtures/calm.wav
uv run python scripts/test_wav.py tests/fixtures/anxious.wav
uv run python scripts/test_wav.py --compare tests/fixtures/calm.wav tests/fixtures/anxious.wav
```

## Troubleshooting

If startup tries to export ONNX, make sure `SENSEVOICE_MODEL_DIR` is `iic/SenseVoiceSmall-onnx` or a local ONNX model directory.
If PyAV cannot decode browser chunks, try Chrome or Firefox with `audio/webm;codecs=opus`.
If short English phrases appear as other languages during a demo, set `SENSEVOICE_LANGUAGE=en` and keep the default rolling live context enabled.
