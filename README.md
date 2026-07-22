# Voice Perception Service

Real-time FastAPI microservice for paralinguistic voice state.
It ingests microphone audio over WebSocket and exposes live speech activity, acoustic voice state, transcript, audio events, experimental emotion, and hesitation or stress scores.
Use the bundled browser page to test calm vs anxious speech without any other service.

## Requirements

- Python 3.11
- uv (`brew install uv` if missing)
- About 1.5 GB free disk for SenseVoice plus Emotion2Vec+ model caches
- Browser microphone access

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv run uvicorn voice_perception.main:app --reload
```

Open http://localhost:8000 and click Start for live mode. First startup can take several minutes while SenseVoice and Emotion2Vec+ download and load.

Emotion2Vec+ base (`iic/emotion2vec_plus_base`) is enabled by default as the primary exact-emotion classifier for the hackathon demo. Treat it as demo use until its license is verified for broader shipping. To run only the acoustic plus SenseVoice lanes, set `SER_ENABLED=false`.

The page also includes a separate record then submit area for one-shot testing. Click Record, speak for a few seconds, click Stop, then click Submit to upload the completed MediaRecorder audio blob to `POST /classify` and render a single classification result.

Both live and one-shot paths apply a configurable no-speech guard before SenseVoice or Emotion2Vec+ inference. Audio below `NO_SPEECH_RMS_THRESHOLD`, `NO_SPEECH_MIN_NON_SILENT_RATIO`, or `NO_SPEECH_MIN_NON_SILENT_SECONDS` returns an idle no-speech state instead of a hallucinated transcript, `Speech` event, or emotion label.

Live microphone inference uses a rolling PCM context window so SenseVoice and Emotion2Vec+ do not process each MediaRecorder slice in isolation. The browser sends 200 ms chunks for fast acoustic updates while model inference waits for the rolling context.
For an English-only demo, run with `SENSEVOICE_LANGUAGE=en`; the default remains `auto` for multilingual use.

## API summary

See `AGENTS.md` for the full API contract and architecture.

- `POST /session/start` returns a `session_id`
- `WebSocket /audio/{session_id}` accepts MediaRecorder binary audio chunks
- `GET /state/{session_id}` returns live transcript, emotion, events, hesitation score, chunk count, `voice_state`, `signals`, `signal_events`, and `ser` debug fields
- `POST /classify` accepts a multipart `file` audio upload and returns one-shot transcript, emotion, events, hesitation score, latency, audio sample count, `voice_state`, `signals`, `signal_events`, and `ser` debug fields
- `POST /session/{session_id}/end` cleans up a session
- `GET /health` reports service health and model load state

## Curl integration example

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/session/start | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
curl -s http://localhost:8000/state/$SESSION_ID
curl -s -X POST http://localhost:8000/session/$SESSION_ID/end
```

## One-shot browser test cases

The service responds to paralinguistic audio cues, not guaranteed sentiment of the words alone. The reliable product signal is `voice_state` plus `signals` such as speech activity, arousal, stress, hesitation, speaking confidence, and score drivers. Exact emotion is experimental: Emotion2Vec+ is primary when enabled, while SenseVoice emotion tokens are secondary debug hints.

For the clearest comparison, repeat a neutral phrase such as "I need a moment to think" with different delivery:

- Calm: steady volume and pace, relaxed tone.
- Anxious or hesitant: uneven pace, uncertain tone, filler sounds, slight tremble.
- Breathy pause: speak a phrase, pause with an audible breath, then continue.
- Sad or low energy: lower pitch, softer voice, slower pacing.
- Angry or agitated: sharper attack, louder voice, faster clipped phrasing.

Empirical benchmark reports in firstmate data selected Emotion2Vec+ base because it separated the bundled calm and anxious fixtures while SenseVoice emotion tokens stayed neutral or unknown. The acoustic lane also separates the fixtures through pause and silence drivers without depending on exact emotion.

## CLI WAV test

```bash
uv run python -m voice_perception.perception tests/fixtures/calm.wav
uv run python scripts/test_wav.py tests/fixtures/anxious.wav
uv run python scripts/test_wav.py --compare tests/fixtures/calm.wav tests/fixtures/anxious.wav
```

## Troubleshooting

If startup tries to export ONNX, make sure `SENSEVOICE_MODEL_DIR` is `iic/SenseVoiceSmall-onnx` or a local ONNX model directory.
If Emotion2Vec+ download or memory use blocks local testing, set `SER_ENABLED=false` to keep SenseVoice transcript/events and the acoustic lane running, or set `SER_MODEL_DIR` to a pre-downloaded `iic/emotion2vec_plus_base` cache.
If PyAV cannot decode browser chunks, try Chrome or Firefox with `audio/webm;codecs=opus`.
If short English phrases appear as other languages during a demo, set `SENSEVOICE_LANGUAGE=en` and keep the default rolling live context enabled.
