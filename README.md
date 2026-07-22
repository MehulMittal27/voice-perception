# Voice Perception Service

Real-time FastAPI microservice for paralinguistic voice state.
It ingests microphone audio over WebSocket and exposes live emotion, audio events, transcript, and hesitation score.
Use the bundled browser page to test calm vs anxious speech without any other service.

## Requirements

- Python 3.11
- About 500 MB free disk for the SenseVoice model cache
- Browser microphone access

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
uvicorn voice_perception.main:app --reload
```

Open http://localhost:8000 and click Start. First startup can take 1 to 3 minutes while SenseVoice downloads and loads.

## API summary

See `AGENTS.md` for the full API contract and architecture.

- `POST /session/start` returns a `session_id`
- `WebSocket /audio/{session_id}` accepts MediaRecorder binary audio chunks
- `GET /state/{session_id}` returns live transcript, emotion, events, hesitation score, and chunk count
- `POST /session/{session_id}/end` cleans up a session
- `GET /health` reports service health and model load state

## Curl integration example

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/session/start | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
curl -s http://localhost:8000/state/$SESSION_ID
curl -s -X POST http://localhost:8000/session/$SESSION_ID/end
```

## CLI WAV test

```bash
python -m voice_perception.perception tests/fixtures/calm.wav
python scripts/test_wav.py tests/fixtures/anxious.wav
python scripts/test_wav.py --compare tests/fixtures/calm.wav tests/fixtures/anxious.wav
```

## Troubleshooting

If startup tries to export ONNX, make sure `SENSEVOICE_MODEL_DIR` is `iic/SenseVoiceSmall-onnx` or a local ONNX model directory.
If PyAV cannot decode browser chunks, try Chrome or Firefox with `audio/webm;codecs=opus`.
