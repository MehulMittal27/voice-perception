# Voice Perception Service

Real-time FastAPI microservice for paralinguistic speech perception.
It ingests microphone audio over WebSocket and exposes live speech activity, transcript, audio events, Emotion2Vec+ emotion, and hesitation or stress scores.
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

Emotion2Vec+ base (`iic/emotion2vec_plus_base`) is enabled by default as the primary exact-emotion classifier for the hackathon demo. Treat it as demo use until its license is verified for broader shipping. Acoustic features are supporting guardrails and numeric metrics, not categorical emotion labels. To run only the acoustic plus SenseVoice lanes, set `SER_ENABLED=false`.

The page also includes a separate record then submit area for one-shot testing. Click Record, speak for a few seconds, click Stop, then click Submit to upload the completed MediaRecorder audio blob to `POST /classify` and render a single classification result.

Both live and one-shot paths apply a configurable no-speech guard before SenseVoice or Emotion2Vec+ inference. Audio below `NO_SPEECH_RMS_THRESHOLD`, `NO_SPEECH_MIN_NON_SILENT_RATIO`, or `NO_SPEECH_MIN_NON_SILENT_SECONDS` returns an idle no-speech state instead of a hallucinated transcript, `Speech` event, or emotion label.

Live microphone inference uses a rolling PCM context window so SenseVoice and Emotion2Vec+ do not process each MediaRecorder slice in isolation. The browser sends 200 ms chunks for fast acoustic updates while model inference waits for the rolling context. Live state stabilizes the primary `emotion` fields across rolling Emotion2Vec+ windows to avoid flicker, while raw live model reads are retained in `live_raw_emotion`, `live_raw_emotion_confidence`, `live_stabilized_emotion`, and `ser` debug fields. One-shot `/classify` remains a direct full-clip Emotion2Vec+ result.
For an English-only demo, run with `SENSEVOICE_LANGUAGE=en`; the default remains `auto` for multilingual use.

## API summary

See `AGENTS.md` for the full API contract and architecture.

- `POST /session/start` returns a `session_id`
- `WebSocket /audio/{session_id}` accepts MediaRecorder binary audio chunks
- `GET /state/{session_id}` returns live transcript, stabilized live Emotion2Vec+ emotion, events, hesitation score, chunk count, `signals`, `signal_events`, `score_drivers`, and `ser` debug fields. Raw rolling-window emotion reads are available in `live_raw_emotion`, `live_raw_emotion_confidence`, and `live_stabilized_emotion`. A legacy `voice_state` object may still appear for compatibility, but it is debug-only and not a product emotion label.
- `POST /classify` accepts a multipart `file` audio upload and returns one-shot transcript, Emotion2Vec+ emotion, events, hesitation score, latency, audio sample count, `signals`, `signal_events`, `score_drivers`, and `ser` debug fields. A legacy `voice_state` object may still appear for compatibility, but it is debug-only.
- `POST /session/{session_id}/end` cleans up a session
- `GET /health` reports service health and model load state

## Curl integration example

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/session/start | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
curl -s http://localhost:8000/state/$SESSION_ID
curl -s -X POST http://localhost:8000/session/$SESSION_ID/end
```

## One-shot browser test cases

The service responds to paralinguistic audio cues, not guaranteed sentiment of the words alone. The product emotion source is Emotion2Vec+ when enabled, with SenseVoice emotion tokens retained only as secondary debug hints. Acoustic features remain useful as guardrails and numeric metrics such as speech activity, arousal, stress, hesitation, speaking confidence, silence ratio, and score drivers. Do not present the legacy acoustic `voice_state` categorical label as an emotion result.

For the clearest comparison, repeat a neutral phrase such as "I need a moment to think" with different delivery:

- Calm: steady volume and pace, relaxed tone.
- Anxious or hesitant: uneven pace, uncertain tone, filler sounds, slight tremble.
- Breathy pause: speak a phrase, pause with an audible breath, then continue.
- Sad or low energy: lower pitch, softer voice, slower pacing.
- Angry or agitated: sharper attack, louder voice, faster clipped phrasing.

Empirical benchmark reports in firstmate data selected Emotion2Vec+ base because it separated the bundled calm and anxious fixtures while SenseVoice emotion tokens stayed neutral or unknown. The acoustic lane also separates the fixtures through pause and silence numeric drivers without depending on categorical acoustic labels.

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
