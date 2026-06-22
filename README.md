# pipecat turn detection demo

Simple examples showing different approaches to barge-in and turn detection in voice agents, using [pipecat](https://github.com/pipecat-ai/pipecat) + Cartesia STT/TTS + Amazon Bedrock (Claude Haiku 4.5).

## Examples

| Script | Approach | How it works |
|---|---|---|
| `01-silero-vad.py` | Silence detection (VAD) | Silero ONNX model detects when audio energy drops below a threshold — simplest approach, no semantic understanding |
| `02-stt-with-turn-detection.py` | STT with built-in turn detection | Cartesia Ink-2 emits `turn.end` events over WebSocket — the model decides when you've finished speaking |
| `03-smart-turn.py` | Local smart turn model | Pipecat's open `smart-turn-v3.2` model understands speech semantics to decide if a thought is complete |

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env  # fill in CARTESIA_API_KEY
python 01-silero-vad.py
```

AWS credentials use the default boto3 chain (IAM role, SSO, etc.) — no key needed in `.env`.

## Glossary

**Barge-in / interruption detection** — The same thing, two names. The user speaks while the agent is still talking; the agent stops and starts listening immediately. Requires distinguishing the user's voice from the agent's own audio playback. Poor barge-in handling is one of the most frustrating voice UX problems.

**Silence detection / VAD (Voice Activity Detection)** — Detects when audio energy falls silent to infer the user has stopped speaking. Fast and lightweight, but fooled by mid-sentence pauses and thinking gaps.

**Turn detection** — Deciding *when* the user has finished their full conversational turn and the agent should respond. More intelligent than silence detection — accounts for sentence completion, not just audio energy.

**Smart turn / semantic turn detection** — Uses a small language or acoustic model to predict whether the user's utterance is semantically complete, even if they paused mid-thought. Reduces false triggers from filler words ("um", "uh") and natural pauses.

**End-of-utterance (EOU)** — The specific moment a turn detection model commits to: the user has finished and the agent should begin responding. Getting this wrong in either direction hurts — too early cuts the user off, too late creates awkward lag.

## Why it matters

The gap between a clunky IVR and a natural voice agent is mostly turn-taking. A bad EOU model either interrupts the user constantly or adds a noticeable delay before every response. The three examples here illustrate the tradeoff: simplicity (VAD) vs. accuracy (semantic turn models) vs. latency (server-driven).
