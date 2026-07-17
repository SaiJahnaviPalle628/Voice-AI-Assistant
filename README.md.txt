Voice AI Assistant

Real-time, offline, audio-in/audio-out conversational AI assistant.

->Features
- Voice input through microphone (Faster-Whisper, offline STT)
- Natural voice output (Piper TTS, offline)
- Offline LLM using Ollama (llama3.2:3b) - completely offline, no API keys required
- Silent mode detection (push-to-talk not required)
- Graceful fallback management in case of slow/undeliverable LLM response
- Detection of natural exit phrases ("bye", "exit", "goodbye", etc.)

->Architecture

Mic → STT (Faster-Whisper) → LLM (Ollama, llama3.2:3b) → TTS (Piper) → Speaker

The Fallback layer adds a timeout to the LLM call; if the call is timed out or fails,
the assistant replies with a natural filler message rather than failing.

-> Installation

1. Install Python 3.10+
2. Create virtualenv: