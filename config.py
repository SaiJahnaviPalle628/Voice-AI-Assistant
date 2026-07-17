"""
config.py

Centralized configuration for the AI Assistant.
All tunable values (models, paths, timeouts, device IDs) live here
so the rest of the codebase never hardcodes them.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(BASE_DIR, "voices")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ── Audio Input (Microphone) ─────────────────────────────────────────
SAMPLE_RATE = 16000          # Whisper expects 16kHz audio
CHANNELS = 1                 # mono audio
INPUT_DEVICE = None          # None = use system default mic (Device 1 on your machine)

# ── Voice Activity Detection (VAD) ───────────────────────────────────
# Instead of always recording a fixed duration, we listen in small chunks
# and stop automatically once the user has been silent for a short while.
VAD_CHUNK_MS = 100                # size of each audio chunk analyzed, in milliseconds
VAD_SILENCE_THRESHOLD = 0.01       # RMS volume below this = considered "silence"
VAD_SILENCE_DURATION = 1.0         # stop after this many seconds of continuous silence
VAD_MAX_RECORD_SECONDS = 15        # hard safety cap, in case someone talks non-stop
VAD_MIN_RECORD_SECONDS = 0.5       # ignore silence detection for at least this long
                                    # (prevents stopping instantly if you pause before speaking)

# ── Speech-to-Text (Faster-Whisper) ──────────────────────────────────
STT_MODEL_SIZE = "tiny"      # tiny / base / small / medium / large — bigger = slower but more accurate
STT_DEVICE = "cpu"           # your laptop has no GPU, so this stays "cpu"
STT_COMPUTE_TYPE = "int8"    # int8 = fastest on CPU with minimal accuracy loss
STT_LANGUAGE = "en"

# ── LLM (Ollama) ──────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"   # default Ollama local server address
OLLAMA_KEEP_ALIVE = "30m"    # keep the model loaded in memory for 30 min of inactivity, avoids reload delay
LLM_MAX_TOKENS = 60   # hard cap on generated tokens, keeps replies short and fast
LLM_TIMEOUT_SECONDS = 8       # if LLM takes longer than this, trigger fallback filler
LLM_SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. "
    "Since your responses will be spoken out loud, keep them SHORT — "
    "1 to 2 sentences maximum, unless the user explicitly asks for "
    "a detailed explanation. Be conversational and natural, not robotic. "
    "Never use bullet points, numbered lists, or markdown formatting, "
    "since this is spoken audio, not text."
)

# ── Text-to-Speech (Piper) ────────────────────────────────────────────
TTS_VOICE_MODEL = os.path.join(VOICES_DIR, "en_US-lessac-medium.onnx")
TTS_OUTPUT_WAV = os.path.join(BASE_DIR, "response.wav")   # temp file for each spoken reply
OUTPUT_DEVICE = None          # None = use system default speakers (Device 3 on your machine)

# ── Fallback / Timeout Behavior ──────────────────────────────────────
FALLBACK_NO_SPEECH_DETECTED = "Sorry, I didn't catch that. Could you repeat that?"
FALLBACK_LLM_SLOW_FILLER = "Let me think about that for a moment..."
FALLBACK_LLM_FAILED = "I'm having trouble thinking right now. Please try again in a moment."
FALLBACK_TTS_FAILED_NOTICE = "(Voice output failed — showing text response instead.)"

# ── Logging ────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"            # DEBUG for more verbose logs while developing
LOG_FILE = os.path.join(BASE_DIR, "assistant.log")