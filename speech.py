"""
speech.py

Speech-to-Text module.
Records audio from the microphone (using simple energy-based Voice
Activity Detection to stop automatically when the user pauses) and
transcribes it to text using Faster-Whisper (runs fully offline, on CPU).
"""

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

import config
from utils import setup_logger, timeit

logger = setup_logger(__name__)

# Load the Whisper model once, at import time, so we don't reload it
# on every single turn of the conversation (that would be slow).
logger.info(f"Loading Whisper model '{config.STT_MODEL_SIZE}'...")
_model = WhisperModel(
    config.STT_MODEL_SIZE,
    device=config.STT_DEVICE,
    compute_type=config.STT_COMPUTE_TYPE
)
logger.info("Whisper model loaded.")


@timeit
def record_audio() -> np.ndarray:
    """
    Records audio from the microphone using simple energy-based
    Voice Activity Detection (VAD): recording stops automatically
    once the user has been silent for config.VAD_SILENCE_DURATION
    seconds, instead of always recording a fixed duration.

    Returns:
        A 1D numpy array of float32 audio samples containing only
        the recorded speech (plus the trailing silence buffer).
    """
    chunk_samples = int(config.SAMPLE_RATE * config.VAD_CHUNK_MS / 1000)

    recorded_chunks = []
    silence_duration = 0.0
    elapsed = 0.0
    speech_started = False

    logger.info("Listening... speak now (will stop automatically when you pause).")

    stream = sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="float32",
        device=config.INPUT_DEVICE,
        blocksize=chunk_samples
    )

    with stream:
        while True:
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk.flatten()
            recorded_chunks.append(chunk)

            # Measure volume of this chunk using RMS (root mean square)
            rms = np.sqrt(np.mean(chunk ** 2))
            chunk_duration = config.VAD_CHUNK_MS / 1000
            elapsed += chunk_duration

            if rms >= config.VAD_SILENCE_THRESHOLD:
                # Loud enough to count as speech
                speech_started = True
                silence_duration = 0.0
            else:
                silence_duration += chunk_duration

            # Stop conditions:
            # 1. We've heard speech, then enough silence afterward
            # 2. We've hit the hard safety cap (someone talking very long)
            if speech_started and elapsed >= config.VAD_MIN_RECORD_SECONDS:
                if silence_duration >= config.VAD_SILENCE_DURATION:
                    logger.info(f"Silence detected, stopping after {elapsed:.1f}s.")
                    break

            if elapsed >= config.VAD_MAX_RECORD_SECONDS:
                logger.warning(f"Max recording duration ({config.VAD_MAX_RECORD_SECONDS}s) reached.")
                break

    logger.info("Recording finished.")
    return np.concatenate(recorded_chunks)


@timeit
def transcribe(audio: np.ndarray) -> str:
    """
    Transcribes a numpy audio array to text using Faster-Whisper.

    Args:
        audio: 1D float32 numpy array of audio samples.

    Returns:
        The transcribed text (stripped of leading/trailing whitespace).
        Returns an empty string if nothing intelligible was detected.
    """
    segments, info = _model.transcribe(audio, language=config.STT_LANGUAGE)
    text = "".join(segment.text for segment in segments).strip()

    if text:
        logger.info(f"Transcription: '{text}' (lang confidence: {info.language_probability:.2f})")
    else:
        logger.warning("Transcription returned empty text.")

    return text


def listen() -> str:
    """
    High-level function: records audio from the mic and returns
    the transcribed text. This is the main function other modules
    (like assistant.py) should call.

    Returns:
        Transcribed text, or empty string if nothing was understood.
    """
    audio = record_audio()
    return transcribe(audio)