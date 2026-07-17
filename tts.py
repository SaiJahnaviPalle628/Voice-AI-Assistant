"""
tts.py

Text-to-Speech module.
Converts text into spoken audio using Piper (runs fully offline, on CPU)
and plays it through the default speakers.
"""

import wave

import sounddevice as sd
import soundfile as sf
from piper import PiperVoice

import config
from utils import setup_logger, timeit

logger = setup_logger(__name__)

# Load the Piper voice once at import time (avoids reloading on every turn)
logger.info(f"Loading Piper voice from {config.TTS_VOICE_MODEL}...")
_voice = PiperVoice.load(config.TTS_VOICE_MODEL)
logger.info("Piper voice loaded.")


@timeit
def synthesize(text: str) -> str:
    """
    Converts text to speech and saves it as a WAV file.

    Args:
        text: the text to convert to speech.

    Returns:
        Path to the generated WAV file.
    """
    with wave.open(config.TTS_OUTPUT_WAV, "wb") as wav_file:
        _voice.synthesize_wav(text, wav_file)
    return config.TTS_OUTPUT_WAV


@timeit
def play_audio(filepath: str) -> None:
    """
    Plays a WAV file through the default (or configured) output device.

    Args:
        filepath: path to the WAV file to play.
    """
    data, samplerate = sf.read(filepath, dtype="float32")
    sd.play(data, samplerate, device=config.OUTPUT_DEVICE)
    sd.wait()  # block until playback finishes


def speak(text: str) -> None:
    """
    High-level function: converts text to speech and plays it out loud.
    This is the main function other modules (like assistant.py) should call.

    Args:
        text: the text to speak.
    """
    if not text:
        logger.warning("speak() called with empty text, skipping.")
        return

    try:
        filepath = synthesize(text)
        play_audio(filepath)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        print(f"{config.FALLBACK_TTS_FAILED_NOTICE}\n{text}")