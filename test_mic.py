import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# Settings
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
DURATION = 5         # seconds to record

print("Recording for 5 seconds... speak now!")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()  # wait until recording finishes
print("Recording done. Transcribing...")

# Load the Whisper model (small, runs on CPU)
# "base" model is a good balance of speed vs accuracy for CPU
model = WhisperModel("base", device="cpu", compute_type="int8")

# faster-whisper expects a 1D numpy array
audio_flat = audio.flatten()

segments, info = model.transcribe(audio_flat, language="en")

print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
print("Transcription:")
for segment in segments:
    print(f"  {segment.text}")