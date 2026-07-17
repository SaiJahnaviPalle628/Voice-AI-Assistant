from piper import PiperVoice
import wave

# Load the voice model we downloaded
voice = PiperVoice.load("voices/en_US-lessac-medium.onnx")

# Text we want to convert to speech
text = "Hello! This is a test of my voice assistant. If you can hear this clearly, text to speech is working."

# Generate audio and save it as a WAV file
with wave.open("test_output.wav", "wb") as wav_file:
    voice.synthesize_wav(text, wav_file)

print("Done! Audio saved to test_output.wav")