import sherpa_onnx
import wave
import numpy as np

# 1. Set the paths to the model files you just extracted
# Note: We are using the .int8 versions for faster CPU processing
model_dir = "./sherpa-onnx-zipformer-gigaspeech-2023-12-12/"

tokens = f"{model_dir}tokens.txt"
encoder = f"{model_dir}encoder-epoch-30-avg-1.int8.onnx"
decoder = f"{model_dir}decoder-epoch-30-avg-1.onnx"
joiner = f"{model_dir}joiner-epoch-30-avg-1.int8.onnx"
test_audio = f"{model_dir}test_wavs/output.wav"

print("Loading Offline Model into CPU...")

# 2. Initialize the OFFLINE recognizer
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    tokens=tokens,
    encoder=encoder,
    decoder=decoder,
    joiner=joiner,
    num_threads=1,
    sample_rate=16000,
    feature_dim=80,
    provider="cpu"
)

# 3. Read the test audio file
print(f"Reading audio from: {test_audio}")
with wave.open(test_audio, "rb") as f:
    # Ensure it's 16kHz, mono, 16-bit
    assert f.getframerate() == 16000
    assert f.getnchannels() == 1
    assert f.getsampwidth() == 2
    
    # Read the raw audio bytes and convert to float32 array
    audio_data = f.readframes(f.getnframes())
    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

# 4. Transcribe the audio
print("\nTranscribing...")
stream = recognizer.create_stream()
stream.accept_waveform(16000, samples)

recognizer.decode_stream(stream)

# 5. Print the final result
result = stream.result.text
print("\n" + "="*50)
print(f"Result: {result}")
print("="*50 + "\n")