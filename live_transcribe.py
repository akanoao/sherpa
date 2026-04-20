#!/usr/bin/env python3

import sys
from pathlib import Path

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first: pip install sounddevice")
    sys.exit(-1)

import sherpa_onnx

def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please make sure you have downloaded and extracted the model."
    )

def create_recognizer():
    # Hardcoded paths for the Bilingual Zipformer model
    model_dir = "./sherpa-onnx-streaming-zipformer-en-2023-06-21/"
    
    tokens = f"{model_dir}tokens.txt"
    encoder = f"{model_dir}encoder-epoch-99-avg-1.onnx"
    decoder = f"{model_dir}decoder-epoch-99-avg-1.onnx"
    joiner = f"{model_dir}joiner-epoch-99-avg-1.onnx"

    # Verify all files exist before loading
    assert_file_exists(encoder)
    assert_file_exists(decoder)
    assert_file_exists(joiner)
    assert_file_exists(tokens)

    # Initialize the CPU-optimized recognizer
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=1,               # Best performance for streaming on CPU
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,  
        provider="cpu",              # Force CPU to avoid CUDA dependency crashes
    )
    return recognizer


def main():
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    # Print available devices and select default
    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'\nUsing default device: {devices[default_input_device_idx]["name"]}')

    print("Loading model into memory...")
    recognizer = create_recognizer()
    
    print("\n" + "="*50)
    print("🎙️  Started! Please speak in English or Chinese.")
    print("   Press [Ctrl+C] to exit.")
    print("="*50 + "\n")

    # The model uses 16 kHz, but capturing at 48 kHz and letting sherpa 
    # downsample it internally is often safer for standard Windows mics.
    sample_rate = 48000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    stream = recognizer.create_stream()
    
    # Sherpa's built-in display class handles terminal carriage returns nicely
    display = sherpa_onnx.Display()

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            is_endpoint = recognizer.is_endpoint(stream)
            result = recognizer.get_result(stream)

            display.update_text(result)
            display.display()

            if is_endpoint:
                if result:
                    display.finalize_current_sentence()
                    display.display()
                recognizer.reset(stream)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting application.")