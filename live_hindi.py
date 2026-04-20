import queue
import sys
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# 1. Point to the extracted Hindi model folder
model_path = "vosk-model-small-hi-0.22"
# model_path = "vosk-model-ja-0.22"
print(f"Loading Hindi model from {model_path} into CPU...")
model = Model(model_path)

# Initialize the recognizer at 16kHz
recognizer = KaldiRecognizer(model, 16000)

# Create a queue to securely pass audio data from the mic to Vosk
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    # Put the raw audio bytes into the queue
    q.put(bytes(indata))

print("\n" + "="*50)
print("🎙️  Microphone activated! Speak in Hindi.")
print("   Press [Ctrl+C] to exit.")
print("="*50 + "\n")

try:
    # Open the microphone stream at 16kHz, 16-bit, Mono
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            # Grab audio from the queue
            data = q.get()
            
            # If a sentence is finished (the user paused)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    # Clear the line and print the finalized sentence
                    print(" " * 50, end="\r") 
                    print(f"✅ Final: {text}\n")
                    with open("transcriptions.txt", "a", encoding="utf-8") as f:
                        f.write(text + "\n")
            
            # If the user is currently speaking (live streaming)
            else:
                partial_result = json.loads(recognizer.PartialResult())
                partial_text = partial_result.get("partial", "")
                if partial_text:
                    # Overwrite the current line with the live guess
                    print(f"Live: {partial_text}", end="\r", flush=True)

except KeyboardInterrupt:
    print("\n\nExiting application...")