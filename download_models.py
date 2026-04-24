"""
Downloads and extracts all required STT, TTS, and Translation models.
Run this script once to populate the ./models/ directory.
"""

import os
import zipfile
import tarfile
import requests
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

MODELS_DIR = "./models"

MODELS = [
    # --- STT Models ---
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2",
    "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
    "https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip",
    
    # --- TTS Models ---
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-en_US-ljspeech-medium.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-priyamvada-medium.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-hf-fanchen-C.tar.bz2"
]

def download_file(url: str, dest_path: str):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def extract_file(file_path: str, dest_dir: str):
    print(f"Extracting {os.path.basename(file_path)}...")
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif file_path.endswith('.tar.bz2') or file_path.endswith('.tar.gz'):
        with tarfile.open(file_path, 'r:*') as tar_ref:
            tar_ref.extractall(dest_dir)
    else:
        print(f"⚠️ Unknown archive format for {file_path}")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Download and extract standard archives
    for url in MODELS:
        filename = url.split('/')[-1]
        filepath = os.path.join(MODELS_DIR, filename)
        
        # Heuristic to check if already extracted
        ext_dir = filename.replace('.tar.bz2', '').replace('.zip', '').replace('.tar.gz', '')
        if os.path.exists(os.path.join(MODELS_DIR, ext_dir)):
            print(f"✅ {ext_dir} already exists. Skipping.")
            continue
            
        print(f"\nDownloading {filename}...")
        try:
            if not os.path.exists(filepath):
                download_file(url, filepath)
            extract_file(filepath, MODELS_DIR)
            # Clean up the compressed file to save space
            os.remove(filepath)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # 2. Download NLLB Model via HuggingFace
    if snapshot_download:
        print("\nDownloading NLLB Translation model...")
        snapshot_download(
            repo_id="Adeptschneider/nllb-200-distilled-600M-int8",
            local_dir=os.path.join(MODELS_DIR, "nllb"),
            local_dir_use_symlinks=False
        )

    print("\n🎉 All models downloaded and extracted successfully!")

if __name__ == "__main__":
    main()