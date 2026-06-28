"""Download and extract the STT, TTS, and NLLB translation assets."""

from __future__ import annotations

import os
import tarfile
import zipfile

import requests
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

MODELS_DIR = "./models"

NLLB_MODEL_REPO = "Adeptschneider/nllb-200-distilled-600M-int8"
NLLB_MODEL_DIR = os.path.join(MODELS_DIR, "nllb")
NLLB_TOKENIZER_REPO = "facebook/nllb-200-distilled-600M"
NLLB_TOKENIZER_DIR = os.path.join(MODELS_DIR, "nllb-tokenizer")
NLLB_TOKENIZER_FILES = [
    "config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

MODELS = [
    # STT models
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
    "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
    "https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip",
    # TTS models
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-en_US-ljspeech-medium.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-priyamvada-medium.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-hf-fanchen-C.tar.bz2",
]


def download_file(url: str, dest_path: str) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


def extract_file(file_path: str, dest_dir: str) -> None:
    print(f"Extracting {os.path.basename(file_path)}...")
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    elif file_path.endswith((".tar.bz2", ".tar.gz")):
        with tarfile.open(file_path, "r:*") as tar_ref:
            tar_ref.extractall(dest_dir)
    else:
        print(f"WARNING: unknown archive format for {file_path}")


def download_archive_models() -> None:
    for url in MODELS:
        filename = url.split("/")[-1]
        filepath = os.path.join(MODELS_DIR, filename)

        ext_dir = filename.replace(".tar.bz2", "").replace(".zip", "").replace(".tar.gz", "")
        if os.path.exists(os.path.join(MODELS_DIR, ext_dir)):
            print(f"{ext_dir} already exists. Skipping.")
            continue

        print(f"\nDownloading {filename}...")
        try:
            if not os.path.exists(filepath):
                download_file(url, filepath)
            extract_file(filepath, MODELS_DIR)
            os.remove(filepath)
        except Exception as exc:
            print(f"ERROR processing {filename}: {exc}")


def download_nllb_assets() -> None:
    if snapshot_download is None:
        print("\nWARNING: huggingface_hub is not installed; skipped NLLB downloads.")
        return

    if os.path.exists(os.path.join(NLLB_MODEL_DIR, "model.bin")):
        print("\nNLLB CTranslate2 model already exists. Skipping.")
    else:
        print("\nDownloading NLLB CTranslate2 translation model...")
        snapshot_download(
            repo_id=NLLB_MODEL_REPO,
            local_dir=NLLB_MODEL_DIR,
            local_dir_use_symlinks=False,
        )

    if os.path.exists(os.path.join(NLLB_TOKENIZER_DIR, "sentencepiece.bpe.model")):
        print("NLLB tokenizer already exists. Skipping.")
    else:
        print("Downloading NLLB tokenizer cache...")
        snapshot_download(
            repo_id=NLLB_TOKENIZER_REPO,
            local_dir=NLLB_TOKENIZER_DIR,
            local_dir_use_symlinks=False,
            allow_patterns=NLLB_TOKENIZER_FILES,
        )


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    download_archive_models()
    download_nllb_assets()
    print("\nAll models downloaded and extracted successfully.")


if __name__ == "__main__":
    main()
