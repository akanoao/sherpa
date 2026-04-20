# Documentation: Real-Time Speech Recognition (`download_model.py`)

## 📖 General Description

The `download_model.py` script performs real-time speech-to-text recognition using the **Sherpa-ONNX** Python API. It captures live audio from your system's default microphone using the `sounddevice` library, automatically handles sample rate conversion (e.g., from 48kHz microphone input down to the model's 16kHz requirement), and uses speech endpoint detection to determine when you have stopped speaking.

_Note: Despite the file name being `download_model.py`, this script is actually used for real-time speech recognition (Speech-to-Text) using your microphone. It does not download models automatically; it expects models to be already downloaded and passed as arguments._

## ⚙️ Requirements

To run this script, you need Python installed along with the following libraries:

```bash
pip install sounddevice sherpa-onnx numpy
```

_Note: Depending on your operating system (specifically Linux), `sounddevice` might also require installing system-level portaudio libraries (e.g., `sudo apt-get install libportaudio2`)._

## 📥 Process of Downloading the Model

The script requires a pre-trained **Transducer** model (which consists of an encoder, decoder, joiner, and a tokens text file).

1. Visit the [Sherpa-ONNX Pre-trained Models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html) page.
2. Choose an appropriate streaming/online Transducer model (e.g., Zipformer).
3. Download the `.onnx` files and the `tokens.txt` file into a folder in your workspace.

_Note: Based on your workspace structure, you already have models downloaded! You can use the `sherpa-onnx-streaming-zipformer-en-2023-06-21` model directory that is already present._

## 🚀 How to Run the Script

You must provide the paths to the model files when executing the script. Using the English Zipformer model already available in your directory, you would run:

```bash
python download_model.py \
  --tokens sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt \
  --encoder sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.onnx \
  --decoder sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx \
  --joiner sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.onnx
```

### Optional Arguments

- `--decoding-method`: Set to `greedy_search` (default) or `modified_beam_search`.
- `--provider`: execution provider, you can use `cpu` (default), `cuda` (if you have an NVIDIA GPU), or `coreml` (for Apple Silicon).
- `--hotwords-file`: Path to a text file containing phrases you want the model to favorably bias towards (good for specific jargon/names).

## 🔗 Documentation Links

- **Official Sherpa-ONNX Documentation**: [https://k2-fsa.github.io/sherpa/onnx/](https://k2-fsa.github.io/sherpa/onnx/)
- **Pre-trained Model Repository**: [Sherpa ONNX Models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html)
- **Sounddevice Documentation**: [https://python-sounddevice.readthedocs.io/](https://python-sounddevice.readthedocs.io/)
