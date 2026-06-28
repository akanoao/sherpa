# SherpaConnect

Real-time, offline-first multilingual voice communication.

Each side performs local speech-to-text, sends only text over the network, then
the receiver translates locally with NLLB/CTranslate2 and optionally plays the
result through local text-to-speech. Raw audio does not leave the device in the
text translation pipeline.

## Architecture

```text
sherpa_talk/
  app/
    client.py                 Main session orchestrator
    media.py                  WebRTC media tracks and video UI
    ui.py                     Thread-safe terminal UI
  core/
    packet.py                 TextEvent and SignalingEvent wire packets
    model_manager.py          Lazy provider loading and routing
    stt/
      base.py                 STT provider interface
      sherpa_provider.py      Sherpa-ONNX streaming STT
      vosk_provider.py        Vosk streaming STT
    tts/
      base.py                 TTS provider interface
      sherpa_provider.py      Sherpa-ONNX VITS TTS
    translation/
      base.py                 Translation provider interface
      ctranslate2_provider.py Local NLLB + CTranslate2 translation
  transport/
    ws_server.py              WebSocket relay server
    ws_client.py              WebSocket client with reconnect
    webrtc_manager.py         WebRTC offer/answer media connection
main.py                       CLI entry point
download_models.py            Model and tokenizer downloader
config.example.json           Example configuration
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux, install PortAudio for `sounddevice`.

## Download Models

```bash
python download_models.py
```

This downloads STT/TTS assets, the local NLLB CTranslate2 model to
`./models/nllb`, and the offline tokenizer cache to `./models/nllb-tokenizer`.

## Configure

```bash
copy config.example.json config.json
```

The translation section should stay NLLB-only:

```json
"translation": {
  "engine": "nllb",
  "models_dir": "./models",
  "nllb_dir": "nllb",
  "tokenizer_dir": "./models/nllb-tokenizer",
  "device": "cpu",
  "inter_threads": 1
}
```

## Run

Start the relay:

```bash
python main.py serve --port 8765
```

Machine A:

```bash
python main.py talk ^
  --input-lang en ^
  --output-lang hi ^
  --server ws://localhost:8765/myroom ^
  --speaker-id alice
```

Machine B:

```bash
python main.py talk ^
  --input-lang hi ^
  --output-lang en ^
  --server ws://<relay-ip>:8765/myroom ^
  --speaker-id bob
```

## CLI

```text
python main.py serve [--host HOST] [--port PORT]

python main.py talk
    --input-lang LANG
    --output-lang LANG
    --server URI
    [--config FILE]
    [--speaker-id ID]
    [--session ID]
    [--no-tts]
    [--no-original]
    [--tts-speed FLOAT]
    [--call]
    [--log-level DEBUG|INFO|WARNING|ERROR]
```

## Translation

Argos Translate has been removed. Translation now uses only local NLLB through
CTranslate2. The runtime loads both the model and tokenizer from local paths and
does not call Hugging Face during translation.

To add a language, make sure the language exists in `NLLB_LANG_MAP` in
`sherpa_talk/core/translation/ctranslate2_provider.py`.
