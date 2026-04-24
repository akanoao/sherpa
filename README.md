# SherpaConnect 🎙️🌐

Real-time, fully-offline, multilingual two-way voice communication.

> **Core idea:** each side does local Speech-to-Text, sends only _text_ over the
> network, and the receiver translates the text locally before playing it back
> through a local Text-to-Speech engine. Raw audio never leaves the device.

```
┌──────────────── Speaker A  (e.g. speaks English) ──────────────────┐
│                                                                      │
│  Microphone ─► STT (Sherpa/Vosk) ─► TextEvent ─►─┐                 │
│                                                    │ WebSocket relay │
│  Speaker  ◄─ TTS (Sherpa VITS) ◄─ Translation ◄──┘                 │
└──────────────────────────────────────────────────────────────────────┘
                  ▲                             │
             (text only)                   (text only)
                  │                             ▼
┌──────────────── Speaker B  (e.g. speaks Hindi) ────────────────────┐
│                                                                      │
│  Microphone ─► STT (Vosk hi) ─► TextEvent ─►─────────────────────┐ │
│                                                                    │ │
│  Speaker  ◄─ TTS (Sherpa VITS hi) ◄─ Translation ◄───────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
sherpa_talk/
├── core/
│   ├── packet.py               # TextEvent dataclass (JSON wire format)
│   ├── model_manager.py        # routes STT / TTS / Translation by language
│   ├── stt/
│   │   ├── base.py             # STTProvider ABC + TranscriptEvent
│   │   ├── sherpa_provider.py  # Sherpa-ONNX streaming (Zipformer / Paraformer)
│   │   └── vosk_provider.py    # Vosk (Hindi, Japanese, …)
│   ├── tts/
│   │   ├── base.py             # TTSProvider ABC
│   │   └── sherpa_provider.py  # Sherpa-ONNX VITS
│   └── translation/
│       ├── base.py             # TranslationProvider ABC
│       ├── argos_provider.py   # Argos Translate (MVP, easiest setup)
│       └── ctranslate2_provider.py  # CTranslate2 + MarianMT / NLLB (production)
├── transport/
│   ├── base.py                 # TransportBase ABC
│   ├── ws_server.py            # WebSocket relay server (room-based broadcast)
│   └── ws_client.py            # WebSocket client with auto-reconnect
└── app/
    ├── ui.py                   # Thread-safe terminal UI
    └── client.py               # Main orchestrator (STT→send, recv→translate→TTS)
main.py                         # CLI entry point
config.example.json             # Fully-annotated example configuration
```

---

## Installation

```bash
# 1. Clone and enter the repo
git clone https://github.com/akanoao/sherpa
cd sherpa

# 2. Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# Linux only: PortAudio system library (required by sounddevice)
sudo apt-get install libportaudio2
```

---

## Quick Start

### 1. Download models

#### STT — English (Sherpa-ONNX Zipformer)

```bash
mkdir -p models && cd models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
```

#### STT — Hindi (Vosk small)

```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip
unzip vosk-model-small-hi-0.22.zip
```

#### TTS — English (VITS LJSpeech)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-en_US-ljspeech-medium.tar.bz2
tar xf vits-icefall-en_US-ljspeech-medium.tar.bz2
```

#### TTS — Hindi (VITS MMS)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-priyamvada-medium.tar.bz2
tar xf vits-piper-hi_IN-priyamvada-medium.tar.bz2
```

### 2. Configure

```bash
cp config.example.json config.json
# Edit config.json to adjust model paths for your setup.
```

### 3. Install translation language pair

```bash
# Install the en → hi and hi → en packages (one-time, ~50 MB each)
python main.py install-lang --from en --to hi
python main.py install-lang --from hi --to en
```

### 4. Start the relay server (on one of the machines, or a VPS)

```bash
python main.py serve --port 8765
```

### 5. Start the clients

**Machine A** (Alice speaks English, wants Hindi):

```bash
python main.py talk \
  --input-lang en \
  --output-lang hi \
  --server ws://localhost:8765/myroom \
  --speaker-id alice
```

**Machine B** (Bob speaks Hindi, wants English):

```bash
python main.py talk \
  --input-lang hi \
  --output-lang en \
  --server ws://<IP-of-relay>:8765/myroom \
  --speaker-id bob
```

---

## CLI Reference

```
python main.py serve [--host HOST] [--port PORT]
    Start the WebSocket relay server.

python main.py talk
    --input-lang LANG       Language you speak (e.g. en, hi, zh, ja)
    --output-lang LANG      Language you want to hear/read
    --server URI            Relay server URI (ws://host:port/room)
    [--config FILE]         Path to config.json  (default: config.json)
    [--speaker-id ID]       Your display name    (default: random)
    [--session ID]          Session / group ID   (default: random)
    [--no-tts]              Disable TTS playback
    [--no-original]         Hide untranslated remote text
    [--tts-speed FLOAT]     Synthesis speed      (default: 1.0)
    [--log-level LEVEL]     DEBUG/INFO/WARNING/ERROR

python main.py install-lang --from LANG --to LANG
    Download and install an Argos Translate language-pair package.

python main.py list-langs
    List installed Argos Translate language pairs.
```

---

## Translation Engines

### Argos Translate (default, easiest)

Fully offline, no model conversion needed. Language packages are ~50 MB each
and are downloaded once with `install-lang`.

```json
"translation": { "engine": "argos" }
```

### CTranslate2 + MarianMT / NLLB (faster, production)

For heavy use or larger models, convert HuggingFace weights to CTranslate2 INT8:

```bash
pip install ctranslate2 transformers sentencepiece

# MarianMT direct pair (fast, ~300 MB)
ct2-opus-mt-converter \
  --model Helsinki-NLP/opus-mt-en-hi \
  --output_dir ./models/ct2/opus-mt-en-hi \
  --quantization int8

# NLLB-200 distilled (multilingual fallback, ~1.3 GB)
ct2-transformers-converter \
  --model facebook/nllb-200-distilled-600M \
  --output_dir ./models/ct2/nllb-200-distilled-600M \
  --quantization int8 --force
```

Then set `"engine": "ctranslate2"` and `"models_dir": "./models/ct2"` in
`config.json`.

---

## Adding a New Language

1. **STT:** Add an entry under `"stt"` in `config.json` pointing to a Sherpa-ONNX
   or Vosk model for that language.
2. **TTS:** Add an entry under `"tts"` pointing to a Sherpa-ONNX VITS model.
3. **Translation:** Install the Argos language pair (`install-lang`) or convert
   a MarianMT model if using CTranslate2.

---

## Privacy & Security

- Raw audio is processed **entirely locally** – only transcribed text leaves
  the device.
- The relay server never stores any messages; it only forwards them in real
  time.
- For end-to-end encrypted transport, place the relay server behind a TLS
  reverse proxy (nginx, Caddy) and use a `wss://` URI.

---

## Supported STT Models

| Language        | Engine             | Model                                                                     |
| --------------- | ------------------ | ------------------------------------------------------------------------- |
| English         | Sherpa-ONNX        | [Zipformer EN 2023-06-21](https://github.com/k2-fsa/sherpa-onnx/releases) |
| Chinese/English | Sherpa-ONNX        | [Paraformer bilingual](https://github.com/k2-fsa/sherpa-onnx/releases)    |
| Hindi           | Vosk               | [vosk-model-small-hi-0.22](https://alphacephei.com/vosk/models)           |
| Japanese        | Vosk               | [vosk-model-ja-0.22](https://alphacephei.com/vosk/models)                 |
| Many others     | Vosk / Sherpa-ONNX | See respective model pages                                                |

## Supported TTS Models

See [Sherpa-ONNX TTS model releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)
for a full list of VITS models covering English, Hindi, Chinese, Spanish,
German, French, and many others.
