"""
Sherpa-ONNX VITS Text-to-Speech provider.

Supports any VITS / MeloTTS / Coqui model that Sherpa-ONNX exposes through
its ``OfflineTts`` API.

Config dict keys
----------------
model        : path to the VITS .onnx file
tokens       : path to tokens.txt
data_dir     : path to espeak-ng-data directory  (optional – only for
               models that use eSpeak phonemisation)
lexicon      : path to lexicon file  (optional)
speaker_id   : int  (default 0)
noise_scale  : float  (default 0.667)
noise_scale_w: float  (default 0.8)
length_scale : float  (default 1.0)
provider     : "cpu" | "cuda" | "coreml"  (default "cpu")
num_threads  : int  (default 1)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SherpaOnnxTTSProvider:
    """Sherpa-ONNX VITS TTS adapter."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._tts = self._build_tts()

    # ------------------------------------------------------------------
    # TTSProvider interface
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    def synthesize(self, text: str, lang: str = "", speed: float = 1.0) -> np.ndarray:
        sid = self._config.get("speaker_id", 0)
        audio = self._tts.generate(text=text, sid=sid, speed=speed)
        return np.array(audio.samples, dtype=np.float32)

    def speak(self, text: str, lang: str = "", speed: float = 1.0) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            logger.error("sounddevice not installed: %s", exc)
            return

        samples = self.synthesize(text, lang=lang, speed=speed)
        sd.play(samples, samplerate=self._tts.sample_rate)
        sd.wait()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tts(self):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError(
                "sherpa_onnx is not installed. Run: pip install sherpa-onnx"
            ) from exc

        cfg = self._config

        vits_model = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=cfg["model"],
            tokens=cfg["tokens"],
            data_dir=cfg.get("data_dir", ""),
            lexicon=cfg.get("lexicon", ""),
            noise_scale=cfg.get("noise_scale", 0.667),
            noise_scale_w=cfg.get("noise_scale_w", 0.8),
            length_scale=cfg.get("length_scale", 1.0),
        )

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits_model,
            provider=cfg.get("provider", "cpu"),
            num_threads=cfg.get("num_threads", 1),
            debug=False,
        )

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            max_num_sentences=1,
        )

        return sherpa_onnx.OfflineTts(tts_config)
