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
from pathlib import Path

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
        model_path, tokens_path = self._resolve_model_and_tokens(cfg)

        vits_model = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=model_path,
            tokens=tokens_path,
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

    def _resolve_model_and_tokens(self, cfg: dict) -> tuple[str, str]:
        model_path = Path(cfg["model"]).expanduser()
        tokens_path = Path(cfg["tokens"]).expanduser()

        if not tokens_path.is_file():
            raise FileNotFoundError(
                f"TTS tokens file not found: {tokens_path}. "
                "Update config.json -> tts.<lang>.tokens"
            )

        if not model_path.is_file():
            inferred_model = self._find_onnx_near(tokens_path)
            if inferred_model is None:
                raise FileNotFoundError(
                    f"TTS model file not found: {model_path}. "
                    "No .onnx model found in the same directory as tokens.txt. "
                    "Update config.json -> tts.<lang>.model"
                )
            logger.warning(
                "Configured TTS model %s was not found. Falling back to discovered model %s",
                model_path,
                inferred_model,
            )
            model_path = inferred_model

        if model_path.suffix.lower() != ".onnx":
            raise ValueError(
                f"TTS model must be an .onnx file, got: {model_path}. "
                "Update config.json -> tts.<lang>.model"
            )

        return str(model_path), str(tokens_path)

    def _find_onnx_near(self, tokens_path: Path) -> Path | None:
        base_dir = tokens_path.parent
        candidates = sorted(base_dir.glob("*.onnx"))
        if not candidates:
            return None

        preferred = base_dir / "model.onnx"
        if preferred in candidates:
            return preferred

        return candidates[0]
