"""
ModelManager – central registry for STT, TTS, and Translation providers.

Models are loaded lazily on first use and cached for the lifetime of the
session.  The manager reads its routing table from the application config
dict (loaded from ``config.json`` by default).

Config schema (abbreviated)
----------------------------
::

    {
      "stt": {
        "<lang>": { "engine": "sherpa_onnx" | "vosk", ...engine-specific keys }
      },
      "tts": {
        "<lang>": { "engine": "sherpa_onnx", ...engine-specific keys }
      },
      "translation": {
        "engine": "argos" | "ctranslate2",
        ...engine-specific keys
      }
    }
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages lazy loading and routing of STT, TTS, and Translation models.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._stt_cache: dict = {}
        self._tts_cache: dict = {}
        self._translation_provider = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stt_provider(self, language: str):
        """Return the STT provider for *language*, loading it if necessary."""
        if language not in self._stt_cache:
            logger.info("Loading STT provider for language %r", language)
            self._stt_cache[language] = self._build_stt(language)
        return self._stt_cache[language]

    def get_tts_provider(self, language: str):
        """Return the TTS provider for *language*, loading it if necessary."""
        if language not in self._tts_cache:
            logger.info("Loading TTS provider for language %r", language)
            self._tts_cache[language] = self._build_tts(language)
        return self._tts_cache[language]

    def get_translation_provider(self):
        """Return the global translation provider, loading it if necessary."""
        if self._translation_provider is None:
            logger.info("Loading translation provider")
            self._translation_provider = self._build_translation()
        return self._translation_provider

    def has_stt(self, language: str) -> bool:
        return language in self._config.get("stt", {})

    def has_tts(self, language: str) -> bool:
        return language in self._config.get("tts", {})

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_stt(self, language: str):
        stt_cfg = self._config.get("stt", {})
        if language not in stt_cfg:
            raise ValueError(
                f"No STT configuration found for language {language!r}. "
                "Add an entry under 'stt' in your config.json."
            )
        cfg = dict(stt_cfg[language])
        cfg.setdefault("language", language)
        engine = cfg.get("engine", "sherpa_onnx")

        if engine == "sherpa_onnx":
            from .stt.sherpa_provider import SherpaOnnxSTTProvider

            return SherpaOnnxSTTProvider(cfg)

        if engine == "vosk":
            from .stt.vosk_provider import VoskSTTProvider

            return VoskSTTProvider(cfg)

        raise ValueError(f"Unknown STT engine: {engine!r}")

    def _build_tts(self, language: str):
        tts_cfg = self._config.get("tts", {})
        if language not in tts_cfg:
            raise ValueError(
                f"No TTS configuration found for language {language!r}. "
                "Add an entry under 'tts' in your config.json, or use --no-tts."
            )
        cfg = dict(tts_cfg[language])
        cfg.setdefault("language", language)
        engine = cfg.get("engine", "sherpa_onnx")

        if engine == "sherpa_onnx":
            from .tts.sherpa_provider import SherpaOnnxTTSProvider

            return SherpaOnnxTTSProvider(cfg)

        raise ValueError(f"Unknown TTS engine: {engine!r}")

    def _build_translation(self):
        trans_cfg = self._config.get("translation", {})
        engine = trans_cfg.get("engine", "argos")

        if engine == "argos":
            from .translation.argos_provider import ArgosTranslationProvider

            return ArgosTranslationProvider(trans_cfg)

        if engine == "ctranslate2":
            from .translation.ctranslate2_provider import CTranslate2TranslationProvider

            return CTranslate2TranslationProvider(trans_cfg)

        raise ValueError(f"Unknown translation engine: {engine!r}")
