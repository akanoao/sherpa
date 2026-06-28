"""NLLB translation provider backed by CTranslate2.

The app uses one local NLLB CTranslate2 model for all language pairs. Runtime
translation is fully offline: both the CT2 model and the Hugging Face tokenizer
must exist on disk before the provider is constructed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Mapping from ISO-639-1 short code to FLORES-200 / NLLB tag.
NLLB_LANG_MAP: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ko": "kor_Hang",
    "it": "ita_Latn",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "id": "ind_Latn",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
    "fa": "pes_Arab",
    "he": "heb_Hebr",
    "th": "tha_Thai",
    "ms": "msa_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
}


class CTranslate2TranslationProvider:
    """NLLB/CTranslate2 translation provider."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._models_dir: str = config.get("models_dir", "./models")
        self._device: str = config.get("device", "cpu")
        self._inter_threads: int = config.get("inter_threads", 1)
        self._intra_threads: int = config.get("intra_threads", 0)
        self._nllb_dir_name: str = config.get("nllb_dir", "nllb")
        self._tokenizer_dir: str = config.get("tokenizer_dir", "./models/nllb-tokenizer")

        self._translators: dict[str, object] = {}
        self._tokenizers: dict[str, object] = {}

        self._nllb_model_path = self._resolve_nllb_model_dir()
        self._tokenizer_path = self._resolve_tokenizer_dir()

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text through local NLLB, returning the original on inference failure."""
        if not text.strip() or source_lang == target_lang:
            return text

        normalized_text = text.capitalize()
        return self._nllb_translate(
            normalized_text,
            self._nllb_model_path,
            source_lang,
            target_lang,
        )

    def is_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        return source_lang in NLLB_LANG_MAP and target_lang in NLLB_LANG_MAP

    def _resolve_nllb_model_dir(self) -> str:
        configured = Path(self._nllb_dir_name).expanduser()
        path = configured if configured.is_absolute() else Path(self._models_dir) / configured
        path = path.resolve()

        if not path.is_dir():
            raise FileNotFoundError(
                f"NLLB CTranslate2 model directory not found: {path}. "
                "Run download_models.py or update translation.models_dir/nllb_dir."
            )

        missing = [name for name in ("model.bin", "config.json") if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"NLLB CTranslate2 model directory is missing {missing}: {path}"
            )

        return str(path)

    def _resolve_tokenizer_dir(self) -> str:
        path = Path(self._tokenizer_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"NLLB tokenizer directory not found: {path}. "
                "Run download_models.py or update translation.tokenizer_dir."
            )

        missing = [
            name
            for name in ("tokenizer_config.json", "sentencepiece.bpe.model")
            if not (path / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"NLLB tokenizer directory is missing {missing}: {path}"
            )

        return str(path)

    def _intra_threads_arg(self) -> Optional[int]:
        return self._intra_threads if self._intra_threads > 0 else 0

    def _make_translator(self, model_dir: str):
        import ctranslate2

        return ctranslate2.Translator(
            model_dir,
            device=self._device,
            inter_threads=self._inter_threads,
            intra_threads=self._intra_threads_arg(),
        )

    def _load_nllb(self, model_dir: str):
        if model_dir not in self._translators:
            from transformers import AutoTokenizer

            translator = self._make_translator(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path,
                local_files_only=True,
            )
            self._translators[model_dir] = translator
            self._tokenizers[model_dir] = tokenizer
        return self._translators[model_dir], self._tokenizers[model_dir]

    def _nllb_translate(self, text: str, model_dir: str, src: str, tgt: str) -> str:
        try:
            src_code = NLLB_LANG_MAP.get(src, src)
            tgt_code = NLLB_LANG_MAP.get(tgt, tgt)
            translator, tokenizer = self._load_nllb(model_dir)

            tokenizer.src_lang = src_code
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            results = translator.translate_batch(
                [tokens],
                target_prefix=[[tgt_code]],
            )

            target_tokens = results[0].hypotheses[0][1:]
            output_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            return tokenizer.decode(output_ids, skip_special_tokens=True)
        except Exception as exc:
            logger.warning("NLLB translation %s -> %s failed: %s", src, tgt, exc)
            return text
