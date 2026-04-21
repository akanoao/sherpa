"""
CTranslate2 translation provider (production-grade, fast CPU/GPU inference).

Uses locally converted MarianMT / OPUS-MT models (Helsinki-NLP) for direct
language pairs and NLLB-200-distilled-600M as a broad multilingual fallback.

Model conversion (one-time, offline)
-------------------------------------
Install the conversion tool::

    pip install ctranslate2 transformers sentencepiece

Convert a MarianMT model::

    ct2-opus-mt-converter --model Helsinki-NLP/opus-mt-en-hi \\
        --output_dir ./models/opus-mt-en-hi --quantization int8

Convert the NLLB model::

    ct2-transformers-converter --model facebook/nllb-200-distilled-600M \\
        --output_dir ./models/nllb-200-distilled-600M --quantization int8 --force

Config dict keys
----------------
models_dir    : base directory that contains converted model sub-directories
device        : "cpu" | "cuda"  (default "cpu")
inter_threads : int  (default 1)
intra_threads : int  (default 0 = use all available cores)
pivot_lang    : intermediate language for indirect pairs  (default "en")
nllb_dir      : name / path of the NLLB model dir inside models_dir
                (default "nllb-200-distilled-600M")

NLLB language codes
--------------------
NLLB uses FLORES-200 codes, not plain ISO-639-1 codes.  A small built-in
mapping covers the most common languages; extend ``NLLB_LANG_MAP`` as needed.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Mapping from ISO-639-1 short code → FLORES-200 / NLLB tag
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
    """
    CTranslate2 translation provider.

    Strategy:
    1. If a direct MarianMT model (``opus-mt-{src}-{tgt}``) exists in
       *models_dir*, use it.
    2. Else fall back to NLLB-200-distilled-600M (if available).
    3. Else attempt English pivot: src→en, then en→tgt.
    4. Return original text when nothing works.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._models_dir: str = config["models_dir"]
        self._device: str = config.get("device", "cpu")
        self._inter_threads: int = config.get("inter_threads", 1)
        self._intra_threads: int = config.get("intra_threads", 0)
        self._pivot_lang: str = config.get("pivot_lang", "en")
        self._nllb_dir_name: str = config.get("nllb_dir", "nllb-200-distilled-600M")

        # Lazy-loaded caches: model_dir → (Translator, tokenizer)
        self._translators: dict = {}
        self._tokenizers: dict = {}

    # ------------------------------------------------------------------
    # TranslationProvider interface
    # ------------------------------------------------------------------

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip() or source_lang == target_lang:
            return text

        # 1. Direct MarianMT pair
        direct = self._marian_dir(source_lang, target_lang)
        if direct:
            return self._marian_translate(text, direct, source_lang, target_lang)

        # 2. NLLB fallback
        nllb = self._nllb_model_dir()
        if nllb:
            return self._nllb_translate(text, nllb, source_lang, target_lang)

        # 3. English pivot via MarianMT
        if source_lang != "en" and target_lang != "en":
            src_en = self._marian_dir(source_lang, "en")
            en_tgt = self._marian_dir("en", target_lang)
            if src_en and en_tgt:
                intermediate = self._marian_translate(text, src_en, source_lang, "en")
                return self._marian_translate(intermediate, en_tgt, "en", target_lang)

        logger.warning(
            "CTranslate2: no model found for %s→%s; returning original.",
            source_lang,
            target_lang,
        )
        return text

    # ------------------------------------------------------------------
    # Internal helpers – model discovery
    # ------------------------------------------------------------------

    def _marian_dir(self, src: str, tgt: str) -> Optional[str]:
        """Return the path to a MarianMT CT2 model dir if it exists."""
        candidates = [
            f"opus-mt-{src}-{tgt}",
            f"Helsinki-NLP-opus-mt-{src}-{tgt}",
        ]
        for name in candidates:
            path = os.path.join(self._models_dir, name)
            if os.path.isdir(path):
                return path
        return None

    def _nllb_model_dir(self) -> Optional[str]:
        path = os.path.join(self._models_dir, self._nllb_dir_name)
        return path if os.path.isdir(path) else None

    # ------------------------------------------------------------------
    # Internal helpers – translation
    # ------------------------------------------------------------------

    def _load_marian(self, model_dir: str):
        if model_dir not in self._translators:
            import ctranslate2
            from transformers import MarianTokenizer

            translator = ctranslate2.Translator(
                model_dir,
                device=self._device,
                inter_threads=self._inter_threads,
                intra_threads=self._intra_threads if self._intra_threads > 0 else None,
            )
            tokenizer = MarianTokenizer.from_pretrained(model_dir)
            self._translators[model_dir] = translator
            self._tokenizers[model_dir] = tokenizer
        return self._translators[model_dir], self._tokenizers[model_dir]

    def _load_nllb(self, model_dir: str):
        if model_dir not in self._translators:
            import ctranslate2
            from transformers import NllbTokenizer

            translator = ctranslate2.Translator(
                model_dir,
                device=self._device,
                inter_threads=self._inter_threads,
                intra_threads=self._intra_threads if self._intra_threads > 0 else None,
            )
            tokenizer = NllbTokenizer.from_pretrained(model_dir)
            self._translators[model_dir] = translator
            self._tokenizers[model_dir] = tokenizer
        return self._translators[model_dir], self._tokenizers[model_dir]

    def _marian_translate(
        self, text: str, model_dir: str, src: str, tgt: str
    ) -> str:
        try:
            translator, tokenizer = self._load_marian(model_dir)
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            results = translator.translate_batch([tokens])
            output_ids = tokenizer.convert_tokens_to_ids(results[0].hypotheses[0])
            return tokenizer.decode(output_ids, skip_special_tokens=True)
        except Exception as exc:
            logger.warning("CTranslate2 MarianMT %s→%s failed: %s", src, tgt, exc)
            return text

    def _nllb_translate(
        self, text: str, model_dir: str, src: str, tgt: str
    ) -> str:
        try:
            src_code = NLLB_LANG_MAP.get(src, src)
            tgt_code = NLLB_LANG_MAP.get(tgt, tgt)
            translator, tokenizer = self._load_nllb(model_dir)
            tokenizer.src_lang = src_code
            tokens = tokenizer.convert_ids_to_tokens(
                tokenizer.encode(text, add_special_tokens=True)
            )
            tgt_prefix = [tokenizer.lang_code_to_id[tgt_code]]
            results = translator.translate_batch(
                [tokens],
                target_prefix=[tgt_prefix],
            )
            output_ids = tokenizer.convert_tokens_to_ids(results[0].hypotheses[0])
            return tokenizer.decode(output_ids, skip_special_tokens=True)
        except Exception as exc:
            logger.warning("CTranslate2 NLLB %s→%s failed: %s", src, tgt, exc)
            return text
