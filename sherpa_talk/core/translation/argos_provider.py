"""
Argos Translate offline translation provider (MVP).

Argos Translate is a fully offline, open-source translation library that
ships pre-built language-pair packages.  It is the easiest option to get
started with because it requires no model conversion steps.

Install
-------
    pip install argostranslate

Then install the language pair(s) you need.  This can be done programmatically
via ``ArgosTranslationProvider.install_pair()`` or through the Argos GUI.

Config dict keys
----------------
packages_dir : (optional) override where Argos stores packages

References
----------
https://github.com/argosopentech/argos-translate
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


class ArgosTranslationProvider:
    """Translation provider backed by Argos Translate."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or {}
        # Eagerly validate the import so errors are visible at startup.
        try:
            import argostranslate.translate  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "argostranslate is not installed. Run: pip install argostranslate"
            ) from exc

    # ------------------------------------------------------------------
    # TranslationProvider interface
    # ------------------------------------------------------------------

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip() or source_lang == target_lang:
            return text

        import argostranslate.translate

        try:
            translated = argostranslate.translate.translate(text, source_lang, target_lang)
            if translated:
                return translated
        except Exception as exc:
            logger.warning(
                "Argos direct translation %s→%s failed: %s", source_lang, target_lang, exc
            )

        # Fallback: pivot via English when a direct model is unavailable.
        if source_lang != "en" and target_lang != "en":
            try:
                intermediate = argostranslate.translate.translate(text, source_lang, "en")
                if intermediate:
                    result = argostranslate.translate.translate(intermediate, "en", target_lang)
                    if result:
                        return result
            except Exception as exc:
                logger.warning(
                    "Argos pivot translation %s→en→%s failed: %s",
                    source_lang,
                    target_lang,
                    exc,
                )

        logger.warning(
            "Argos translation %s→%s unavailable; returning original text.",
            source_lang,
            target_lang,
        )
        return text

    def is_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        import argostranslate.translate

        installed = argostranslate.translate.get_installed_languages()
        lang_codes = {lang.code for lang in installed}
        return source_lang in lang_codes and target_lang in lang_codes

    # ------------------------------------------------------------------
    # Package management helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_installed_pairs() -> list[tuple[str, str]]:
        """Return a list of (source_lang, target_lang) pairs that are installed."""
        import argostranslate.translate

        pairs: list[tuple[str, str]] = []
        for lang in argostranslate.translate.get_installed_languages():
            for translation in lang.translations_to:
                pairs.append((lang.code, translation.to_lang.code))
        return pairs

    @staticmethod
    def install_pair(source_lang: str, target_lang: str) -> bool:
        """
        Download and install the Argos package for the given language pair.

        Returns ``True`` on success, ``False`` if the pair is not available
        in the Argos package index.
        """
        import argostranslate.package

        logger.info("Updating Argos package index …")
        argostranslate.package.update_package_index()

        available = argostranslate.package.get_available_packages()
        pkg = next(
            (
                p
                for p in available
                if p.from_code == source_lang and p.to_code == target_lang
            ),
            None,
        )
        if pkg is None:
            logger.error("No Argos package found for %s → %s", source_lang, target_lang)
            return False

        logger.info("Downloading %s → %s package …", source_lang, target_lang)
        argostranslate.package.install_from_path(pkg.download())
        logger.info("Installed %s → %s", source_lang, target_lang)
        return True
