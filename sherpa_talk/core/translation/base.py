"""
Abstract base class for translation providers.

Any translation backend (Argos Translate, CTranslate2/MarianMT, NLLB, …)
must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TranslationProvider(ABC):
    """
    Abstract offline translation provider.

    All implementations must translate locally – no cloud API calls.
    """

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate *text* from *source_lang* to *target_lang*.

        Language codes follow BCP-47 / ISO 639-1 conventions (e.g. "en",
        "hi", "zh", "ja").  Returns the original *text* unchanged when
        translation is not possible.
        """

    def is_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        """
        Return ``True`` if this provider can translate the given language pair.

        The default implementation always returns ``True``; subclasses should
        override this if they have limited language coverage.
        """
        return True
