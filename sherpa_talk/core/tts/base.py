"""
Abstract base class for Text-to-Speech providers.

Any TTS backend (Sherpa-ONNX VITS, Coqui, Piper, …) must implement this
interface so the rest of the application is engine-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TTSProvider(ABC):
    """
    Abstract Text-to-Speech provider.

    Both ``synthesize`` and ``speak`` must be callable from a regular
    (non-async) thread.  Blocking inside ``speak`` (waiting for audio to
    finish playing) is acceptable because the application runs TTS in a
    dedicated worker thread.
    """

    @abstractmethod
    def synthesize(self, text: str, lang: str = "", speed: float = 1.0) -> np.ndarray:
        """
        Convert *text* to audio samples.

        Returns a 1-D ``float32`` numpy array of normalised audio samples
        at the provider's native sample rate.
        """

    @abstractmethod
    def speak(self, text: str, lang: str = "", speed: float = 1.0) -> None:
        """Synthesize *text* and play it through the default audio output."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native audio sample rate of synthesised output."""
