"""
Abstract base class for Speech-to-Text providers.

Any STT backend (Vosk, Sherpa-ONNX, Whisper, …) must implement this interface
so the rest of the application is engine-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class TranscriptType(Enum):
    PARTIAL = "partial"
    FINAL = "final"


@dataclass
class TranscriptEvent:
    """Emitted by an STTProvider whenever new transcript text is available."""

    text: str
    type: TranscriptType
    language: str
    confidence: Optional[float] = None
    sequence_id: int = 0


# Callback type: called by the STT provider on every transcript event.
TranscriptCallback = Callable[[TranscriptEvent], None]


class STTProvider(ABC):
    """
    Abstract Speech-to-Text provider.

    Implementations must be thread-safe: ``start()`` launches a background
    thread/loop; ``stop()`` requests a graceful shutdown and returns only
    after the background work has ended.
    """

    @abstractmethod
    def start(self, callback: TranscriptCallback) -> None:
        """
        Begin capturing audio from the microphone and call *callback* for
        every partial or final transcript event.

        This method should return quickly; the actual capture runs in a
        daemon thread or background task.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop audio capture and clean up resources."""
