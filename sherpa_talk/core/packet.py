"""
TextEvent – the unit of data sent over the network between peers.

Only text (never raw audio) travels the wire. Each event carries enough
metadata for the receiver to translate and display the message correctly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class TextEvent:
    """A finalized or partial transcript sent from one peer to another."""

    text: str
    source_lang: str
    speaker_id: str
    session_id: str
    sequence_id: int
    timestamp: float
    is_final: bool = True
    confidence: Optional[float] = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "TextEvent":
        d = json.loads(data)
        return cls(**d)

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        *,
        text: str,
        source_lang: str,
        speaker_id: str,
        session_id: str,
        sequence_id: int,
        is_final: bool = True,
        confidence: Optional[float] = None,
    ) -> "TextEvent":
        return cls(
            text=text,
            source_lang=source_lang,
            speaker_id=speaker_id,
            session_id=session_id,
            sequence_id=sequence_id,
            timestamp=time.time(),
            is_final=is_final,
            confidence=confidence,
        )
