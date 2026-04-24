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
class SignalingEvent:
    """A WebRTC signaling packet for SDP and ICE candidates."""

    speaker_id: str
    session_id: str
    payload_type: str  # "offer", "answer", "ice"
    sdp: Optional[str] = None
    candidate: Optional[dict] = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        # Tag the JSON so the receiver knows it's signaling, not text
        data = asdict(self)
        data["packet_type"] = "signaling"
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: dict) -> "SignalingEvent":
        return cls(
            speaker_id=data.get("speaker_id", ""),
            session_id=data.get("session_id", ""),
            payload_type=data.get("payload_type", ""),
            sdp=data.get("sdp"),
            candidate=data.get("candidate"),
        )


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
        data = asdict(self)
        data["packet_type"] = "text"
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "TextEvent":
        d = json.loads(data)
        d.pop("packet_type", None)
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
