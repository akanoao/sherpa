"""
Abstract base class for the transport layer.

A transport implementation must be able to:
  * Send a TextEvent to connected peer(s).
  * Receive TextEvents from peer(s) and deliver them via a callback.
  * Handle reconnections transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..core.packet import TextEvent

# Callback type: called whenever a TextEvent arrives from a peer.
MessageCallback = Callable[[TextEvent], None]


class TransportBase(ABC):
    """Abstract network transport for TextEvent messages."""

    @abstractmethod
    async def connect(self, on_message: MessageCallback) -> None:
        """
        Connect to the relay and start processing messages.

        Blocks (in the asyncio sense) until disconnected.  Implementations
        should auto-reconnect on transient failures.
        """

    @abstractmethod
    async def send(self, event: TextEvent) -> None:
        """Enqueue *event* for delivery to connected peer(s)."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the connection."""
