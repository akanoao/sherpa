"""
WebSocket client transport.

Connects to a RelayServer, sends outbound TextEvents, and delivers inbound
TextEvents to a callback.  Automatically reconnects on connection loss.

The client uses an asyncio Queue as the outbound buffer so that the
synchronous STT thread can enqueue messages without blocking.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from .base import MessageCallback, SignalingCallback, TransportBase
from ..core.packet import TextEvent, SignalingEvent

logger = logging.getLogger(__name__)

_RECONNECT_DELAY = 2.0       # seconds between reconnection attempts
_MAX_RECONNECTS  = 50        # give up after this many consecutive failures


class WebSocketClient(TransportBase):
    """
    WebSocket client with auto-reconnect.

    ``connect()`` drives the entire receive/send loop and only returns when
    ``disconnect()`` is called or the maximum reconnect count is reached.
    """

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._on_message: Optional[MessageCallback] = None
        self._on_signaling: Optional[SignalingCallback] = None
        self._send_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._ws = None

    # ------------------------------------------------------------------
    # TransportBase interface
    # ------------------------------------------------------------------

    async def send(self, event) -> None:
        """Enqueue *event* for delivery.  Thread-safe via asyncio Queue."""
        await self._send_queue.put(event.to_json())

    def send_nowait(self, event) -> None:
        """
        Enqueue *event* without awaiting.

        Safe to call from a synchronous thread running in the same event
        loop (use ``asyncio.run_coroutine_threadsafe`` from other threads).
        """
        self._send_queue.put_nowait(event.to_json())

    async def connect(self, on_message: MessageCallback, on_signaling: Optional[SignalingCallback] = None) -> None:
        """Connect and process messages until ``disconnect()`` is called."""
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "websockets is not installed. Run: pip install websockets"
            ) from exc

        self._on_message = on_message
        self._on_signaling = on_signaling
        self._running = True
        attempts = 0

        while self._running and attempts < _MAX_RECONNECTS:
            try:
                async with websockets.connect(self._uri) as ws:
                    self._ws = ws
                    attempts = 0  # reset on successful connection
                    logger.info("Connected to %s", self._uri)
                    await asyncio.gather(
                        self._recv_loop(ws),
                        self._send_loop(ws),
                    )
            except OSError as exc:
                if not self._running:
                    break
                attempts += 1
                logger.warning(
                    "Connection to %s failed (%s). Retry %d/%d in %.1fs …",
                    self._uri,
                    exc,
                    attempts,
                    _MAX_RECONNECTS,
                    _RECONNECT_DELAY,
                )
                await asyncio.sleep(_RECONNECT_DELAY)
            except Exception as exc:
                if not self._running:
                    break
                attempts += 1
                logger.warning(
                    "Transport error (%s). Reconnecting %d/%d …",
                    exc,
                    attempts,
                    _MAX_RECONNECTS,
                )
                await asyncio.sleep(_RECONNECT_DELAY)

        self._ws = None

    async def disconnect(self) -> None:
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    async def _recv_loop(self, ws) -> None:
        async for raw_message in ws:
            try:
                data = json.loads(raw_message)
                if data.get("packet_type") == "signaling":
                    if self._on_signaling:
                        self._on_signaling(SignalingEvent.from_json(data))
                else:
                    event = TextEvent.from_json(raw_message)
                    if self._on_message:
                        self._on_message(event)
            except Exception as exc:
                logger.warning("Failed to parse incoming message: %s", exc)

    async def _send_loop(self, ws) -> None:
        payload: str | None = None
        while self._running:
            try:
                if payload is None:
                    payload = await asyncio.wait_for(self._send_queue.get(), timeout=0.5)
                await ws.send(payload)
                payload = None  # sent successfully; clear the buffer
            except asyncio.TimeoutError:
                continue
            except Exception:
                # Keep payload in the buffer so it will be (re)sent after reconnect
                break
