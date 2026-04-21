"""
WebSocket relay server.

Acts as a simple broadcast hub: every message received from one client is
forwarded to all other clients in the same *room*.  Rooms are identified by
the URL path component (e.g. ``ws://host:port/myroom``).

This server needs only the ``websockets`` library and is intentionally kept
minimal – its only job is to relay text packets between two (or more) peers.

Usage::

    python main.py serve --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Set

logger = logging.getLogger(__name__)


class RelayServer:
    """
    Minimal WebSocket relay server.

    Each URL path defines an isolated room.  Messages sent by any client
    in a room are forwarded verbatim to every *other* client in the same
    room.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self._host = host
        self._port = port
        # room_name → set of connected websocket objects
        self._rooms: dict[str, Set] = defaultdict(set)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the relay server and run until interrupted."""
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "websockets is not installed. Run: pip install websockets"
            ) from exc

        logger.info("Relay server listening on ws://%s:%d", self._host, self._port)
        print(f"🚀 Relay server running on ws://{self._host}:{self._port}")
        print("   Connect clients with:  ws://<host>:<port>/<room-name>")
        print("   Press Ctrl+C to stop.")

        async with websockets.serve(self._handler, self._host, self._port):
            await asyncio.Future()  # run until cancelled

    # ------------------------------------------------------------------
    # Internal handler
    # ------------------------------------------------------------------

    async def _handler(self, ws) -> None:
        # Extract room name from URL path; default to "default"
        try:
            path = ws.request.path
        except AttributeError:
            # older websockets versions expose the path differently
            path = getattr(ws, "path", "/default")

        room = path.strip("/") or "default"
        self._rooms[room].add(ws)
        n = len(self._rooms[room])
        logger.info("Client joined room %r (%d total)", room, n)
        print(f"📡 Client joined room '{room}' ({n} peer(s) in room)")

        try:
            async for message in ws:
                # Broadcast to all other clients in the same room
                recipients = [c for c in self._rooms[room] if c is not ws]
                if recipients:
                    await asyncio.gather(
                        *(c.send(message) for c in recipients),
                        return_exceptions=True,
                    )
        except Exception:
            logger.debug("Connection error in room %r", room, exc_info=True)
        finally:
            self._rooms[room].discard(ws)
            remaining = len(self._rooms[room])
            logger.info("Client left room %r (%d remaining)", room, remaining)
            print(f"🔌 Client left room '{room}' ({remaining} peer(s) remaining)")
            # Clean up empty rooms
            if not self._rooms[room]:
                del self._rooms[room]
