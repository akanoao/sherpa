"""
SherpaClient – the main application orchestrator.

Outbound pipeline (microphone → network):
    Microphone audio
        → STT provider (local, engine-agnostic)
        → TranscriptEvent (partial / final)
        → TextEvent packet (JSON)
        → WebSocket relay server
        → remote peer

Inbound pipeline (network → speaker):
    WebSocket relay server
        → TextEvent packet
        → Translation provider (local, engine-agnostic)
        → display on terminal
        → TTS provider (local, engine-agnostic)
        → speaker output

Both pipelines run concurrently.  Audio I/O uses dedicated daemon threads;
translation + TTS run in a single TTS-worker thread to preserve ordering;
network I/O runs on the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Optional

from ..core.model_manager import ModelManager
from ..core.packet import TextEvent
from ..core.stt.base import TranscriptEvent, TranscriptType
from ..transport.ws_client import WebSocketClient
from .ui import ConversationEntry, TerminalUI

logger = logging.getLogger(__name__)


class SherpaClient:
    """
    Full-duplex multilingual voice communication client.

    Parameters
    ----------
    model_manager : ModelManager
        Pre-configured manager that knows how to load STT/TTS/translation.
    server_uri : str
        WebSocket URI of the relay server (e.g. ``ws://192.168.1.10:8765/room1``).
    speaker_id : str
        Unique name shown on the remote peer's screen.
    session_id : str
        Shared session identifier (both peers should use the same room in the URI).
    input_lang : str
        BCP-47 language code of what *you* speak (drives STT model selection).
    output_lang : str
        BCP-47 language code of what *you* want to hear/read (drives translation + TTS).
    tts_enabled : bool
        Whether to play translated speech through the local speakers.
    show_original : bool
        Whether to display the original (untranslated) remote text.
    tts_speed : float
        Synthesis speed multiplier (1.0 = normal).
    """

    def __init__(
        self,
        model_manager: ModelManager,
        server_uri: str,
        speaker_id: str,
        session_id: str,
        input_lang: str,
        output_lang: str,
        tts_enabled: bool = True,
        show_original: bool = True,
        tts_speed: float = 1.0,
    ) -> None:
        self._mm = model_manager
        self._speaker_id = speaker_id
        self._session_id = session_id
        self._input_lang = input_lang
        self._output_lang = output_lang
        self._tts_enabled = tts_enabled
        self._tts_speed = tts_speed

        self._seq_id = 0
        self._running = False

        self._ui = TerminalUI(speaker_id, show_original=show_original)
        self._transport = WebSocketClient(server_uri)

        # Ordered TTS queue – keeps audio playback in sequence.
        self._tts_queue: queue.Queue = queue.Queue()

        # Will be set to the running event loop in run().
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start all pipelines and block until the session ends."""
        self._loop = asyncio.get_running_loop()
        self._running = True

        # TTS worker (daemon thread so it dies with the process)
        tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        tts_thread.start()

        # STT – runs in its own daemon thread (sounddevice blocking loop)
        stt = self._mm.get_stt_provider(self._input_lang)
        stt_thread = threading.Thread(
            target=stt.start, args=(self._on_transcript,), daemon=True
        )
        stt_thread.start()

        self._ui.show_status(
            f"Session ready  |  you speak: {self._input_lang}  "
            f"|  you hear: {self._output_lang}  "
            f"|  speaker-id: {self._speaker_id}"
        )
        self._ui.show_status("Press Ctrl+C to end the session.")

        try:
            # connect() blocks until disconnected / reconnect budget exhausted
            await self._transport.connect(self._on_remote_message)
        finally:
            self._running = False
            stt.stop()

    # ------------------------------------------------------------------
    # Outbound: STT callback → network
    # ------------------------------------------------------------------

    def _on_transcript(self, event: TranscriptEvent) -> None:
        """
        Called from the STT daemon thread on every transcript event.

        Partial events update the live display only.
        Final events are sent to the peer.
        """
        if event.type == TranscriptType.PARTIAL:
            self._ui.show_my_partial(event.text)
            return

        # Final transcript
        self._ui.show_my_final(event.text, self._input_lang)

        text_event = TextEvent.make(
            text=event.text,
            source_lang=self._input_lang,
            speaker_id=self._speaker_id,
            session_id=self._session_id,
            sequence_id=self._seq_id,
            is_final=True,
            confidence=event.confidence,
        )
        self._seq_id += 1

        # Schedule the async send on the event loop from this sync thread
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._transport.send(text_event), self._loop
            )

        # Record in history
        self._ui.record(
            ConversationEntry(
                speaker_id=self._speaker_id,
                original_text=event.text,
                translated_text=None,
                source_lang=self._input_lang,
                target_lang=None,
                timestamp=time.time(),
                is_mine=True,
            )
        )

    # ------------------------------------------------------------------
    # Inbound: network → translation → TTS
    # ------------------------------------------------------------------

    def _on_remote_message(self, event: TextEvent) -> None:
        """
        Called from the asyncio event loop (via the websocket recv loop)
        whenever a TextEvent arrives from the remote peer.

        This method must not block for long because it runs inline in the
        recv loop.  Heavy work (translation, TTS) is dispatched to the
        dedicated TTS worker thread.
        """
        # Show the original (untranslated) text immediately
        self._ui.show_remote_original(event.speaker_id, event.text, event.source_lang)

        # Translate
        if event.source_lang != self._output_lang:
            try:
                translator = self._mm.get_translation_provider()
                translated = translator.translate(
                    event.text, event.source_lang, self._output_lang
                )
            except Exception as exc:
                logger.error("Translation error: %s", exc)
                translated = event.text
        else:
            translated = event.text

        self._ui.show_remote_translated(event.speaker_id, translated, self._output_lang)

        # Record in history
        self._ui.record(
            ConversationEntry(
                speaker_id=event.speaker_id,
                original_text=event.text,
                translated_text=translated,
                source_lang=event.source_lang,
                target_lang=self._output_lang,
                timestamp=event.timestamp,
                is_mine=False,
            )
        )

        # Queue for TTS playback
        if self._tts_enabled:
            self._tts_queue.put_nowait((translated, self._output_lang))

    # ------------------------------------------------------------------
    # TTS worker (runs in a dedicated thread)
    # ------------------------------------------------------------------

    def _tts_worker(self) -> None:
        """
        Consume the TTS queue sequentially so that audio clips play one
        after another without overlapping.
        """
        while self._running:
            try:
                text, lang = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not self._mm.has_tts(lang):
                logger.debug("No TTS configured for language %r; skipping.", lang)
                continue

            try:
                tts = self._mm.get_tts_provider(lang)
                tts.speak(text, lang=lang, speed=self._tts_speed)
            except Exception as exc:
                logger.error("TTS playback error: %s", exc)
