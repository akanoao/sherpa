"""Main application orchestrator for a SherpaConnect session."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Optional

from ..core.model_manager import ModelManager
from ..core.packet import SignalingEvent, TextEvent
from ..core.stt.base import TranscriptEvent, TranscriptType
from ..transport.webrtc_manager import WebRTCEngine
from ..transport.ws_client import WebSocketClient
from .media import AudioReceiver, CameraVideoStreamTrack, MicrophoneAudioTrack, VideoUI
from .ui import ConversationEntry, TerminalUI

logger = logging.getLogger(__name__)


class SherpaClient:
    """Full-duplex multilingual communication client."""

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
        initiate_call: bool = False,
        translation_enabled: bool = True,
        webrtc_video_enabled: bool = False,
        webrtc_audio_enabled: bool = False,
    ) -> None:
        self._mm = model_manager
        self._speaker_id = speaker_id
        self._session_id = session_id
        self._input_lang = input_lang
        self._output_lang = output_lang
        self._tts_enabled = tts_enabled
        self._tts_speed = tts_speed
        self._initiate_call = initiate_call
        self._translation_enabled = translation_enabled
        self._webrtc_video_enabled = webrtc_video_enabled
        self._webrtc_audio_enabled = webrtc_audio_enabled

        self._seq_id = 0
        self._running = False

        self._ui = TerminalUI(speaker_id, show_original=show_original)
        self._transport = WebSocketClient(server_uri)
        self._webrtc: Optional[WebRTCEngine] = None
        self._video_ui: Optional[VideoUI] = None
        self._audio_receiver: Optional[AudioReceiver] = None

        self._inbound_queue: queue.Queue[TextEvent] = queue.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _webrtc_enabled(self) -> bool:
        return self._webrtc_video_enabled or self._webrtc_audio_enabled or self._initiate_call

    async def run(self) -> None:
        """Start selected pipelines and block until the session ends."""
        self._loop = asyncio.get_running_loop()
        self._running = True

        if self._webrtc_enabled():
            self._webrtc = WebRTCEngine(
                speaker_id=self._speaker_id,
                session_id=self._session_id,
                send_signaling_cb=self._send_signaling_async,
                on_audio_track_cb=self._on_audio_track,
                on_video_track_cb=self._on_video_track,
            )

            if self._webrtc_video_enabled:
                self._video_ui = VideoUI(window_name=f"SherpaConnect - {self._speaker_id}")
                self._webrtc.add_video_source(CameraVideoStreamTrack())

            if self._webrtc_audio_enabled:
                self._audio_receiver = AudioReceiver()
                # When translated TTS is enabled, keep raw remote WebRTC audio muted.
                self._audio_receiver.set_mute(self._tts_enabled)
                self._webrtc.add_audio_source(MicrophoneAudioTrack())

        stt = None
        if self._translation_enabled:
            inbound_thread = threading.Thread(target=self._inbound_worker, daemon=True)
            inbound_thread.start()

            stt = self._mm.get_stt_provider(self._input_lang)
            stt_thread = threading.Thread(
                target=stt.start,
                args=(self._on_transcript,),
                daemon=True,
            )
            stt_thread.start()

        async def _do_call() -> None:
            await asyncio.sleep(2.0)
            if self._webrtc:
                self._ui.show_status("Initiating WebRTC media call...")
                await self._webrtc.initiate_call()

        if self._initiate_call:
            asyncio.create_task(_do_call())

        self._ui.show_status(
            f"Session ready  |  you speak: {self._input_lang}  "
            f"|  you hear: {self._output_lang}  "
            f"|  speaker-id: {self._speaker_id}"
        )
        self._ui.show_status(
            "Pipelines  |  "
            f"translation: {'on' if self._translation_enabled else 'off'}  "
            f"|  webrtc-video: {'on' if self._webrtc_video_enabled else 'off'}  "
            f"|  webrtc-audio: {'on' if self._webrtc_audio_enabled else 'off'}"
        )
        self._ui.show_status("Press Ctrl+C to end the session.")

        try:
            await self._transport.connect(self._on_remote_message, self._on_remote_signaling)
        finally:
            self._running = False
            if stt:
                stt.stop()
            if self._video_ui:
                self._video_ui.stop()
            if self._audio_receiver:
                self._audio_receiver.stop()
            if self._webrtc:
                await self._webrtc.close()

    def _on_transcript(self, event: TranscriptEvent) -> None:
        """Handle local STT events and send final text packets."""
        if not self._translation_enabled:
            return

        if event.type == TranscriptType.PARTIAL:
            self._ui.show_my_partial(event.text)
            return

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

        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(self._transport.send(text_event), self._loop)

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

    def _on_remote_message(self, event: TextEvent) -> None:
        """Queue a remote text packet for translation/TTS."""
        if not self._translation_enabled:
            return

        self._ui.show_remote_original(event.speaker_id, event.text, event.source_lang)
        self._inbound_queue.put_nowait(event)

    async def _send_signaling_async(self, event: SignalingEvent) -> None:
        """Send WebRTC signaling via the WebSocket transport."""
        await self._transport.send(event)

    def _on_remote_signaling(self, event: SignalingEvent) -> None:
        """Handle incoming WebRTC signaling."""
        if self._loop and self._running and self._webrtc:
            asyncio.run_coroutine_threadsafe(
                self._webrtc.handle_signaling_event(event),
                self._loop,
            )

    def _on_audio_track(self, track) -> None:
        logger.info("Receiving remote WebRTC audio track")
        if self._audio_receiver and self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._audio_receiver.consume_audio_track(track),
                self._loop,
            )

    def _on_video_track(self, track) -> None:
        logger.info("Receiving remote WebRTC video track")
        if self._video_ui and self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._video_ui.consume_video_track(track),
                self._loop,
            )

    def _inbound_worker(self) -> None:
        """Translate inbound text sequentially and optionally play TTS."""
        while self._running:
            try:
                event = self._inbound_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if event.source_lang != self._output_lang:
                try:
                    translator = self._mm.get_translation_provider()
                    translated = translator.translate(
                        event.text,
                        event.source_lang,
                        self._output_lang,
                    )
                except Exception as exc:
                    logger.error("Translation error: %s", exc)
                    translated = event.text
            else:
                translated = event.text

            self._ui.show_remote_translated(event.speaker_id, translated, self._output_lang)

            if self._video_ui:
                self._video_ui.update_text(translated)

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

            if self._tts_enabled and translated:
                lang = self._output_lang
                if not self._mm.has_tts(lang):
                    logger.debug("No TTS configured for language %r; skipping.", lang)
                    continue
                try:
                    tts = self._mm.get_tts_provider(lang)
                    tts.speak(translated, lang=lang, speed=self._tts_speed)
                except Exception as exc:
                    logger.error("TTS playback error: %s", exc)
