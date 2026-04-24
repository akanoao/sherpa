"""
Modular WebRTC Transport Engine.

Handles end-to-end peer-to-peer audio and video streams using aiortc.
Separates media logic from the signaling transport layer (WebSocket).

Requirements:
    pip install aiortc
"""

import asyncio
import logging
from typing import Awaitable, Callable, Optional

try:
    from aiortc import (
        RTCIceCandidate,
        RTCIceServer,
        RTCConfiguration,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.mediastreams import MediaStreamTrack
except ImportError as exc:
    raise ImportError("aiortc is required for WebRTC. Run: pip install aiortc") from exc

from ..core.packet import SignalingEvent

logger = logging.getLogger(__name__)

# Type alias for injecting your websocket send method
SignalingCallback = Callable[[SignalingEvent], Awaitable[None]]
TrackCallback = Callable[[MediaStreamTrack], None]


class WebRTCEngine:
    """
    Modular engine for processing independent WebRTC audio and video tracks.
    """

    def __init__(
        self,
        speaker_id: str,
        session_id: str,
        send_signaling_cb: SignalingCallback,
        on_audio_track_cb: TrackCallback,
        on_video_track_cb: TrackCallback,
        stun_server: str = "stun:stun.l.google.com:19302",
    ) -> None:
        self._speaker_id = speaker_id
        self._session_id = session_id
        self._send_signaling = send_signaling_cb
        self._on_audio_track = on_audio_track_cb
        self._on_video_track = on_video_track_cb

        # Configure ICE Servers (STUN for NAT traversal)
        config = RTCConfiguration([RTCIceServer(urls=[stun_server])])
        self.pc = RTCPeerConnection(configuration=config)

        self._setup_events()

    def _setup_events(self) -> None:
        """Wire up the internal WebRTC event listeners."""
        
        @self.pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            logger.info("📡 Receiving remote %s track", track.kind)
            # Route tracks modularly to their respective sinks
            if track.kind == "audio":
                self._on_audio_track(track)
            elif track.kind == "video":
                self._on_video_track(track)

    def add_audio_source(self, track: MediaStreamTrack) -> None:
        """Attach a local audio generator track."""
        logger.info("🎙️ Adding local audio track to PeerConnection")
        self.pc.addTrack(track)

    def add_video_source(self, track: MediaStreamTrack) -> None:
        """Attach a local video generator track."""
        logger.info("📹 Adding local video track to PeerConnection")
        self.pc.addTrack(track)

    async def initiate_call(self) -> None:
        """Generate an SDP Offer to initiate the P2P connection."""
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        
        event = SignalingEvent(
            speaker_id=self._speaker_id,
            session_id=self._session_id,
            payload_type="offer",
            sdp=self.pc.localDescription.sdp,
        )
        await self._send_signaling(event)

    async def handle_signaling_event(self, event: SignalingEvent) -> None:
        """Process incoming WebRTC signaling from the WebSocket."""
        if event.payload_type == "offer":
            offer = RTCSessionDescription(sdp=event.sdp, type="offer")
            await self.pc.setRemoteDescription(offer)
            
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            
            response = SignalingEvent(
                speaker_id=self._speaker_id,
                session_id=self._session_id,
                payload_type="answer",
                sdp=self.pc.localDescription.sdp,
            )
            await self._send_signaling(response)
            
        elif event.payload_type == "answer":
            answer = RTCSessionDescription(sdp=event.sdp, type="answer")
            await self.pc.setRemoteDescription(answer)

    async def close(self) -> None:
        """Tear down the WebRTC connection."""
        await self.pc.close()