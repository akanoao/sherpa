"""WebRTC transport engine for optional peer-to-peer audio/video media."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

try:
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamTrack
    from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
except ImportError as exc:
    raise ImportError("aiortc is required for WebRTC. Run: pip install aiortc") from exc

from ..core.packet import SignalingEvent

logger = logging.getLogger(__name__)

SignalingCallback = Callable[[SignalingEvent], Awaitable[None]]
TrackCallback = Callable[[MediaStreamTrack], None]


class WebRTCEngine:
    """Owns one RTCPeerConnection and sends signaling over the injected callback."""

    def __init__(
        self,
        speaker_id: str,
        session_id: str,
        send_signaling_cb: SignalingCallback,
        on_audio_track_cb: TrackCallback,
        on_video_track_cb: TrackCallback,
        ice_servers: Optional[list[dict | str]] = None,
    ) -> None:
        self._speaker_id = speaker_id
        self._session_id = session_id
        self._send_signaling = send_signaling_cb
        self._on_audio_track = on_audio_track_cb
        self._on_video_track = on_video_track_cb

        servers = self._build_ice_servers(ice_servers)
        self.pc = RTCPeerConnection(configuration=RTCConfiguration(servers))
        self._setup_events()

    def _build_ice_servers(self, ice_servers: Optional[list[dict | str]]) -> list[RTCIceServer]:
        configured = ice_servers or ["stun:stun.l.google.com:19302"]
        servers: list[RTCIceServer] = []
        for item in configured:
            if isinstance(item, str):
                servers.append(RTCIceServer(urls=[item]))
            else:
                servers.append(
                    RTCIceServer(
                        urls=item["urls"],
                        username=item.get("username"),
                        credential=item.get("credential"),
                    )
                )
        return servers

    def _setup_events(self) -> None:
        @self.pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            logger.info("Receiving remote %s track", track.kind)
            if track.kind == "audio":
                self._on_audio_track(track)
            elif track.kind == "video":
                self._on_video_track(track)

        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate) -> None:
            if candidate is None:
                return
            payload = {
                "candidate": candidate_to_sdp(candidate),
                "sdpMid": candidate.sdpMid,
                "sdpMLineIndex": candidate.sdpMLineIndex,
            }
            await self._send_signaling(
                SignalingEvent(
                    speaker_id=self._speaker_id,
                    session_id=self._session_id,
                    payload_type="ice",
                    candidate=payload,
                )
            )

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info("WebRTC connection state: %s", self.pc.connectionState)

        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange() -> None:
            logger.info("WebRTC ICE state: %s", self.pc.iceConnectionState)

    def add_audio_source(self, track: MediaStreamTrack) -> None:
        logger.info("Adding local audio track to PeerConnection")
        self.pc.addTrack(track)

    def add_video_source(self, track: MediaStreamTrack) -> None:
        logger.info("Adding local video track to PeerConnection")
        self.pc.addTrack(track)

    async def initiate_call(self) -> None:
        """Generate and send an SDP offer."""
        if self.pc.signalingState != "stable":
            logger.info("Skipping offer; signaling state is %s", self.pc.signalingState)
            return

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await self._send_signaling(
            SignalingEvent(
                speaker_id=self._speaker_id,
                session_id=self._session_id,
                payload_type="offer",
                sdp=self.pc.localDescription.sdp,
            )
        )

    async def handle_signaling_event(self, event: SignalingEvent) -> None:
        """Process incoming offer, answer, and ICE signaling packets."""
        if event.speaker_id == self._speaker_id:
            return

        if event.payload_type == "offer":
            if self.pc.signalingState != "stable":
                logger.info(
                    "Ignoring offer from %s while signaling state is %s",
                    event.speaker_id,
                    self.pc.signalingState,
                )
                return

            offer = RTCSessionDescription(sdp=event.sdp, type="offer")
            await self.pc.setRemoteDescription(offer)

            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            await self._send_signaling(
                SignalingEvent(
                    speaker_id=self._speaker_id,
                    session_id=self._session_id,
                    payload_type="answer",
                    sdp=self.pc.localDescription.sdp,
                )
            )
            return

        if event.payload_type == "answer":
            if self.pc.signalingState == "have-local-offer":
                answer = RTCSessionDescription(sdp=event.sdp, type="answer")
                await self.pc.setRemoteDescription(answer)
            else:
                logger.info(
                    "Ignoring answer from %s while signaling state is %s",
                    event.speaker_id,
                    self.pc.signalingState,
                )
            return

        if event.payload_type == "ice" and event.candidate:
            candidate = candidate_from_sdp(event.candidate["candidate"])
            candidate.sdpMid = event.candidate.get("sdpMid")
            candidate.sdpMLineIndex = event.candidate.get("sdpMLineIndex")
            await self.pc.addIceCandidate(candidate)

    async def close(self) -> None:
        await self.pc.close()