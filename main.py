#!/usr/bin/env python3
"""
SherpaConnect - Real-time multilingual voice communication.

Every side does local STT -> sends text -> peer translates locally -> peer does
local TTS. No raw audio ever leaves the device.

Sub-commands
------------
serve
    Start the WebSocket relay server that peers connect to.

talk
    Join a session: capture microphone audio, transcribe it, send it to the
    peer; receive the peer's text, translate it, display it, and optionally
    play it through the local TTS engine.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the WebSocket relay server."""
    from sherpa_talk.transport.ws_server import RelayServer

    server = RelayServer(host=args.host, port=args.port)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nRelay server stopped.")


def cmd_talk(args: argparse.Namespace) -> None:
    """Join a voice session."""
    from sherpa_talk.app.client import SherpaClient
    from sherpa_talk.config import load_config
    from sherpa_talk.core.model_manager import ModelManager

    config = load_config(args.config)
    manager = ModelManager(config)

    speaker_id = args.speaker_id or str(uuid.uuid4())[:8]
    session_id = args.session or str(uuid.uuid4())[:8]

    client = SherpaClient(
        model_manager=manager,
        server_uri=args.server,
        speaker_id=speaker_id,
        session_id=session_id,
        input_lang=args.input_lang,
        output_lang=args.output_lang,
        tts_enabled=not args.no_tts,
        show_original=not args.no_original,
        tts_speed=args.tts_speed,
        initiate_call=args.call,
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nSession ended.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="SherpaConnect - real-time multilingual voice communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Start the WebSocket relay server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    p_serve.set_defaults(func=cmd_serve)

    p_talk = sub.add_parser("talk", help="Join a voice session")
    p_talk.add_argument(
        "--input-lang",
        required=True,
        metavar="LANG",
        help="Language you speak (BCP-47, e.g. en, hi, zh, ja)",
    )
    p_talk.add_argument(
        "--output-lang",
        required=True,
        metavar="LANG",
        help="Language you want to hear/read (BCP-47)",
    )
    p_talk.add_argument(
        "--server",
        required=True,
        metavar="URI",
        help="Relay server URI including room path, e.g. ws://192.168.1.10:8765/room1",
    )
    p_talk.add_argument(
        "--config",
        default="config.json",
        metavar="FILE",
        help="Path to config.json (default: config.json)",
    )
    p_talk.add_argument(
        "--speaker-id",
        default=None,
        metavar="ID",
        help="Your display name shown to the remote peer (default: random short ID)",
    )
    p_talk.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="Session ID (default: random UUID)",
    )
    p_talk.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable local TTS playback of translated speech",
    )
    p_talk.add_argument(
        "--no-original",
        action="store_true",
        help="Hide the untranslated original text from the remote peer",
    )
    p_talk.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        metavar="SPEED",
        help="TTS playback speed multiplier (default: 1.0)",
    )
    p_talk.add_argument(
        "--call",
        action="store_true",
        help="Initiate the WebRTC video/audio call upon connecting",
    )
    p_talk.set_defaults(func=cmd_talk)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
