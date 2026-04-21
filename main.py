#!/usr/bin/env python3
"""
SherpaConnect – Real-time multilingual voice communication.

Every side does local STT → sends text → peer translates locally → peer does
local TTS.  No raw audio ever leaves the device.

Sub-commands
------------
serve
    Start the WebSocket relay server that peers connect to.

talk
    Join a session: capture microphone audio, transcribe it, send it to the
    peer; receive the peer's text, translate it, display it, and optionally
    play it through the local TTS engine.

install-lang
    Download and install an Argos Translate language-pair package so that
    offline translation works without internet access at runtime.

Quick start (two machines on the same network)
----------------------------------------------
Machine A (runs the relay + speaks English, wants Hindi):

    python main.py serve --port 8765 &
    python main.py talk \\
        --input-lang en --output-lang hi \\
        --server ws://localhost:8765/myroom \\
        --config config.json --speaker-id alice

Machine B (speaks Hindi, wants English):

    python main.py talk \\
        --input-lang hi --output-lang en \\
        --server ws://<IP-of-A>:8765/myroom \\
        --config config.json --speaker-id bob
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the WebSocket relay server."""
    from sherpa_talk.transport.ws_server import RelayServer

    server = RelayServer(host=args.host, port=args.port)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n🛑 Relay server stopped.")


def cmd_talk(args: argparse.Namespace) -> None:
    """Join a voice session."""
    from sherpa_talk.config import load_config
    from sherpa_talk.core.model_manager import ModelManager
    from sherpa_talk.app.client import SherpaClient

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
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n🛑 Session ended.")


def cmd_install_lang(args: argparse.Namespace) -> None:
    """Install an Argos Translate language-pair package."""
    from sherpa_talk.core.translation.argos_provider import ArgosTranslationProvider

    provider = ArgosTranslationProvider()
    src = args.from_lang
    tgt = args.to_lang
    print(f"Installing Argos language pair: {src} → {tgt} …")
    ok = ArgosTranslationProvider.install_pair(src, tgt)
    if ok:
        print(f"✅ Installed {src} → {tgt}")
    else:
        print(
            f"❌ Language pair {src} → {tgt} was not found in the Argos package index.",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_list_langs(_args: argparse.Namespace) -> None:
    """List installed Argos Translate language pairs."""
    from sherpa_talk.core.translation.argos_provider import ArgosTranslationProvider

    pairs = ArgosTranslationProvider.list_installed_pairs()
    if not pairs:
        print("No Argos language pairs installed.")
        print("Install one with:  python main.py install-lang --from en --to hi")
        return
    print("Installed Argos language pairs:")
    for src, tgt in sorted(pairs):
        print(f"  {src} → {tgt}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="SherpaConnect – real-time multilingual voice communication",
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

    # ---- serve ------------------------------------------------------------
    p_serve = sub.add_parser("serve", help="Start the WebSocket relay server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    p_serve.set_defaults(func=cmd_serve)

    # ---- talk -------------------------------------------------------------
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
    p_talk.set_defaults(func=cmd_talk)

    # ---- install-lang -----------------------------------------------------
    p_install = sub.add_parser(
        "install-lang",
        help="Download and install an Argos Translate language-pair package",
    )
    p_install.add_argument(
        "--from",
        dest="from_lang",
        required=True,
        metavar="LANG",
        help="Source language code (e.g. en)",
    )
    p_install.add_argument(
        "--to",
        dest="to_lang",
        required=True,
        metavar="LANG",
        help="Target language code (e.g. hi)",
    )
    p_install.set_defaults(func=cmd_install_lang)

    # ---- list-langs -------------------------------------------------------
    p_list = sub.add_parser(
        "list-langs",
        help="List installed Argos Translate language pairs",
    )
    p_list.set_defaults(func=cmd_list_langs)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
