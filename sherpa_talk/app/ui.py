"""
Thread-safe terminal UI for SherpaConnect.

Displays three types of content:
  - Your own live (partial) transcript while you speak.
  - Your own finalized utterances.
  - Remote peer utterances (original language + translated language).

All print operations are serialised with a lock so that concurrent STT /
translation / network threads don't interleave their output.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConversationEntry:
    """A single conversation turn stored in the scrollback history."""

    speaker_id: str
    original_text: str
    translated_text: Optional[str]
    source_lang: str
    target_lang: Optional[str]
    timestamp: float
    is_mine: bool


class TerminalUI:
    """Minimal, thread-safe terminal UI."""

    def __init__(
        self,
        my_speaker_id: str,
        show_original: bool = True,
    ) -> None:
        self._my_id = my_speaker_id
        self._show_original = show_original
        self._lock = threading.Lock()
        self._history: list[ConversationEntry] = []

    # ------------------------------------------------------------------
    # My own speech
    # ------------------------------------------------------------------

    def show_my_partial(self, text: str) -> None:
        """Overwrite the current terminal line with a live partial transcript."""
        with self._lock:
            # Overwrite the line in place (carriage return, no newline)
            line = f"🎙  [you – live]: {text}"
            sys.stdout.write(f"\r{line:<80}")
            sys.stdout.flush()

    def show_my_final(self, text: str, lang: str) -> None:
        """Print a finalized utterance from the local speaker."""
        with self._lock:
            self._clear_line()
            print(f"✅ [you / {lang}]: {text}")

    # ------------------------------------------------------------------
    # Remote peer speech
    # ------------------------------------------------------------------

    def show_remote_original(self, speaker_id: str, text: str, lang: str) -> None:
        """Show the untranslated text received from a remote peer."""
        if not self._show_original:
            return
        with self._lock:
            self._clear_line()
            print(f"📡 [{speaker_id} / {lang}]: {text}")

    def show_remote_translated(
        self, speaker_id: str, text: str, target_lang: str
    ) -> None:
        """Show the translated text that will be fed to TTS."""
        with self._lock:
            self._clear_line()
            print(f"🌐 [{speaker_id} → {target_lang}]: {text}")

    # ------------------------------------------------------------------
    # Status / error
    # ------------------------------------------------------------------

    def show_status(self, message: str) -> None:
        with self._lock:
            self._clear_line()
            print(f"ℹ️  {message}")

    def show_error(self, message: str) -> None:
        with self._lock:
            self._clear_line()
            print(f"❌  {message}", file=sys.stderr)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def record(self, entry: ConversationEntry) -> None:
        with self._lock:
            self._history.append(entry)

    def get_history(self) -> list[ConversationEntry]:
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_line(self) -> None:
        """Clear the partial-transcript overwrite line before printing."""
        sys.stdout.write("\r" + " " * 85 + "\r")
        sys.stdout.flush()
