"""
Vosk Speech-to-Text provider.

Suitable for languages where Vosk models are preferred (Hindi, Japanese, …).

Config dict keys
----------------
model_dir  : path to the extracted Vosk model directory
language   : BCP-47 language tag, e.g. "hi", "ja"
sample_rate: capture sample rate in Hz  (default 16000 – Vosk native)
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class VoskSTTProvider:
    """Vosk online STT adapter."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._language: str = config.get("language", "en")
        self._sample_rate: int = config.get("sample_rate", 16000)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback = None
        self._audio_queue: queue.Queue = queue.Queue()

        # Load the Vosk model eagerly so failures surface at startup.
        try:
            from vosk import Model  # type: ignore

            self._model = Model(config["model_dir"])
        except ImportError as exc:
            raise ImportError("vosk is not installed. Run: pip install vosk") from exc

    # ------------------------------------------------------------------
    # STTProvider interface
    # ------------------------------------------------------------------

    def start(self, callback) -> None:
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("VoskSTTProvider audio status: %s", status)
        self._audio_queue.put(bytes(indata))

    def _run(self) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer  # type: ignore
        except ImportError as exc:
            logger.error("Missing dependency: %s", exc)
            return

        from .base import TranscriptEvent, TranscriptType

        recognizer = KaldiRecognizer(self._model, self._sample_rate)
        seq_id = 0

        try:
            with sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            ):
                while self._running:
                    try:
                        data = self._audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()
                        if text and self._callback:
                            self._callback(
                                TranscriptEvent(
                                    text=text,
                                    type=TranscriptType.FINAL,
                                    language=self._language,
                                    sequence_id=seq_id,
                                )
                            )
                            seq_id += 1
                    else:
                        partial = json.loads(recognizer.PartialResult())
                        partial_text = partial.get("partial", "").strip()
                        if partial_text and self._callback:
                            self._callback(
                                TranscriptEvent(
                                    text=partial_text,
                                    type=TranscriptType.PARTIAL,
                                    language=self._language,
                                    sequence_id=seq_id,
                                )
                            )

        except Exception:
            logger.exception("VoskSTTProvider error")
