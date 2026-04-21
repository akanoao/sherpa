"""
Sherpa-ONNX streaming Speech-to-Text provider.

Supports two model architectures:
  - ``transducer``  (Zipformer, LSTM, …)  – requires encoder/decoder/joiner
  - ``paraformer``  (bilingual zh/en)      – requires encoder/decoder

Config dict keys
----------------
model_type        : "transducer" | "paraformer"   (default "transducer")
tokens            : path to tokens.txt
encoder           : path to encoder .onnx
decoder           : path to decoder .onnx
joiner            : path to joiner  .onnx  (transducer only)
language          : BCP-47 language tag, e.g. "en", "zh"
num_threads       : int  (default 1)
provider          : "cpu" | "cuda" | "coreml"  (default "cpu")
decoding_method   : "greedy_search" | "modified_beam_search"  (default "greedy_search")
input_sample_rate : microphone capture rate in Hz  (default 48000)
rule1_min_trailing_silence : float  (default 2.4)
rule2_min_trailing_silence : float  (default 1.2)
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class SherpaOnnxSTTProvider:
    """Sherpa-ONNX online (streaming) STT adapter."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._language: str = config.get("language", "en")
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback = None
        self._recognizer = self._build_recognizer()

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

    def _build_recognizer(self):
        import sherpa_onnx

        cfg = self._config
        model_type = cfg.get("model_type", "transducer")

        common = dict(
            tokens=cfg["tokens"],
            num_threads=cfg.get("num_threads", 1),
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=cfg.get("rule1_min_trailing_silence", 2.4),
            rule2_min_trailing_silence=cfg.get("rule2_min_trailing_silence", 1.2),
            rule3_min_utterance_length=300,
            provider=cfg.get("provider", "cpu"),
        )

        if model_type == "transducer":
            common.update(
                dict(
                    encoder=cfg["encoder"],
                    decoder=cfg["decoder"],
                    joiner=cfg["joiner"],
                    decoding_method=cfg.get("decoding_method", "greedy_search"),
                )
            )
            return sherpa_onnx.OnlineRecognizer.from_transducer(**common)

        elif model_type == "paraformer":
            common.update(
                dict(
                    encoder=cfg["encoder"],
                    decoder=cfg["decoder"],
                )
            )
            return sherpa_onnx.OnlineRecognizer.from_paraformer(**common)

        else:
            raise ValueError(f"Unknown sherpa_onnx model_type: {model_type!r}")

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            logger.error("sounddevice not installed: %s", exc)
            return

        from .base import TranscriptEvent, TranscriptType

        input_sr = self._config.get("input_sample_rate", 48000)
        chunk = int(0.1 * input_sr)
        stream = self._recognizer.create_stream()
        seq_id = 0

        try:
            with sd.InputStream(channels=1, dtype="float32", samplerate=input_sr) as s:
                while self._running:
                    samples, _ = s.read(chunk)
                    samples = samples.reshape(-1)
                    stream.accept_waveform(input_sr, samples)

                    while self._recognizer.is_ready(stream):
                        self._recognizer.decode_stream(stream)

                    result = self._recognizer.get_result(stream)
                    is_endpoint = self._recognizer.is_endpoint(stream)

                    if result and self._callback:
                        evt_type = (
                            TranscriptType.FINAL
                            if is_endpoint
                            else TranscriptType.PARTIAL
                        )
                        self._callback(
                            TranscriptEvent(
                                text=result,
                                type=evt_type,
                                language=self._language,
                                sequence_id=seq_id,
                            )
                        )

                    if is_endpoint:
                        if result:
                            seq_id += 1
                        self._recognizer.reset(stream)

        except Exception:
            logger.exception("SherpaOnnxSTTProvider error")
