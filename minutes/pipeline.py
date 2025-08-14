"""Main pipeline orchestration combining all components."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .aggregation import MinuteAggregator, MinuteAggregatorConfig
from .audio import AudioConfig, AudioRecorder
from .summarization import OllamaConfig, Summarizer
from .transcription import Transcriber, WhisperConfig
from .writer import NotesWriter, NotesWriterConfig


@dataclass
class PipelineConfig:
    audio: AudioConfig
    whisper: WhisperConfig
    ollama: OllamaConfig
    aggregator: MinuteAggregatorConfig
    writer: NotesWriterConfig
    segment_secs: float = 20.0
    overlap_secs: float = 5.0


class Pipeline:
    """End-to-end pipeline: audio ➜ whisper.cpp ➜ ollama ➜ notes.

    Process flow:
    1. AudioRecorder captures desktop audio in background thread
    2. SegProducer thread consumes audio segments and queues them
    3. Transcriber thread processes queued audio through whisper.cpp
    4. Aggregator thread monitors transcript accumulation and triggers summaries
    5. NotesWriter persists summaries and updates index

    Threading model:
    - Main thread: CLI coordination and signal handling
    - AudioRecorder: Background audio capture
    - SegProducer: Converts audio stream to discrete segments
    - Transcriber: Processes audio segments via subprocess
    - Aggregator: Monitors summary triggers and writes output
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.recorder = AudioRecorder(cfg.audio, cfg.segment_secs, cfg.overlap_secs)
        self.transcriber = Transcriber(cfg.whisper)
        self.summarizer = Summarizer(cfg.ollama)
        self.aggregator = MinuteAggregator(cfg.aggregator, self.summarizer)
        self.writer = NotesWriter(cfg.writer)

        self._q_audio: queue.Queue[Path] = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        logging.info("Starting pipeline…")
        self.recorder.start()
        self._threads = [
            threading.Thread(
                target=self._produce_segments, name="SegProducer", daemon=True
            ),
            threading.Thread(
                target=self._consume_transcribe, name="Transcriber", daemon=True
            ),
            threading.Thread(
                target=self._tick_aggregator, name="Aggregator", daemon=True
            ),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        logging.info("Stopping pipeline…")
        self._stop.set()
        self.recorder.stop()
        # Drain
        for t in self._threads:
            t.join(timeout=2)

    # Thread funcs
    def _produce_segments(self) -> None:
        for wav in self.recorder.segment_generator():
            if self._stop.is_set():
                break
            try:
                self._q_audio.put(wav, timeout=1)
            except queue.Full:
                logging.warning("Audio queue full, dropping segment %s", wav.name)
                wav.unlink(missing_ok=True)

    def _consume_transcribe(self) -> None:
        while not self._stop.is_set():
            try:
                wav = self._q_audio.get(timeout=0.5)
            except queue.Empty:
                continue
            text = self.transcriber.transcribe_wav(wav)
            if text.strip():
                logging.debug("Transcript chunk: %.60s…", text.replace("\n", " "))
                self.aggregator.add(text)

    def _tick_aggregator(self) -> None:
        while not self._stop.is_set():
            emitted = self.aggregator.maybe_emit()
            if emitted:
                summary_md, short = emitted
                path = self.writer.write_minute(summary_md)
                self.writer.append_summary(path, short)
            time.sleep(2.0)
