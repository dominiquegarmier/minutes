"""Aggregation logic for collecting transcripts and triggering minute summaries."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from .summarization import Summarizer


@dataclass
class MinuteAggregatorConfig:
    """Configuration for minute-sized aggregation windows."""

    window_secs: int = 60
    # soft policy: also emit if we collected this many chars, whichever first
    soft_chars: int = 1200


class MinuteAggregator:
    """Collect transcripts and emit minute summaries based on time/size.

    Process flow:
    1. Accumulate transcript snippets with timestamps
    2. Monitor both time elapsed and character count
    3. Trigger summarization when window time OR character limit reached
    4. Extract relevant window of text for summarization
    5. Generate both full summary and short description
    6. Reset aggregation state for next minute
    """

    def __init__(self, cfg: MinuteAggregatorConfig, summarizer: Summarizer) -> None:
        self.cfg = cfg
        self.summarizer = summarizer
        self._items: list[tuple[float, str]] = []  # (timestamp, text)
        self._last_emit_ts: float = time.time()
        self._lock = threading.Lock()

    def add(self, text: str) -> None:
        if not text.strip():
            return
        with self._lock:
            self._items.append((time.time(), text.strip()))

    def maybe_emit(self) -> Optional[tuple[str, str]]:
        """Maybe produce a summary.

        Returns:
            Tuple of (summary_markdown, short_description) if emitted, else None.
        """
        now = time.time()
        with self._lock:
            texts = [t for (ts, t) in self._items]
            age = now - self._last_emit_ts
            size = sum(len(t) for t in texts)
            if age < self.cfg.window_secs and size < self.cfg.soft_chars:
                return None
            # Slice to last minute of items by timestamp
            cutoff = now - self.cfg.window_secs
            window = [t for (ts, t) in self._items if ts >= cutoff]
            if not window:
                # if empty (e.g., silence), still try with last few items
                window = texts[-5:]
            summary = self.summarizer.summarize(window)
            if not summary.strip():
                return None
            # Reset
            self._last_emit_ts = now
            # Keep items, but drop those too old to matter
            self._items = [(ts, t) for (ts, t) in self._items if ts >= cutoff]

        # Make a short description from the first bullet (stripped markers)
        first_line = next(
            (ln for ln in summary.splitlines() if ln.strip(" -*")), ""
        ).strip()
        # Strip bullet markers from the beginning
        short = first_line.lstrip(" -*").strip()[:120]
        return summary, short
