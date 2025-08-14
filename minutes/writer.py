"""File writing functionality for persisting meeting notes and maintaining index."""

from __future__ import annotations

import datetime as dt
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from .utils import ensure_dir


@dataclass
class NotesWriterConfig:
    """Configuration for note writing."""

    output_dir: Path


class NotesWriter:
    """Persist per-minute notes and maintain SUMMARY.md.

    Process flow:
    1. Receive summarized content from aggregator
    2. Generate timestamped filename
    3. Write markdown file with frontmatter header
    4. Update SUMMARY.md index with link and description
    5. Use file locking for thread-safe writes
    """

    def __init__(self, cfg: NotesWriterConfig) -> None:
        self.cfg = cfg
        ensure_dir(self.cfg.output_dir)
        self.summary_file = self.cfg.output_dir / "SUMMARY.md"
        if not self.summary_file.exists():
            self.summary_file.write_text("# Minutes Summary\n\n", encoding="utf-8")
        self._lock = threading.Lock()

    def write_minute(self, content_md: str) -> Path:
        ts = dt.datetime.now().astimezone()
        stamp = ts.strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"minute_{stamp}.md"
        path = self.cfg.output_dir / fname
        header = f"---\ncreated: {ts.isoformat()}\n---\n\n"
        path.write_text(header + content_md.strip() + "\n", encoding="utf-8")
        logging.info("Wrote note: %s", path)
        return path

    def append_summary(self, note_path: Path, short_desc: str) -> None:
        rel = note_path.name
        line = f"- [{rel}]({rel}) — {short_desc}\n"
        with self._lock:
            with self.summary_file.open("a", encoding="utf-8") as f:
                f.write(line)
        logging.info("Updated SUMMARY.md")
