"""Transcription functionality using whisper.cpp subprocess calls."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WhisperConfig:
    """Configuration for whisper.cpp transcription."""

    model_path: str
    whisper_bin: str = "whisper-cli"
    language: str = "en"
    threads: int = max(1, os.cpu_count() or 1)
    print_progress: bool = False


class Transcriber:
    """Run whisper.cpp as a subprocess for each audio segment.

    Process flow:
    1. Receive WAV file path from audio recorder
    2. Execute whisper.cpp binary with model and audio file
    3. Read generated text output file
    4. Clean up temporary files
    5. Return transcribed text
    """

    def __init__(self, cfg: WhisperConfig) -> None:
        self.cfg = cfg

    def transcribe_wav(self, wav: Path, timeout: float = 120.0) -> str:
        """Transcribe a WAV file using whisper.cpp.

        Args:
            wav: Path to mono 16k WAV file.
            timeout: Max seconds to allow the subprocess.

        Returns:
            Transcribed text.
        """
        out_prefix = wav.with_suffix("")
        args = [
            self.cfg.whisper_bin,
            "-m",
            self.cfg.model_path,
            "-l",
            self.cfg.language,
            "-f",
            wav.as_posix(),
            "-otxt",
            "-of",
            out_prefix.as_posix(),
            "-nt",  # no timestamps; cleaner plain text
        ]
        if not self.cfg.print_progress:
            args.append("-np")

        logging.debug("Running whisper.cpp: %s", " ".join(args))
        try:
            subprocess.run(args, check=True, timeout=timeout, capture_output=True)
        except subprocess.CalledProcessError as e:
            logging.error(
                "whisper.cpp failed: %s\n%s", e, e.stderr.decode(errors="ignore")
            )
            return ""
        except subprocess.TimeoutExpired:
            logging.error("whisper.cpp timed out on %s", wav.name)
            return ""

        txt_file = Path(f"{out_prefix}.txt")
        if txt_file.exists():
            text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            try:
                txt_file.unlink(missing_ok=True)
            finally:
                wav.unlink(missing_ok=True)
            return text
        wav.unlink(missing_ok=True)
        return ""
