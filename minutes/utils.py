"""Utility functions for audio processing and file operations."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np


def iso_timestamp(ts: float | None = None) -> str:
    """Return an ISO8601 timestamp string for a UNIX epoch.

    Args:
        ts: Seconds since epoch. If None, use current time.

    Returns:
        ISO8601-formatted timestamp with seconds precision (UTC offset local).
    """
    if ts is None:
        ts = time.time()
    return dt.datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def linear_resample_mono(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample a mono signal using linear interpolation.

    This is intentionally minimal to avoid heavy dependencies. Good enough for
    speech transcription.

    Args:
        x: Mono audio; shape (n,).
        src_sr: Source sample rate.
        dst_sr: Target sample rate.

    Returns:
        Resampled mono audio at dst_sr.
    """
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    duration = x.shape[0] / src_sr
    n_dst = int(round(duration * dst_sr))
    if n_dst <= 1:
        return np.zeros((0,), dtype=np.float32)
    src_idx = np.linspace(0, x.shape[0] - 1, num=n_dst)
    idx0 = np.floor(src_idx).astype(int)
    idx1 = np.minimum(idx0 + 1, x.shape[0] - 1)
    frac = src_idx - idx0
    y = (1 - frac) * x[idx0] + frac * x[idx1]
    return y.astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert (n, channels) to mono (n,) by averaging channels if needed."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)
