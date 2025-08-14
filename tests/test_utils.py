"""Tests for utility functions in the utils module."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from minutes.utils import ensure_dir, iso_timestamp, linear_resample_mono, to_mono


class TestIsoTimestamp:
    """Test ISO timestamp generation."""

    def test_current_time(self) -> None:
        """Test ISO timestamp generation with current time."""
        timestamp = iso_timestamp()
        assert len(timestamp) > 20  # Should be lengthy ISO format
        assert "T" in timestamp or " " in timestamp  # Should have date/time separator
        assert "+" in timestamp or "-" in timestamp  # Should have timezone offset

    def test_specific_time(self) -> None:
        """Test ISO timestamp generation with specific timestamp."""
        test_time = 1609459200.0  # 2021-01-01 00:00:00 UTC
        timestamp = iso_timestamp(test_time)
        assert "2021" in timestamp
        assert "01-01" in timestamp

    def test_timestamp_format(self) -> None:
        """Test that timestamp follows expected format."""
        timestamp = iso_timestamp(time.time())
        # Should match YYYY-MM-DD HH:MM:SS+OFFSET format
        parts = timestamp.split()
        assert len(parts) == 2  # date and time parts
        date_part, time_part = parts
        assert len(date_part.split("-")) == 3  # year-month-day
        assert ":" in time_part  # time should have colons


class TestEnsureDir:
    """Test directory creation utility."""

    def test_create_new_directory(self) -> None:
        """Test creating a new directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "new_dir"
            assert not test_path.exists()
            ensure_dir(test_path)
            assert test_path.exists()
            assert test_path.is_dir()

    def test_create_nested_directories(self) -> None:
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "parent" / "child" / "grandchild"
            assert not test_path.exists()
            ensure_dir(test_path)
            assert test_path.exists()
            assert test_path.is_dir()

    def test_existing_directory(self) -> None:
        """Test that existing directories are not affected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir)
            assert test_path.exists()
            # Should not raise an error
            ensure_dir(test_path)
            assert test_path.exists()


class TestLinearResampleMono:
    """Test linear resampling functionality."""

    def test_same_sample_rate(self) -> None:
        """Test resampling with same source and destination sample rates."""
        signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = linear_resample_mono(signal, 44100, 44100)
        np.testing.assert_array_almost_equal(result, signal.astype(np.float32))

    def test_downsample(self) -> None:
        """Test downsampling to lower sample rate."""
        # Create a simple signal
        signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = linear_resample_mono(signal, 4, 2)  # Half the sample rate
        assert len(result) == 2
        assert result.dtype == np.float32

    def test_upsample(self) -> None:
        """Test upsampling to higher sample rate."""
        signal = np.array([1.0, 2.0], dtype=np.float64)
        result = linear_resample_mono(signal, 2, 4)  # Double the sample rate
        assert len(result) == 4
        assert result.dtype == np.float32

    def test_empty_signal(self) -> None:
        """Test resampling empty signal."""
        signal = np.array([], dtype=np.float64)
        result = linear_resample_mono(signal, 44100, 16000)
        assert len(result) == 0
        assert result.dtype == np.float32

    def test_very_short_signal(self) -> None:
        """Test resampling very short signal."""
        signal = np.array([1.0], dtype=np.float64)
        result = linear_resample_mono(signal, 44100, 16000)
        # Should handle edge case gracefully
        assert result.dtype == np.float32

    def test_realistic_audio_resample(self) -> None:
        """Test realistic audio resampling scenario."""
        # Create a sine wave at 48kHz, resample to 16kHz
        duration = 0.1  # 100ms
        src_sr = 48000
        dst_sr = 16000
        t = np.linspace(0, duration, int(src_sr * duration), endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        
        result = linear_resample_mono(signal, src_sr, dst_sr)
        expected_length = int(dst_sr * duration)
        assert len(result) == expected_length
        assert result.dtype == np.float32


class TestToMono:
    """Test stereo to mono conversion."""

    def test_already_mono(self) -> None:
        """Test that mono signals are returned unchanged."""
        mono_signal = np.array([1.0, 2.0, 3.0])
        result = to_mono(mono_signal)
        np.testing.assert_array_equal(result, mono_signal)

    def test_stereo_to_mono(self) -> None:
        """Test stereo to mono conversion."""
        # Create stereo signal [left, right] for each sample
        stereo_signal = np.array([[1.0, 3.0], [2.0, 4.0], [0.0, 2.0]])
        result = to_mono(stereo_signal)
        # Should average left and right channels
        expected = np.array([2.0, 3.0, 1.0])  # (1+3)/2, (2+4)/2, (0+2)/2
        np.testing.assert_array_equal(result, expected)

    def test_multichannel_to_mono(self) -> None:
        """Test multi-channel to mono conversion."""
        # Create 3-channel signal
        multi_signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = to_mono(multi_signal)
        # Should average all channels
        expected = np.array([2.0, 5.0])  # (1+2+3)/3, (4+5+6)/3
        np.testing.assert_array_equal(result, expected)

    def test_empty_stereo(self) -> None:
        """Test empty stereo signal."""
        empty_stereo = np.array([]).reshape(0, 2)
        result = to_mono(empty_stereo)
        assert len(result) == 0

    def test_single_sample_stereo(self) -> None:
        """Test single sample stereo signal."""
        single_stereo = np.array([[1.0, -1.0]])
        result = to_mono(single_stereo)
        assert len(result) == 1
        assert result[0] == 0.0  # (1 + (-1)) / 2