"""Tests for audio recording functionality."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from minutes.audio import AudioConfig, AudioRecorder


class TestAudioConfig:
    """Test audio configuration dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AudioConfig()
        assert config.device is None
        assert config.sample_rate == 48_000
        assert config.block_size == 2048
        assert config.dtype == "float32"
        assert config.loopback is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = AudioConfig(
            device="test_device",
            sample_rate=44100,
            block_size=1024,
            dtype="int16",
            loopback=False,
        )
        assert config.device == "test_device"
        assert config.sample_rate == 44100
        assert config.block_size == 1024
        assert config.dtype == "int16"
        assert config.loopback is False


class TestAudioRecorder:
    """Test audio recorder functionality."""

    def test_initialization(self) -> None:
        """Test audio recorder initialization."""
        config = AudioConfig()
        recorder = AudioRecorder(config, segment_secs=10.0, overlap_secs=2.0)
        assert recorder.cfg == config
        assert recorder.segment_secs == 10.0
        assert recorder.overlap_secs == 2.0
        assert len(recorder._buf) == 0
        assert recorder._thread is None

    def test_callback_adds_audio(self) -> None:
        """Test that audio callback adds data to buffer."""
        config = AudioConfig()
        recorder = AudioRecorder(config)

        # Simulate audio data
        test_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        recorder._callback(test_data, 2, None, None)

        assert len(recorder._buf) == 1
        np.testing.assert_array_equal(recorder._buf[0], test_data)

    def test_callback_with_status(self) -> None:
        """Test that callback logs status messages."""
        config = AudioConfig()
        recorder = AudioRecorder(config)

        test_data = np.array([[1.0, 2.0]], dtype=np.float32)

        with patch("minutes.audio.logging.debug") as mock_debug:
            recorder._callback(test_data, 1, None, "test status")
            mock_debug.assert_called_once_with("Audio status: %s", "test status")

    @patch("minutes.audio.sd.InputStream")
    def test_start_creates_stream(self, mock_stream_class: MagicMock) -> None:
        """Test that start creates audio stream with correct parameters."""
        mock_stream = MagicMock()
        mock_stream_class.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream_class.return_value.__exit__ = MagicMock(return_value=None)

        config = AudioConfig(device="test_device", sample_rate=44100)
        recorder = AudioRecorder(config)

        # Start and immediately stop to avoid hanging
        recorder.start()
        time.sleep(0.1)  # Let thread start
        recorder.stop()

        # Verify stream was created with expected parameters
        mock_stream_class.assert_called_once()
        call_args = mock_stream_class.call_args[1]
        assert call_args["samplerate"] == 44100
        assert call_args["device"] == "test_device"
        assert call_args["channels"] == 2
        assert call_args["callback"] == recorder._callback

    def test_stop_cleans_up_thread(self) -> None:
        """Test that stop properly cleans up the thread."""
        config = AudioConfig()
        recorder = AudioRecorder(config)

        with patch("minutes.audio.sd.InputStream"):
            recorder.start()
            assert recorder._thread is not None
            assert recorder._thread.is_alive()

            thread_ref = recorder._thread  # Keep reference to check if it stops
            recorder.stop()
            time.sleep(0.1)  # Allow thread to finish
            assert recorder._thread is None  # Reference should be cleared
            assert not thread_ref.is_alive()  # Original thread should be stopped

    def test_write_temp_wav_creates_file(self) -> None:
        """Test that temp WAV file creation works."""
        # Create test stereo audio data
        stereo_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)

        wav_path = AudioRecorder._write_temp_wav(stereo_data, 48000)

        try:
            assert wav_path.exists()
            assert wav_path.suffix == ".wav"
            assert "minute_taker_" in wav_path.name

            # Verify file is readable
            import soundfile as sf

            data, sr = sf.read(wav_path.as_posix())
            assert sr == 16000  # Should be resampled to target SR
            assert data.ndim == 1  # Should be mono

        finally:
            # Clean up
            wav_path.unlink(missing_ok=True)

    def test_write_temp_wav_normalizes_audio(self) -> None:
        """Test that temp WAV creation normalizes loud audio."""
        # Create loud stereo audio data
        loud_data = np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32)

        wav_path = AudioRecorder._write_temp_wav(loud_data, 16000)

        try:
            import soundfile as sf

            data, _ = sf.read(wav_path.as_posix())

            # Should be normalized to avoid clipping
            assert np.max(np.abs(data)) <= 1.0
            assert np.max(np.abs(data)) > 0.5  # But still reasonably loud

        finally:
            wav_path.unlink(missing_ok=True)

    def test_segment_generator_timing(self) -> None:
        """Test segment generator produces segments at expected intervals."""
        config = AudioConfig(sample_rate=1000)  # Low sample rate for testing
        recorder = AudioRecorder(config, segment_secs=1.0, overlap_secs=0.2)

        # Add some audio data to buffer
        test_data = np.random.rand(2000, 2).astype(np.float32)  # 2 seconds of audio
        recorder._buf.extend(
            [test_data[i : i + 100] for i in range(0, len(test_data), 100)]
        )

        segments = []
        start_time = time.time()

        # Collect segments for a short time
        for segment_path in recorder.segment_generator():
            segments.append(segment_path)
            if len(segments) >= 2 or time.time() - start_time > 3:
                break

        recorder.stop()

        # Clean up temp files
        for seg in segments:
            seg.unlink(missing_ok=True)

        # Should have generated at least one segment
        assert len(segments) >= 1

    def test_empty_buffer_handling(self) -> None:
        """Test that segment generator handles empty buffers gracefully."""
        config = AudioConfig()
        recorder = AudioRecorder(config, segment_secs=0.1, overlap_secs=0.0)

        segments = []
        start_time = time.time()

        # Run briefly with empty buffer
        for segment_path in recorder.segment_generator():
            segments.append(segment_path)
            if time.time() - start_time > 0.5:
                break

        recorder.stop()

        # Should not generate segments with empty buffer
        assert len(segments) == 0

    @pytest.mark.slow
    def test_integration_with_mock_sounddevice(self) -> None:
        """Integration test with mocked sounddevice."""
        config = AudioConfig()
        recorder = AudioRecorder(config, segment_secs=0.5, overlap_secs=0.1)

        # Mock data to simulate audio input
        mock_audio_data = np.random.rand(1000, 2).astype(np.float32)

        with patch("minutes.audio.sd.InputStream") as mock_stream:
            # Configure mock to call callback with test data
            def mock_stream_context(*args, **kwargs):
                # Simulate calling the callback multiple times
                callback = kwargs.get("callback")
                if callback:
                    for i in range(0, len(mock_audio_data), 100):
                        chunk = mock_audio_data[i : i + 100]
                        callback(chunk, len(chunk), None, None)
                return MagicMock()

            mock_stream.return_value.__enter__ = mock_stream_context
            mock_stream.return_value.__exit__ = MagicMock()

            recorder.start()

            # Let it run briefly
            time.sleep(0.1)

            # Should have audio in buffer
            assert len(recorder._buf) > 0

            recorder.stop()
