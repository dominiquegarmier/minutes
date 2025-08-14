"""Tests for transcription functionality using whisper.cpp."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


from minutes.transcription import Transcriber, WhisperConfig


class TestWhisperConfig:
    """Test whisper configuration dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = WhisperConfig(
            whisper_bin="test_bin",
            model_path="test_model.bin"
        )
        assert config.whisper_bin == "test_bin"
        assert config.model_path == "test_model.bin"
        assert config.language == "en"
        assert config.threads >= 1
        assert config.print_progress is False

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = WhisperConfig(
            whisper_bin="./whisper",
            model_path="./model.bin",
            language="es",
            threads=8,
            print_progress=True
        )
        assert config.whisper_bin == "./whisper"
        assert config.model_path == "./model.bin"
        assert config.language == "es"
        assert config.threads == 8
        assert config.print_progress is True


class TestTranscriber:
    """Test transcriber functionality."""

    def test_initialization(self) -> None:
        """Test transcriber initialization."""
        config = WhisperConfig(whisper_bin="test", model_path="test.bin")
        transcriber = Transcriber(config)
        assert transcriber.cfg == config

    @patch('minutes.transcription.subprocess.run')
    def test_successful_transcription(self, mock_run: MagicMock) -> None:
        """Test successful transcription process."""
        config = WhisperConfig(whisper_bin="whisper", model_path="model.bin")
        transcriber = Transcriber(config)
        
        # Mock successful subprocess execution
        mock_run.return_value = MagicMock(returncode=0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)
            
            # Create expected output file
            txt_path = wav_path.with_suffix(".txt")
            txt_path.write_text("This is the transcribed text.")
            
            try:
                result = transcriber.transcribe_wav(wav_path)
                
                # Verify correct command was called
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert args[0] == "whisper"
                assert "-m" in args
                assert "model.bin" in args
                assert "-l" in args
                assert "en" in args
                assert "-f" in args
                assert str(wav_path) in args
                assert "-otxt" in args
                assert "-nt" in args
                assert "-np" in args  # no progress by default
                
                assert result == "This is the transcribed text."
                
                # Files should be cleaned up
                assert not txt_path.exists()
                assert not wav_path.exists()
                
            finally:
                # Clean up in case of test failure
                wav_path.unlink(missing_ok=True)
                txt_path.unlink(missing_ok=True)

    @patch('minutes.transcription.subprocess.run')
    def test_transcription_with_progress(self, mock_run: MagicMock) -> None:
        """Test transcription with progress enabled."""
        config = WhisperConfig(
            whisper_bin="whisper", 
            model_path="model.bin",
            print_progress=True
        )
        transcriber = Transcriber(config)
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)
            txt_path = wav_path.with_suffix(".txt")
            txt_path.write_text("test")
            
            try:
                transcriber.transcribe_wav(wav_path)
                
                # Should not include -np flag when progress is enabled
                args = mock_run.call_args[0][0]
                assert "-np" not in args
                
            finally:
                wav_path.unlink(missing_ok=True)
                txt_path.unlink(missing_ok=True)

    @patch('minutes.transcription.subprocess.run')
    def test_subprocess_error(self, mock_run: MagicMock) -> None:
        """Test handling of subprocess errors."""
        config = WhisperConfig(whisper_bin="whisper", model_path="model.bin")
        transcriber = Transcriber(config)
        
        # Mock subprocess failure
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "whisper", stderr=b"Error message"
        )
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)
            
            try:
                with patch('minutes.transcription.logging.error') as mock_log:
                    result = transcriber.transcribe_wav(wav_path)
                    
                    assert result == ""  # Should return empty string on error
                    mock_log.assert_called_once()
                    assert not wav_path.exists()  # WAV should be cleaned up
                    
            finally:
                wav_path.unlink(missing_ok=True)

    @patch('minutes.transcription.subprocess.run')
    def test_subprocess_timeout(self, mock_run: MagicMock) -> None:
        """Test handling of subprocess timeout."""
        import subprocess
        
        config = WhisperConfig(whisper_bin="whisper", model_path="model.bin")
        transcriber = Transcriber(config)
        
        # Mock subprocess timeout
        mock_run.side_effect = subprocess.TimeoutExpired("whisper", 120)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)
            
            try:
                with patch('minutes.transcription.logging.error') as mock_log:
                    result = transcriber.transcribe_wav(wav_path, timeout=1.0)
                    
                    assert result == ""  # Should return empty string on timeout
                    mock_log.assert_called_once()
                    assert not wav_path.exists()  # WAV should be cleaned up
                    
            finally:
                wav_path.unlink(missing_ok=True)

    @patch('minutes.transcription.subprocess.run')
    def test_missing_output_file(self, mock_run: MagicMock) -> None:
        """Test handling when output file is not created."""
        config = WhisperConfig(whisper_bin="whisper", model_path="model.bin")
        transcriber = Transcriber(config)
        
        # Mock successful subprocess but no output file
        mock_run.return_value = MagicMock(returncode=0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)
            
            try:
                result = transcriber.transcribe_wav(wav_path)
                
                assert result == ""  # Should return empty string
                assert not wav_path.exists()  # WAV should be cleaned up
                
            finally:
                wav_path.unlink(missing_ok=True)

    @patch('minutes.transcription.subprocess.run')
    def test_empty_output_file(self, mock_run: MagicMock) -> None:
        """Test handling of empty transcription output."""
        config = WhisperConfig(whisper_bin="whisper", model_path="model.bin")
        transcriber = Transcriber(config)
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)
            txt_path = wav_path.with_suffix(".txt")
            txt_path.write_text("   \n  \t  \n   ")  # Whitespace only
            
            try:
                result = transcriber.transcribe_wav(wav_path)
                
                assert result == ""  # Should return empty string after strip
                assert not txt_path.exists()  # Should be cleaned up
                assert not wav_path.exists()  # Should be cleaned up
                
            finally:
                wav_path.unlink(missing_ok=True)
                txt_path.unlink(missing_ok=True)

    @patch('minutes.transcription.subprocess.run')
    def test_custom_timeout(self, mock_run: MagicMock) -> None:
        """Test custom timeout parameter."""
        config = WhisperConfig(whisper_bin="whisper", model_path="model.bin")
        transcriber = Transcriber(config)
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)
            txt_path = wav_path.with_suffix(".txt")
            txt_path.write_text("test")
            
            try:
                transcriber.transcribe_wav(wav_path, timeout=300.0)
                
                # Verify timeout was passed to subprocess
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs['timeout'] == 300.0
                
            finally:
                wav_path.unlink(missing_ok=True)
                txt_path.unlink(missing_ok=True)

    def test_command_line_construction(self) -> None:
        """Test that command line arguments are constructed correctly."""
        config = WhisperConfig(
            whisper_bin="./whisper_cpp",
            model_path="/models/base.en.bin",
            language="fr",
            threads=4,
            print_progress=False
        )
        transcriber = Transcriber(config)
        
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
            wav_path = Path(wav_file.name)
            
            with patch('minutes.transcription.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                
                transcriber.transcribe_wav(wav_path)
                
                args = mock_run.call_args[0][0]
                
                # Check all expected arguments are present
                assert args[0] == "./whisper_cpp"
                assert "-m" in args
                assert "/models/base.en.bin" in args
                assert "-l" in args
                assert "fr" in args
                assert "-f" in args
                assert str(wav_path) in args
                assert "-otxt" in args
                assert "-of" in args
                assert str(wav_path.with_suffix("")) in args
                assert "-nt" in args
                assert "-np" in args