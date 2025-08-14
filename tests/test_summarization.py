"""Tests for summarization functionality using Ollama API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from minutes.summarization import OllamaConfig, Summarizer


class TestOllamaConfig:
    """Test Ollama configuration dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = OllamaConfig()
        assert config.model == "llama3.1:8b-instruct"
        assert config.host == "http://127.0.0.1:11434"
        assert config.timeout == 60.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = OllamaConfig(
            model="llama2:7b",
            host="http://localhost:8080",
            timeout=120.0
        )
        assert config.model == "llama2:7b"
        assert config.host == "http://localhost:8080"
        assert config.timeout == 120.0


class TestSummarizer:
    """Test summarizer functionality."""

    def test_initialization(self) -> None:
        """Test summarizer initialization."""
        config = OllamaConfig()
        summarizer = Summarizer(config)
        assert summarizer.cfg == config
        assert summarizer._session is not None

    def test_empty_texts(self) -> None:
        """Test handling of empty text list."""
        config = OllamaConfig()
        summarizer = Summarizer(config)
        
        result = summarizer.summarize([])
        assert result == ""

    def test_whitespace_only_texts(self) -> None:
        """Test handling of whitespace-only texts."""
        config = OllamaConfig()
        summarizer = Summarizer(config)
        
        result = summarizer.summarize(["   ", "\t\n", ""])
        assert result == ""

    @patch('minutes.summarization.requests.Session.post')
    def test_successful_summarization(self, mock_post: MagicMock) -> None:
        """Test successful summarization request."""
        config = OllamaConfig()
        summarizer = Summarizer(config)
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message": {
                "content": "- Discussed project timeline\n- Assigned tasks to team members"
            }
        }
        mock_post.return_value = mock_response
        
        texts = ["We talked about the project timeline.", "Tasks were assigned."]
        result = summarizer.summarize(texts)
        
        assert result == "- Discussed project timeline\n- Assigned tasks to team members"
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://127.0.0.1:11434/api/chat"
        
        # Check payload structure
        payload = call_args[1]['json']
        assert payload['model'] == "llama3.1:8b-instruct"
        assert payload['stream'] is False
        assert len(payload['messages']) == 2
        assert payload['messages'][0]['role'] == "system"
        assert payload['messages'][1]['role'] == "user"
        
        # Check that texts were included in user message
        user_message = payload['messages'][1]['content']
        assert "We talked about the project timeline." in user_message
        assert "Tasks were assigned." in user_message

    @patch('minutes.summarization.requests.Session.post')
    def test_api_request_error(self, mock_post: MagicMock) -> None:
        """Test handling of API request errors."""
        config = OllamaConfig()
        summarizer = Summarizer(config)
        
        # Mock API request failure
        mock_post.side_effect = requests.RequestException("Connection failed")
        
        with patch('minutes.summarization.logging.error') as mock_log:
            texts = ["Some meeting discussion."]
            result = summarizer.summarize(texts)
            
            assert result == ""
            mock_log.assert_called_once()

    @patch('minutes.summarization.requests.Session.post')
    def test_api_http_error(self, mock_post: MagicMock) -> None:
        """Test handling of HTTP errors from API."""
        config = OllamaConfig()
        summarizer = Summarizer(config)
        
        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_post.return_value = mock_response
        
        with patch('minutes.summarization.logging.error') as mock_log:
            texts = ["Meeting content."]
            result = summarizer.summarize(texts)
            
            assert result == ""
            mock_log.assert_called_once()

    @patch('minutes.summarization.requests.Session.post')
    def test_malformed_api_response(self, mock_post: MagicMock) -> None:
        """Test handling of malformed API responses."""
        config = OllamaConfig()
        summarizer = Summarizer(config)
        
        # Mock malformed response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"unexpected": "structure"}
        mock_post.return_value = mock_response
        
        texts = ["Meeting content."]
        result = summarizer.summarize(texts)
        
        # Should handle missing keys gracefully
        assert result == ""

    @patch('minutes.summarization.requests.Session.post')
    def test_empty_api_response_content(self, mock_post: MagicMock) -> None:
        """Test handling of empty content from API."""
        config = OllamaConfig()
        summarizer = Summarizer(config)
        
        # Mock response with empty content
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message": {"content": "   \n  \t  "}
        }
        mock_post.return_value = mock_response
        
        texts = ["Meeting content."]
        result = summarizer.summarize(texts)
        
        assert result == ""

    @patch('minutes.summarization.requests.Session.post')
    def test_custom_model_and_host(self, mock_post: MagicMock) -> None:
        """Test summarization with custom model and host."""
        config = OllamaConfig(
            model="custom-model:latest",
            host="http://custom-host:9999"
        )
        summarizer = Summarizer(config)
        
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message": {"content": "Summary content"}
        }
        mock_post.return_value = mock_response
        
        texts = ["Meeting discussion."]
        summarizer.summarize(texts)
        
        # Verify correct URL and model were used
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://custom-host:9999/api/chat"
        payload = call_args[1]['json']
        assert payload['model'] == "custom-model:latest"

    @patch('minutes.summarization.requests.Session.post')
    def test_timeout_parameter(self, mock_post: MagicMock) -> None:
        """Test that timeout parameter is passed to request."""
        config = OllamaConfig(timeout=30.0)
        summarizer = Summarizer(config)
        
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message": {"content": "Summary"}
        }
        mock_post.return_value = mock_response
        
        texts = ["Content."]
        summarizer.summarize(texts)
        
        # Verify timeout was passed
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['timeout'] == 30.0

    def test_text_filtering_and_joining(self) -> None:
        """Test that texts are properly filtered and joined."""
        config = OllamaConfig()
        summarizer = Summarizer(config)
        
        with patch('minutes.summarization.requests.Session.post') as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "message": {"content": "Summary"}
            }
            mock_post.return_value = mock_response
            
            texts = ["Valid text", "", "  ", "Another valid text", "\t\n"]
            summarizer.summarize(texts)
            
            # Check that only non-empty texts were included
            payload = mock_post.call_args[1]['json']
            user_message = payload['messages'][1]['content']
            assert "Valid text" in user_message
            assert "Another valid text" in user_message
            # Empty/whitespace texts should be filtered out

    @patch('minutes.summarization.requests.Session.post')
    def test_system_prompt_content(self, mock_post: MagicMock) -> None:
        """Test that system prompt contains expected instructions."""
        config = OllamaConfig()
        summarizer = Summarizer(config)
        
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message": {"content": "Summary"}
        }
        mock_post.return_value = mock_response
        
        texts = ["Meeting content."]
        summarizer.summarize(texts)
        
        payload = mock_post.call_args[1]['json']
        system_message = payload['messages'][0]['content']
        
        # Check key instructions are present
        assert "meeting minute-taker" in system_message.lower()
        assert "concise" in system_message.lower()
        assert "bullets" in system_message.lower()
        assert "markdown" in system_message.lower()