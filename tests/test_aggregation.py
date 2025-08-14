"""Tests for aggregation functionality."""

from __future__ import annotations

import time
from unittest.mock import MagicMock


from minutes.aggregation import MinuteAggregator, MinuteAggregatorConfig
from minutes.summarization import Summarizer


class TestMinuteAggregatorConfig:
    """Test minute aggregator configuration dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MinuteAggregatorConfig()
        assert config.window_secs == 60
        assert config.soft_chars == 1200

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = MinuteAggregatorConfig(window_secs=90, soft_chars=2000)
        assert config.window_secs == 90
        assert config.soft_chars == 2000


class TestMinuteAggregator:
    """Test minute aggregator functionality."""

    def test_initialization(self) -> None:
        """Test aggregator initialization."""
        config = MinuteAggregatorConfig()
        mock_summarizer = MagicMock(spec=Summarizer)
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        assert aggregator.cfg == config
        assert aggregator.summarizer == mock_summarizer
        assert len(aggregator._items) == 0

    def test_add_text(self) -> None:
        """Test adding text to aggregator."""
        config = MinuteAggregatorConfig()
        mock_summarizer = MagicMock(spec=Summarizer)
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        aggregator.add("First text")
        aggregator.add("Second text")
        
        assert len(aggregator._items) == 2
        assert aggregator._items[0][1] == "First text"
        assert aggregator._items[1][1] == "Second text"

    def test_add_empty_text(self) -> None:
        """Test that empty/whitespace text is ignored."""
        config = MinuteAggregatorConfig()
        mock_summarizer = MagicMock(spec=Summarizer)
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        aggregator.add("")
        aggregator.add("   ")
        aggregator.add("\t\n")
        aggregator.add("Valid text")
        
        assert len(aggregator._items) == 1
        assert aggregator._items[0][1] == "Valid text"

    def test_maybe_emit_no_content(self) -> None:
        """Test maybe_emit with no content returns None."""
        config = MinuteAggregatorConfig()
        mock_summarizer = MagicMock(spec=Summarizer)
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        result = aggregator.maybe_emit()
        assert result is None

    def test_maybe_emit_insufficient_time_and_chars(self) -> None:
        """Test maybe_emit with insufficient time and characters returns None."""
        config = MinuteAggregatorConfig(window_secs=60, soft_chars=1000)
        mock_summarizer = MagicMock(spec=Summarizer)
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        # Add some text but not enough to trigger emission
        aggregator.add("Short text")
        
        result = aggregator.maybe_emit()
        assert result is None

    def test_maybe_emit_by_time_trigger(self) -> None:
        """Test maybe_emit triggered by time window."""
        config = MinuteAggregatorConfig(window_secs=1, soft_chars=1000)
        mock_summarizer = MagicMock(spec=Summarizer)
        mock_summarizer.summarize.return_value = "- Summary point 1\n- Summary point 2"
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        # Add text and wait for time trigger
        aggregator.add("Some meeting content")
        time.sleep(1.1)  # Wait longer than window_secs
        
        result = aggregator.maybe_emit()
        
        assert result is not None
        summary, short = result
        assert summary == "- Summary point 1\n- Summary point 2"
        assert "Summary point 1" in short
        mock_summarizer.summarize.assert_called_once()

    def test_maybe_emit_by_character_trigger(self) -> None:
        """Test maybe_emit triggered by character count."""
        config = MinuteAggregatorConfig(window_secs=3600, soft_chars=50)  # Long time, low char limit
        mock_summarizer = MagicMock(spec=Summarizer)
        mock_summarizer.summarize.return_value = "- Long summary content"
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        # Add enough text to trigger character limit
        long_text = "This is a long piece of meeting content that should trigger the character limit for emission."
        aggregator.add(long_text)
        
        result = aggregator.maybe_emit()
        
        assert result is not None
        summary, short = result
        assert summary == "- Long summary content"
        mock_summarizer.summarize.assert_called_once()

    def test_maybe_emit_empty_summary(self) -> None:
        """Test maybe_emit with empty summary from summarizer."""
        config = MinuteAggregatorConfig(window_secs=1, soft_chars=1000)
        mock_summarizer = MagicMock(spec=Summarizer)
        mock_summarizer.summarize.return_value = ""  # Empty summary
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        aggregator.add("Some content")
        time.sleep(1.1)
        
        result = aggregator.maybe_emit()
        assert result is None  # Should return None for empty summary

    def test_maybe_emit_whitespace_summary(self) -> None:
        """Test maybe_emit with whitespace-only summary."""
        config = MinuteAggregatorConfig(window_secs=1, soft_chars=1000)
        mock_summarizer = MagicMock(spec=Summarizer)
        mock_summarizer.summarize.return_value = "   \n  \t  "  # Whitespace only
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        aggregator.add("Some content")
        time.sleep(1.1)
        
        result = aggregator.maybe_emit()
        assert result is None  # Should return None for whitespace-only summary

    def test_state_reset_after_emission(self) -> None:
        """Test that aggregator state is properly reset after emission."""
        config = MinuteAggregatorConfig(window_secs=1, soft_chars=1000)
        mock_summarizer = MagicMock(spec=Summarizer)
        mock_summarizer.summarize.return_value = "- Summary"
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        # Add content and emit
        aggregator.add("Content 1")
        old_timestamp = aggregator._last_emit_ts
        time.sleep(1.1)
        
        result = aggregator.maybe_emit()
        assert result is not None
        
        # Check that timestamp was updated
        assert aggregator._last_emit_ts > old_timestamp
        
        # Old items should be removed or filtered by time window
        current_time = time.time()
        cutoff = current_time - config.window_secs
        assert all(item[0] >= cutoff for item in aggregator._items)

    def test_window_slicing(self) -> None:
        """Test that items are properly sliced by time window."""
        config = MinuteAggregatorConfig(window_secs=2, soft_chars=1000)
        mock_summarizer = MagicMock(spec=Summarizer)
        
        def capture_summarize_args(texts):
            capture_summarize_args.last_texts = texts
            return "- Summary"
        
        mock_summarizer.summarize.side_effect = capture_summarize_args
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        # Add old content
        aggregator.add("Old content")
        time.sleep(1.5)
        
        # Add recent content
        aggregator.add("Recent content 1")
        aggregator.add("Recent content 2")
        time.sleep(1.0)  # Total elapsed > window_secs
        
        result = aggregator.maybe_emit()
        assert result is not None
        
        # Check that only recent content was summarized
        summarized_texts = capture_summarize_args.last_texts
        assert "Recent content 1" in summarized_texts
        assert "Recent content 2" in summarized_texts
        # Old content should be filtered out by time window

    def test_empty_window_fallback(self) -> None:
        """Test fallback behavior when time window is empty."""
        config = MinuteAggregatorConfig(window_secs=1, soft_chars=1000)
        mock_summarizer = MagicMock(spec=Summarizer)
        
        def capture_summarize_args(texts):
            capture_summarize_args.last_texts = texts
            return "- Summary"
        
        mock_summarizer.summarize.side_effect = capture_summarize_args
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        # Add content and wait long enough that it's outside the window
        aggregator.add("Content 1")
        aggregator.add("Content 2")
        aggregator.add("Content 3")
        aggregator.add("Content 4")
        aggregator.add("Content 5")
        aggregator.add("Content 6")  # Add 6 items
        
        time.sleep(2.0)  # Wait longer than window
        
        result = aggregator.maybe_emit()
        assert result is not None
        
        # Should fall back to last 5 items when window is empty
        summarized_texts = capture_summarize_args.last_texts
        assert len(summarized_texts) == 5

    def test_short_description_generation(self) -> None:
        """Test generation of short description from summary."""
        config = MinuteAggregatorConfig(window_secs=1, soft_chars=1000)
        mock_summarizer = MagicMock(spec=Summarizer)
        mock_summarizer.summarize.return_value = (
            "- Discussed project timeline and deliverables for Q4\n"
            "- Assigned tasks to team members\n"
            "- Set up next meeting for next week"
        )
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        aggregator.add("Meeting content")
        time.sleep(1.1)
        
        result = aggregator.maybe_emit()
        assert result is not None
        
        summary, short = result
        # Short description should be derived from first bullet point
        assert "Discussed project timeline" in short
        assert len(short) <= 120  # Should be truncated if too long

    def test_short_description_with_markers(self) -> None:
        """Test short description generation strips bullet markers."""
        config = MinuteAggregatorConfig(window_secs=1, soft_chars=1000)
        mock_summarizer = MagicMock(spec=Summarizer)
        mock_summarizer.summarize.return_value = "* Main discussion point\n- Another point"
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        aggregator.add("Content")
        time.sleep(1.1)
        
        result = aggregator.maybe_emit()
        assert result is not None
        
        summary, short = result
        # Should strip bullet markers
        assert not short.startswith("*")
        assert not short.startswith("-")
        assert "Main discussion point" in short

    def test_thread_safety(self) -> None:
        """Test basic thread safety of add and maybe_emit operations."""
        import threading
        
        config = MinuteAggregatorConfig()
        mock_summarizer = MagicMock(spec=Summarizer)
        aggregator = MinuteAggregator(config, mock_summarizer)
        
        def add_items():
            for i in range(10):
                aggregator.add(f"Item {i}")
                time.sleep(0.01)
        
        # Run add operations in separate threads
        threads = [threading.Thread(target=add_items) for _ in range(3)]
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have collected all items without corruption
        assert len(aggregator._items) == 30