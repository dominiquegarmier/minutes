"""Tests for file writing functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path


from minutes.writer import NotesWriter, NotesWriterConfig


class TestNotesWriterConfig:
    """Test notes writer configuration dataclass."""

    def test_initialization(self) -> None:
        """Test configuration initialization."""
        output_dir = Path("/tmp/test")
        config = NotesWriterConfig(output_dir=output_dir)
        assert config.output_dir == output_dir


class TestNotesWriter:
    """Test notes writer functionality."""

    def test_initialization_creates_directory(self) -> None:
        """Test that initialization creates output directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "new_dir"
            config = NotesWriterConfig(output_dir=output_dir)

            assert not output_dir.exists()
            writer = NotesWriter(config)
            assert output_dir.exists()
            assert writer.summary_file == output_dir / "SUMMARY.md"

    def test_initialization_creates_summary_file(self) -> None:
        """Test that initialization creates SUMMARY.md if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            config = NotesWriterConfig(output_dir=output_dir)

            writer = NotesWriter(config)

            assert writer.summary_file.exists()
            content = writer.summary_file.read_text(encoding="utf-8")
            assert content == "# Minutes Summary\n\n"

    def test_initialization_preserves_existing_summary(self) -> None:
        """Test that existing SUMMARY.md is not overwritten."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            summary_file = output_dir / "SUMMARY.md"

            # Pre-create summary file with content
            existing_content = "# Existing Summary\n\n- Previous entry\n"
            summary_file.write_text(existing_content, encoding="utf-8")

            config = NotesWriterConfig(output_dir=output_dir)
            writer = NotesWriter(config)

            # Should preserve existing content
            content = writer.summary_file.read_text(encoding="utf-8")
            assert content == existing_content

    def test_write_minute_creates_file(self) -> None:
        """Test that write_minute creates a properly formatted file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = NotesWriterConfig(output_dir=Path(tmp_dir))
            writer = NotesWriter(config)

            test_content = "- First bullet point\n- Second bullet point"
            result_path = writer.write_minute(test_content)

            assert result_path.exists()
            assert result_path.parent == Path(tmp_dir)
            assert result_path.name.startswith("minute_")
            assert result_path.suffix == ".md"

            # Check file content
            content = result_path.read_text(encoding="utf-8")
            assert "---" in content  # Should have frontmatter
            assert "created:" in content
            assert test_content in content

    def test_write_minute_filename_format(self) -> None:
        """Test that minute files have correct timestamp format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = NotesWriterConfig(output_dir=Path(tmp_dir))
            writer = NotesWriter(config)

            result_path = writer.write_minute("Test content")

            # Should match minute_YYYY-MM-DD_HH-MM-SS.md pattern
            filename = result_path.name
            assert filename.startswith("minute_")
            assert filename.endswith(".md")

            # Extract timestamp part
            timestamp_part = filename[7:-3]  # Remove "minute_" and ".md"
            parts = timestamp_part.split("_")
            assert len(parts) == 2  # date_time

            date_part, time_part = parts
            assert len(date_part.split("-")) == 3  # YYYY-MM-DD
            assert len(time_part.split("-")) == 3  # HH-MM-SS

    def test_write_minute_frontmatter(self) -> None:
        """Test that frontmatter is properly formatted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = NotesWriterConfig(output_dir=Path(tmp_dir))
            writer = NotesWriter(config)

            result_path = writer.write_minute("Content")
            content = result_path.read_text(encoding="utf-8")

            lines = content.split("\n")
            assert lines[0] == "---"
            assert lines[1].startswith("created: ")
            assert lines[2] == "---"
            assert lines[3] == ""
            assert lines[4] == "Content"

    def test_write_minute_strips_content(self) -> None:
        """Test that content is properly stripped."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = NotesWriterConfig(output_dir=Path(tmp_dir))
            writer = NotesWriter(config)

            content_with_whitespace = "  \n  - Point 1\n- Point 2  \n  "
            result_path = writer.write_minute(content_with_whitespace)

            file_content = result_path.read_text(encoding="utf-8")
            # Should strip leading/trailing whitespace
            assert file_content.endswith("- Point 2\n")
            assert "- Point 1" in file_content

    def test_append_summary_adds_entry(self) -> None:
        """Test that append_summary adds entry to SUMMARY.md."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = NotesWriterConfig(output_dir=Path(tmp_dir))
            writer = NotesWriter(config)

            # Create a minute file
            note_path = Path(tmp_dir) / "minute_2024-01-01_12-00-00.md"
            note_path.write_text("Test content")

            writer.append_summary(note_path, "Short description")

            summary_content = writer.summary_file.read_text(encoding="utf-8")
            expected_line = "- [minute_2024-01-01_12-00-00.md](minute_2024-01-01_12-00-00.md) — Short description"
            assert expected_line in summary_content

    def test_append_summary_multiple_entries(self) -> None:
        """Test appending multiple entries to summary."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = NotesWriterConfig(output_dir=Path(tmp_dir))
            writer = NotesWriter(config)

            # Add multiple entries
            note1 = Path(tmp_dir) / "minute_1.md"
            note2 = Path(tmp_dir) / "minute_2.md"

            writer.append_summary(note1, "First description")
            writer.append_summary(note2, "Second description")

            summary_content = writer.summary_file.read_text(encoding="utf-8")
            assert "First description" in summary_content
            assert "Second description" in summary_content

            # Should maintain order
            lines = summary_content.split("\n")
            first_entry_line = next(
                i for i, line in enumerate(lines) if "First description" in line
            )
            second_entry_line = next(
                i for i, line in enumerate(lines) if "Second description" in line
            )
            assert first_entry_line < second_entry_line

    def test_append_summary_preserves_existing_content(self) -> None:
        """Test that append_summary preserves existing entries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            summary_file = output_dir / "SUMMARY.md"

            # Pre-populate summary
            existing_content = (
                "# Minutes Summary\n\n- [existing.md](existing.md) — Existing entry\n"
            )
            summary_file.write_text(existing_content, encoding="utf-8")

            config = NotesWriterConfig(output_dir=output_dir)
            writer = NotesWriter(config)

            note_path = Path(tmp_dir) / "new_note.md"
            writer.append_summary(note_path, "New entry")

            final_content = writer.summary_file.read_text(encoding="utf-8")
            assert "Existing entry" in final_content
            assert "New entry" in final_content

    def test_thread_safety_summary_append(self) -> None:
        """Test thread safety of summary append operations."""
        import threading
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = NotesWriterConfig(output_dir=Path(tmp_dir))
            writer = NotesWriter(config)

            def append_entries(thread_id):
                for i in range(5):
                    note_path = Path(tmp_dir) / f"note_{thread_id}_{i}.md"
                    writer.append_summary(note_path, f"Description {thread_id}-{i}")
                    time.sleep(0.01)

            # Run multiple threads appending to summary
            threads = [
                threading.Thread(target=append_entries, args=(i,)) for i in range(3)
            ]
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # Check that all entries were written
            summary_content = writer.summary_file.read_text(encoding="utf-8")

            # Should have 15 entries (3 threads * 5 entries each)
            entry_count = summary_content.count("— Description")
            assert entry_count == 15

    def test_relative_path_in_summary(self) -> None:
        """Test that summary uses relative paths for links."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = NotesWriterConfig(output_dir=Path(tmp_dir))
            writer = NotesWriter(config)

            # Use absolute path for note
            note_path = Path(tmp_dir) / "test_note.md"
            writer.append_summary(note_path, "Test description")

            summary_content = writer.summary_file.read_text(encoding="utf-8")

            # Should use just the filename, not absolute path
            assert "[test_note.md](test_note.md)" in summary_content
            assert str(tmp_dir) not in summary_content

    def test_special_characters_in_description(self) -> None:
        """Test handling of special characters in descriptions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = NotesWriterConfig(output_dir=Path(tmp_dir))
            writer = NotesWriter(config)

            note_path = Path(tmp_dir) / "note.md"
            special_description = (
                "Description with [brackets] and (parentheses) & symbols"
            )
            writer.append_summary(note_path, special_description)

            summary_content = writer.summary_file.read_text(encoding="utf-8")
            assert special_description in summary_content

    def test_empty_description(self) -> None:
        """Test handling of empty description."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = NotesWriterConfig(output_dir=Path(tmp_dir))
            writer = NotesWriter(config)

            note_path = Path(tmp_dir) / "note.md"
            writer.append_summary(note_path, "")

            summary_content = writer.summary_file.read_text(encoding="utf-8")
            assert "[note.md](note.md) — " in summary_content
