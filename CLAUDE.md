# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python 3.12 CLI application that continuously records desktop audio, transcribes it using whisper.cpp, and generates minute-sized meeting notes using a local Ollama reasoning model. The application is designed as a single-file, modular system with clean separation of concerns.

## System Dependencies

- **whisper.cpp**: Compiled binary and model file (e.g., `ggml-base.en.bin`)
- **Ollama**: Local server running at `http://127.0.0.1:11434` with a reasoning model pulled
- **Desktop audio capture**: Platform-specific requirements
  - Windows: WASAPI loopback (built-in)
  - macOS: Virtual audio device like BlackHole
  - Linux: PulseAudio/ALSA monitor sources

## Python Dependencies

Install with: `pip install typer sounddevice soundfile numpy requests`

Optional: `pip install colorama` (for better Windows Ctrl+C handling)

## Development Commands

```bash
# Install development dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Type checking
uv run mypy minutes/

# Linting and formatting  
uv run ruff check minutes/
uv run ruff format minutes/
uv run flake8 minutes/ --max-line-length=88 --extend-ignore=E203,W503

# Build package
uv build --wheel --sdist

# Install from source
pip install git+https://github.com/dominiquegarmier/minutes.git
```

## Running the Application

Basic usage:
```bash
# Using the installed CLI
minutes run \
  --output-dir ./minutes \
  --whisper-model ./models/ggml-base.en.bin \
  --whisper-bin ./main \
  --ollama-model "llama3.1:8b-instruct"

# Or using uv run
uv run minutes run --help
```

Key commands:
- `minutes run --help` - Show all available options
- `minutes run --list-devices` - List available audio devices

## Architecture

The application follows a pipeline architecture with these key components:

### Core Pipeline Flow
1. **AudioRecorder** → Captures desktop audio in overlapping segments
2. **Transcriber** → Converts audio segments to text via whisper.cpp subprocess
3. **MinuteAggregator** → Collects transcripts and triggers summary generation
4. **Summarizer** → Uses Ollama API to generate concise meeting notes
5. **NotesWriter** → Persists markdown notes and maintains SUMMARY.md index

### Threading Model
- Main thread: CLI and signal handling
- AudioRecorder: Background audio capture thread
- SegProducer: Generates audio segments from recorder
- Transcriber: Processes audio segments through whisper.cpp
- Aggregator: Monitors for summary triggers and writes output

### Configuration Pattern
Each component uses a dedicated `@dataclass` config object (e.g., `AudioConfig`, `WhisperConfig`) that's assembled into a `PipelineConfig` for the main `Pipeline` orchestrator.

### Output Structure
- Individual notes: `minute_YYYY-MM-DD_HH-MM-SS.md`
- Index file: `SUMMARY.md` with links and descriptions
- Frontmatter: Each note includes creation timestamp

## Audio Handling Details

- Captures stereo audio, converts to mono, resamples to 16kHz for whisper.cpp
- Uses overlapping segments (default: 20s segments, 5s overlap) for better transcription continuity
- Temporary WAV files are automatically cleaned up after transcription
- Audio buffer has rolling maxlen to prevent memory issues

## Summarization Logic

- Aggregates transcripts over configurable windows (default: 60s or 1200 chars, whichever first)
- Uses system prompt optimized for concise, actionable meeting notes
- Generates markdown bullet lists prioritizing decisions, owners, and action items
- Falls back gracefully on API failures or empty content