# Minutes

A Python CLI application that continuously records desktop audio, transcribes it with whisper.cpp, and generates meeting notes using local Ollama reasoning models.

## Features

- 🎙️ **Desktop Audio Recording**: Captures system audio with optional microphone input
- 🗣️ **Real-time Transcription**: Uses whisper.cpp for high-quality speech-to-text
- 🤖 **AI Summarization**: Leverages local Ollama models for intelligent meeting notes
- 👥 **Speaker Identification**: Attempts to identify different speakers in conversations
- 📝 **Markdown Output**: Generates timestamped markdown files with automatic indexing
- ⚡ **Streaming Pipeline**: Processes audio in overlapping segments for continuity

## Quick Start

1. **Install from GitHub:**
   ```bash
   pip install git+https://github.com/dominiquegarmier/minutes.git
   ```

2. **Set up dependencies:**
   ```bash
   # 1. Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # 2. Start Ollama in background and pull model
   ollama serve &
   ollama pull llama3.1:8b  # Best balance of quality/performance (4.7GB)

   # 3. Install whisper-cli (choose one method)
   brew install whisper-cpp  # macOS
   # OR build from source: https://github.com/ggml-org/whisper.cpp

   # 4. Download whisper model
   mkdir -p models && curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -o models/ggml-base.en.bin
   ```

3. **Run the application:**
   ```bash
   minutes run --whisper-model ./models/ggml-base.en.bin
   ```

4. **View your notes:**
   Notes are saved in the `./meeting-notes/` directory with an auto-generated `SUMMARY.md` index.

## Development

```bash
git clone https://github.com/dominiquegarmier/minutes.git
cd minutes
uv sync --all-extras
uv run python demo.py  # Test functionality
```

## Architecture

The application uses a multi-threaded pipeline:
1. **AudioRecorder** → Captures desktop audio in overlapping segments
2. **Transcriber** → Processes audio through whisper.cpp subprocess
3. **Aggregator** → Collects transcripts and triggers summarization
4. **Summarizer** → Uses Ollama API for intelligent note generation
5. **Writer** → Saves markdown files and maintains summary index

See `CLAUDE.md` for detailed development documentation.
