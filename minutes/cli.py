"""Command-line interface for the minutes application."""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import sounddevice as sd  # type: ignore[import-untyped]
import typer

from .audio import AudioConfig, DEFAULT_SR
from .aggregation import MinuteAggregatorConfig
from .pipeline import Pipeline, PipelineConfig
from .summarization import OllamaConfig
from .transcription import WhisperConfig
from .writer import NotesWriterConfig

app = typer.Typer(add_completion=True, no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo("minute-taker 0.1.0")
        raise typer.Exit()


@app.command()
def run(
    output_dir: Path = typer.Option(
        Path("./meeting-notes"), help="Directory to write minute notes and SUMMARY.md"
    ),
    device: Optional[str] = typer.Option(
        None,
        help="Audio device numbers (comma-separated, e.g., '0,1,2') for capture (see --list-devices)",
    ),
    sample_rate: int = typer.Option(
        DEFAULT_SR, help="Capture sample rate before resampling to 16k"
    ),
    loopback: bool = typer.Option(
        True, help="Use WASAPI loopback on Windows if available"
    ),
    enable_microphone: bool = typer.Option(
        True, help="Also capture microphone input along with system audio"
    ),
    microphone_device: Optional[str] = typer.Option(
        None, help="Specific microphone device to use (see --list-devices)"
    ),
    segment_secs: float = typer.Option(
        20.0, min=5.0, help="Length of each audio segment for transcription"
    ),
    overlap_secs: float = typer.Option(5.0, min=0.0, help="Overlap between segments"),
    window_secs: int = typer.Option(
        60, min=20, help="Target minutes window for summaries"
    ),
    soft_chars: int = typer.Option(
        1200, min=200, help="Emit summary early if collected this many characters"
    ),
    whisper_bin: str = typer.Option(
        "whisper-cli", help="Path to whisper.cpp binary (default: whisper-cli on PATH)"
    ),
    whisper_model: Optional[str] = typer.Option(
        None, help="Path to whisper.cpp model (e.g., ggml-base.en.bin)"
    ),
    whisper_lang: str = typer.Option("en", help="Whisper language code"),
    threads: int = typer.Option(
        max(1, os.cpu_count() or 1), help="Threads for whisper.cpp"
    ),
    ollama_model: str = typer.Option(
        "llama3.1:8b", help="Ollama model for summarization"
    ),
    ollama_host: str = typer.Option("http://127.0.0.1:11434", help="Ollama host"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True
    ),
    list_devices: bool = typer.Option(False, help="List audio devices and exit"),
) -> None:
    """Run the minute-taking pipeline until interrupted (Ctrl+C).

    This command starts audio capture, transcribes continuously with whisper.cpp,
    aggregates roughly one-minute windows, summarizes with Ollama, and writes
    markdown notes to the output directory (and updates SUMMARY.md).
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if list_devices:
        devices = sd.query_devices()
        print("Available audio input devices:")
        print("=" * 50)
        input_devices = []
        for i, device in enumerate(devices):
            # Type cast for mypy - sounddevice returns dict-like objects
            dev_dict = dict(device)  # type: ignore[arg-type]
            channels = dev_dict.get("max_input_channels", 0)
            if channels > 0:  # Only show input-capable devices
                name = dev_dict.get("name", f"Device {i}")
                input_devices.append((i, name, channels))

        for idx, (original_id, name, channels) in enumerate(input_devices):
            print(f"{idx:2d}: {name} ({channels} channels)")

        # Show current system audio output
        try:
            default_output = sd.default.device[1]  # Output device
            if default_output is not None:
                output_name = devices[default_output]["name"]
                print(f"\nCurrent system output: {output_name}")
                if "BlackHole" not in output_name:
                    print("⚠️  System audio is NOT routed through BlackHole")
                    print(
                        "   Set BlackHole 2ch as system output in Sound settings to capture system audio"
                    )
                else:
                    print("✅ System audio is routed through BlackHole")
        except:
            pass

        raise typer.Exit(code=0)

    if not whisper_model:
        print("Error: --whisper-model is required")
        raise typer.Exit(code=1)

    audio_cfg = AudioConfig(
        device=device,
        sample_rate=sample_rate,
        loopback=loopback,
        enable_microphone=enable_microphone,
        microphone_device=microphone_device,
    )
    whisper_cfg = WhisperConfig(
        whisper_bin=whisper_bin,
        model_path=whisper_model,
        language=whisper_lang,
        threads=threads,
        print_progress=False,
    )
    ollama_cfg = OllamaConfig(model=ollama_model, host=ollama_host)
    aggregator_cfg = MinuteAggregatorConfig(
        window_secs=window_secs, soft_chars=soft_chars
    )
    writer_cfg = NotesWriterConfig(output_dir=output_dir)

    pipeline_cfg = PipelineConfig(
        audio=audio_cfg,
        whisper=whisper_cfg,
        ollama=ollama_cfg,
        aggregator=aggregator_cfg,
        writer=writer_cfg,
        segment_secs=segment_secs,
        overlap_secs=overlap_secs,
    )

    pipe = Pipeline(pipeline_cfg)

    # Graceful shutdown
    def _shutdown(sig, frame):  # type: ignore[no-untyped-def]
        logging.info("Signal %s received, shutting down…", sig)
        pipe.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    pipe.start()

    # Keep main thread alive
    while True:
        time.sleep(1)


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
