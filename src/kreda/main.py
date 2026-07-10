import typer
from pathlib import Path
from kreda.models.config import WhisperConfig

app = typer.Typer(
    name="kreda-orchestrator",
    add_completion=False,
)


@app.command()
def process(
    run_path: Path = typer.Argument(
        ...,
        help="Path to the raw lecture run directory",
        exists=True,
    ),
    log_file: Path = typer.Option(
        "run.csv",
        "--log-file",
        "-lf",
        help="Override log file path (relative to run path)",
    ),
    grid_file: Path = typer.Option(
        "grid.json",
        "--grid-file",
        "-gf",
        help="Override motion grid file path (relative to run path)",
    ),
    whisper_model: str = typer.Option(
        "large-v3",
        "--whisper-model",
        "-wm",
        help="Whisper model size to use for transcription",
    ),
    whisper_device: str = typer.Option(
        "auto",
        "--whisper-device",
        "-wd",
        help="Whisper device to use for computation (auto/cpu/cuda)",
    ),
    whisper_compute_type: str = typer.Option(
        "float16",
        "--whisper-compute",
        "-wc",
        help="Whisper quantization type (float16/int8/float32)",
    ),
    whisper_language: str = typer.Option(
        "en",
        "--lang",
        "-wl",
        help="Whisper two-letter language code (e.g. 'en', 'pl')",
    ),
    whisper_beam_size: int = typer.Option(
        5,
        "--whisper-beam",
        "-wb",
        help="Whisper transcription beam size (higher = more accurate, slower)",
    ),
    whisper_lag_padding_ms: int = typer.Option(
        5000,
        "--whisper-pad",
        "-wp",
        help="Whisper audio window padding per event frame (in ms)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable verbose debug logging",
    ),
):
    if debug:
        typer.echo("Starting in debug mode...")
        typer.echo(f"Target run: {run_path}")

    whisper_cfg = WhisperConfig(
        model_size=whisper_model,
        device=whisper_device,
        compute_type=whisper_compute_type,
        language=whisper_language,
        beam_size=whisper_beam_size,
        lag_padding_ms=whisper_lag_padding_ms,
    )
    from kreda.pipeline import aligner

    audio_chunks = aligner.run(run_path, log_file, whisper_cfg)
    for line in audio_chunks:
        typer.echo(line.transcript_chunk)

    typer.secho("Process finished successfully.")


if __name__ == "__main__":
    app()
