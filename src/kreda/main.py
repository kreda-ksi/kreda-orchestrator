import typer
from pathlib import Path

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

    typer.secho("Process finished successfully.")


if __name__ == "__main__":
    app()
