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
    audio_file: Path = typer.Option(
        "audio.wav",
        "--audio-file",
        "-af",
        help="Override audio file to transcript (relative to run path)",
    ),
    no_audio: bool = typer.Option(
        False,
        "--no-audio",
        "-na",
        help="Run in vision-only mode. Skisp Whisper transcription.",
    ),
    input_language: str = typer.Option(
        "en",
        "--input-lang",
        "-il",
        help="Two-letter source language code (e.g. 'en', 'pl')",
    ),
    output_language: str = typer.Option(
        "en",
        "--output-lang",
        "-ol",
        help="Target language for the final notes (e.g. 'en', 'pl')",
    ),
    course_domain: str = typer.Option(
        "Computer Science and Mathematics",
        "--course-domain",
        "-cd",
        help="The academic subject (helps the AI format code vs math correctly)",
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
    assembler_diff_threshold_pixels: int = typer.Option(
        500,
        "--assembler-threshold",
        "-at",
        help="Number of changed pixels required to keep a frame (handles camera noise)",
    ),
    assembler_max_occlusion_ratio: float = typer.Option(
        0.10,
        "--assembler-occlusion",
        "-ao",
        help="Max board occlusion ratio (0.0->1.0) before applying masked diff logic",
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
        if no_audio:
            typer.echo("Audio disabled.")

    # step 1 (aligner)

    if no_audio:
        from kreda.pipeline.aligner import get_keyframes, AlignedSegment

        keyframes = get_keyframes(run_path / log_file)
        aligned_segments = [
            AlignedSegment(
                timestamp_ms=int(row["t_ms"]),
                event_type=str(row["changed_or_what"]),
                filename=str(row["filename"]),
                transcript_chunk="",
            )
            for row in keyframes.to_dict("records")
        ]
    else:
        from kreda.pipeline.aligner import run as aligner_run
        from kreda.models.config import WhisperConfig

        whisper_cfg = WhisperConfig(
            model_size=whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
            input_language=input_language,
            beam_size=whisper_beam_size,
            lag_padding_ms=whisper_lag_padding_ms,
        )

        aligned_segments = aligner_run(
            run_path / audio_file, run_path / log_file, whisper_cfg
        )

    # step 2 (assembler)

    from kreda.pipeline.assembler import run as assembler_run
    from kreda.models.config import AssemblerConfig

    assembler_cfg = AssemblerConfig(
        diff_threshold_pixels=assembler_diff_threshold_pixels,
        max_occlusion_ratio=assembler_max_occlusion_ratio,
    )

    curated_segments = assembler_run(
        aligned_segments, run_path, grid_file, assembler_cfg
    )

    # step 3 (synthesizer)

    from kreda.pipeline.synthesizer import debug_print_payload, build_vlm_payload
    from kreda.models.config import SynthesizerConfig

    synthesizer_cfg = SynthesizerConfig(
        input_language=input_language,
        output_language=output_language,
        course_domain=course_domain,
    )

    payload = build_vlm_payload(curated_segments, run_path)
    debug_print_payload(payload)

    # finish logic

    for line in curated_segments:
        typer.echo(line.transcript_chunk)

    typer.secho("Process finished successfully.")


if __name__ == "__main__":
    app()
