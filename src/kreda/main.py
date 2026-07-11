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
    vlm_model: str = typer.Option(
        "gpt-4o",
        "--vlm-model",
        "-vm",
        help="The Vision-Language Model to use.",
    ),
    vlm_api_key: str = typer.Option(
        None,
        "--vlm-key",
        "-vk",
        envvar="VLM_API_KEY",
        help="VLM API key (defaults to VLM_API_KEY env variable)",
    ),
    vlm_base_url: str = typer.Option(
        None,
        "--vlm-base",
        "-vb",
        help="Custom API base URL for VLM (e.g., http://localhost:8000/v1 for vLLM)",
    ),
    vlm_chunk_size: int = typer.Option(
        15,
        "--vlm-chunk",
        "-vc",
        help="Number of board states to send per API call to prevent token overflow.",
    ),
    vlm_output_file: Path = typer.Option(
        "notes.md",
        "--vlm-output",
        "-vo",
        help="VLM output file storing intermediate Markdown representation (relative to run path)",
    ),
    typst_model: str = typer.Option(
        "gpt-4o",
        "--typst-model",
        help="The LLM to use for the Markdown-to-Typst text conversion step.",
    ),
    typst_api_key: str = typer.Option(
        None,
        "--typst-key",
        "-tk",
        envvar="TYPST_API_KEY",
        help="Typst LLM API key (defaults to TYPST_API_KEY env variable)",
    ),
    typst_base_url: str = typer.Option(
        None,
        "--typst-base",
        "-tb",
        help="Custom API base URL for Typst generation (e.g., http://localhost:8000/v1 for vLLM)",
    ),
    typst_file: Path = typer.Option(
        "source.typ",
        "--typst-file",
        "-tf",
        help="Typst output file path (relative to run path)",
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

    # step 3 (synthesizer+generator)

    from kreda.pipeline.generator import run as generator_run
    from kreda.models.config import SynthesizerConfig
    from kreda.models.config import GeneratorConfig

    synthesizer_cfg = SynthesizerConfig(
        input_language=input_language,
        output_language=output_language,
        course_domain=course_domain,
    )

    generator_cfg = GeneratorConfig(
        model=vlm_model,
        api_key=vlm_api_key,
        base_url=vlm_base_url,
        chunk_size=vlm_chunk_size,
    )

    typer.echo(f"Sending payload to {vlm_model}...")

    try:
        markdown_text = generator_run(
            curated_segments=curated_segments,
            run_path=run_path,
            output_file=vlm_output_file,
            synthesizer_cfg=synthesizer_cfg,
            generator_cfg=generator_cfg,
        )

        # step 4 (typst generation)

        from kreda.pipeline.typst import run as typst_run
        from kreda.models.config import TypstConfig

        typst_cfg = TypstConfig(
            model=typst_model,
            api_key=typst_api_key,
            base_url=typst_base_url,
        )

        typer.echo(f"Generating typst note with {typst_model}...")

        typst_run(run_path / typst_file, markdown_text, typst_cfg)

    except Exception as e:
        typer.secho(f"API error during generation: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
