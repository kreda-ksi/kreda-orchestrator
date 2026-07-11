from pathlib import Path
from typing import Any
from openai import OpenAI
from kreda.models.config import GeneratorConfig, SynthesizerConfig
from kreda.models.events import AlignedSegment
from kreda.pipeline.synthesizer import build_vlm_payload
import typer


def _call_llm(payload: list[dict[str, Any]], cfg: GeneratorConfig) -> str:
    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )

    response = client.chat.completions.create(
        model=cfg.model,
        messages=payload,  # type: ignore
        max_tokens=4096,
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


def run(
    curated_segments: list[AlignedSegment],
    run_path: Path,
    output_file: Path,
    synthesizer_cfg: SynthesizerConfig,
    generator_cfg: GeneratorConfig,
) -> str:
    accumulated_notes = ""

    chunks = [
        curated_segments[i : i + generator_cfg.chunk_size]
        for i in range(0, len(curated_segments), generator_cfg.chunk_size)
    ]

    output_path = run_path / output_file

    if output_path.exists():
        output_path.unlink()

    for idx, chunk in enumerate(chunks):
        typer.echo(f"Processing chunk {idx + 1}/{len(chunks)}...")

        payload = build_vlm_payload(
            curated_segments=chunk,
            run_path=run_path,
            cfg=synthesizer_cfg,
            previous_notes=accumulated_notes,
        )

        chunk_markdown = _call_llm(payload, generator_cfg)

        accumulated_notes += "\n\n" + chunk_markdown

        with open(run_path / output_file, "a") as f:
            f.write(chunk_markdown + "\n\n")

    return accumulated_notes
