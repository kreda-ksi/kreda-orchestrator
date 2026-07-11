import base64
import json
from pathlib import Path
from typing import Any
from kreda.models.events import AlignedSegment
from kreda.models.config import SynthesizerConfig
import pycountry


def encode_img_to_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_language_name(language_code: str) -> str:
    if len(language_code) == 2:
        language = pycountry.languages.get(alpha_2=language_code.lower())
        if language:
            return language.name
    return language_code


def debug_print_payload(payload: list[dict]):
    debug_payload = []

    for message in payload:
        new_msg = message.copy()

        if isinstance(message["content"], list):
            new_content = []
            for item in message["content"]:
                if item["type"] == "image_url":
                    new_content.append(
                        {"type": "image_url", "image_url": {"url": "IMAGE GOES HERE"}}
                    )
                else:
                    new_content.append(item)
            new_msg["content"] = new_content

        debug_payload.append(new_msg)

    print(json.dumps(debug_payload, indent=2))


def build_vlm_payload(
    curated_segments: list[AlignedSegment],
    run_path: Path,
    cfg: SynthesizerConfig,
    previous_notes: str = "",
) -> list[dict]:

    full_input_language = get_language_name(cfg.input_language)
    full_output_language = get_language_name(cfg.output_language)

    system_prompt = (
        f"You are an expert academic scribe specializing in {cfg.course_domain}. "
        "You will be provided with a chronological sequence of chalkboard images. "
        "Each image is followed by a `<board_state>` XML block containing the "
        f"`<transcript>` (in {full_input_language}) spoken while that board was visible, "
        f"and occasionally a `<spatial_context>` tag with visual hints.\n\n"
        "Your task is to synthesize this into a beautifully formatted, comprehensive Markdown document "
        f"written in {full_output_language}. "
        "If the input language differs from the output language, seamlessly translate the concepts.\n\n"
        "FORMATTING RULES:\n"
        "- Use standard Markdown for structure (header, bullet points).\n"
        "- Use LaTeX for all mathematical formulas and symbols. "
        "You MUST use `$` for inline math (e.g., `$x^2$`) and `$$` for block equations. "
        "NEVER use `\\(` or `\\[` for math delimiters.\n"
        "- Do NOT wrap your final response in ```markdown fences.\n"
        "- Use standard Markdown code blocks (```) for programming code, algorithms.\n"
        "- NEVER generate placeholder image links (e.g. `![alt](url)`).\n"
        "- When the professor draws a diagram, graph or visual grid, describe it using custom XML tags for downstream processing.\n"
        '- Use `<diagram type="graph|geometry|flowchart"> [detailed spatial and semantic description] </diagram>`.\n'
        '- Use `<grid rows="X" cols="Y"> [describe the contents of the matrix/grid row by row] </grid>`.\n'
        "- Ensure the text inside these tags provides enough geometric and mathematical detail to perfectly reconstruct the visual in Typst later.\n"
        "- Avoid repetition: If a formula remains on the board across multiple images but was already covered, do not rewrite it.\n"
        "- Ignore spoken filler words or tangents. Correct obvious speech-to-text typos using visual context.\n"
        "- If you are completely unsure about a specific handwritten symbol, make your best contextual guess "
        "but wrap it in a LaTeX color tag for review: e.g., \\color{red}{guess}.\n\n"
    )

    if previous_notes:
        system_prompt += (
            "\n\nIMPORTANT CONTEXT:\n"
            "This is a continuation of a lecture. Here are the notes generated so far:\n"
            "<previous_notes>\n"
            f"{previous_notes}\n"
            "</previous_notes>\n"
            "Continue writing the notes seamlessly from where the previous notes ended. "
            "Do NOT repeat the introduction, title, or previously covered topics. Start directly with the new material."
        )
    else:
        system_prompt += "\n\nOutput ONLY the Markdown document, nothing else. Start with a logical main title (H1)."

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    user_content = []

    for idx, segment in enumerate(curated_segments):
        img_path = run_path / segment.filename

        if not img_path.exists():
            continue

        b64_img = encode_img_to_b64(img_path)

        # add board state
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_img}",
                    "detail": "high",
                },
            }
        )

        # add audio context and positional hint
        hint_xml = (
            f"    <spatial_context>{segment.spatial_hint}</spatial_context>\n"
            if segment.spatial_hint
            else ""
        )
        text_payload = (
            f'<board_state index="{idx + 1}">\n'
            f"{hint_xml}"
            f"    <transcript>{segment.transcript_chunk}</transcript>\n"
            f"</board_state>"
        )
        user_content.append(
            {
                "type": "text",
                "text": text_payload,
            }
        )

    # add final instruction
    user_content.append(
        {
            "type": "text",
            "text": "Please generate the final Markdown document now, adhering strictly to the formatting and LaTeX rules.",
        }
    )

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    return messages
