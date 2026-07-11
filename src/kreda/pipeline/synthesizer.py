import base64
import json
from pathlib import Path
from typing import Any
from kreda.models.events import AlignedSegment


def encode_img_to_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


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
    curated_segments: list[AlignedSegment], run_path: Path
) -> list[dict]:
    system_prompt = (
        "You are an expert academinc scribe and mathematician."
        "You will be provided with a chronological sequence of chalkboard images"
        "and the spoken transcript corresponding to each board state."
        "Your task is to synthesize this into a beautifully formatted Markdown"
        "document using LaTeX for all mathematical formulas."
        "Ignore filler words. Correct any obvious speech-to-text typos using the visual context."
        "Output ONLY the Markdown document, nothing else."
    )

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

        # add audio context
        user_content.append(
            {
                "type": "text",
                "text": f"[Board State {idx + 1}] Spoken transcript: {segment.transcript_chunk}",
            }
        )

    # add final instruction
    user_content.append(
        {"type": "text", "text": "Please generate the final Markdown document now."}
    )

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    return messages
