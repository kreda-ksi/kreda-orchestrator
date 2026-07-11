from pathlib import Path
from openai import OpenAI
from typing import Any
from kreda.models.config import TypstConfig


def run(typst_out_path: Path, markdown_text: str, cfg: TypstConfig):

    typst_preamble = (
        '#set page(paper: "a4", margin: 1.5in)\n'
        '#set text(font: "New Computer Modern", size: 11pt)\n'
        '#set heading(numbering: "1.1.")\n'
        "#set par(justify: true)\n\n"
    )

    system_prompt = (
        "You are an expert academic typesetter and a master of the Typst markup language. "
        "Your task is to translate a Markdown document containing LaTeX math and XML tags into pristine, compilable Typst code.\n\n"
        "CRITICAL TRANSLATION RULES:\n"
        "1. Zero Content Alteration: Do NOT summarize, add, or remove any information from the source text. Translate it exactly.\n"
        "2. Headers: Convert Markdown headers (`#`, `##`) to Typst headers (`=`, `==`).\n"
        "3. Math Translation: You must translate ALL LaTeX math syntax into native Typst math syntax.\n"
        "   - Inline math stays in `$ ... $` without line breaks.\n"
        "   - Display (block) math MUST go on separate lines like this:\n"
        "   $\n"
        "   math content here\n"
        "   $\n"
        "   - REMOVE ALL BACKSLASHES: Typst does not use `\\` for symbols. `\\forall` becomes `forall`, `\\alpha` becomes `alpha`, `\\in` becomes `in`.\n"
        "   - Brackets: Remove `\\left` and `\\right`. Typst scales brackets automatically. Just use `(` and `)`.\n"
        "   - Arrows: `\\to` or `\\rightarrow` becomes `->`. `\\implies` becomes `=>`.\n"
        "   - Fractions: `\\frac{a}{b}` becomes `a / b`.\n"
        "   - Sets/logic: `\\mathbb{N}` becomes `NN`, `\\land` becomes `and`.\n"
        "   - Convert Greek letters and operators to Typst equivalents.\n"
        "4. XML Diagrams: When you encounter a `<diagram>` tag, do NOT output the XML. "
        "Instead, create a beautifully styled Typst block to display the description. "
        "Use this exact syntax (ensure all brackets are closed):\n"
        "   `#rect(fill: luma(240), stroke: 1pt + black, width: 100%, inset: 1em)[*Visualization:* \\ Description here]`\n"
        "5. XML Grids: When you encounter a `<grid>` tag, translate it into a native Typst `#table` or `#grid`.\n"
        f"6. Preamble: You MUST start your response exactly with this preamble:\n{typst_preamble}\n"
        "Output Format: Output ONLY the raw Typst code. Do NOT wrap it in ```typst ... ``` markdown fences."
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": markdown_text},
    ]

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,  # type: ignore
        max_tokens=4096,
        temperature=0.1,
    )

    code = response.choices[0].message.content

    if code:
        with open(typst_out_path, "w") as f:
            f.write(code)
