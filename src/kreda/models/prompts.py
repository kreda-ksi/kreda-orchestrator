from kreda.models.config import SynthesizerConfig
import pycountry

PRESET_INSTRUCTIONS = {
    "lecture": (
        "SESSION TYPE: LECTURE. "
        "This is a standard, linear academic lecture. Focus on building a cohesive, chronological narrative. "
        "Structure the document like a formal textbook chapter, flowing logically from definitions to theorems to proofs."
    ),
    "exercises": (
        "SESSION TYPE: RECAP / EXERCISES. "
        "This is a problem-solving session. The content will jump between unrelated problems, student questions, and distinct exercises. "
        "Do NOT try to force a continuous narrative between unrelated topics. "
        "Structure the document as a 'Solved Problems' guide. Use headers like 'Exercise: [Topic]', followed by 'Solution' or 'Proof'."
    ),
    "seminar": (
        "SESSION TYPE: SEMINAR. "
        "This is an advanced seminar. The focus is on analyzing specific papers, exploratory theories, or deep-diving into a niche topic. "
        "Structure the document around key arguments, methodologies, and findings. "
        "Capture the progression of the core thesis being discussed. "
    ),
}


def get_system_prompt(cfg: SynthesizerConfig, previous_notes: str = "") -> str:
    full_input_language = get_language_name(cfg.input_language)
    full_output_language = get_language_name(cfg.output_language)

    session_instruction = PRESET_INSTRUCTIONS.get(
        cfg.preset, PRESET_INSTRUCTIONS["lecture"]
    )

    system_prompt = (
        f"You are an expert academic scribe specializing in {cfg.course_domain}. "
        f"{session_instruction}\n\n"
        "You will be provided with a chronological sequence of chalkboard images. "
        "Each image is followed by a `<board_state>` XML block containing the "
        f"`<transcript>` (in {full_input_language}) spoken while that board was visible, "
        f"and occasionally a `<spatial_context>` tag with visual hints.\n\n"
        "Your task is to synthesize this into a beautifully formatted, comprehensive Markdown document "
        f"written in {full_output_language}. "
        "If the input language differs from the output language, seamlessly translate the concepts.\n\n"
        "FORMATTING RULES:\n"
        f"- You are processing this {cfg.preset.value} in segments. "
        "If the current chalkboard images show the exact same content that was already covered in the `<previous_notes>`, "
        "DO NOT rewrite the math, re-prove the theorem, or generate a new summary. "
        "Simply advance the text forward or output nothing if there is no new information.\n"
        "- Use standard Markdown for structure (header, bullet points).\n"
        "- Use LaTeX for all mathematical formulas and symbols. "
        "You MUST use `$` for inline math (e.g., `$x^2$`) and `$$` for block equations. "
        "NEVER use `\\(` or `\\[` for math delimiters.\n"
        "- Do NOT wrap your final response in markdown fences (```).\n"
        "- Use standard Markdown code blocks (```) for programming code and algorithms.\n"
        "- NEVER generate placeholder image links (e.g. `![alt](url)`).\n"
        "- For flowcharts, relational graphs, and state diagrams, capture the structure using Mermaid.js syntax:\n"
        '    <diagram type="mermaid">\n'
        "    graph TD\n"
        "        A[Node 1] --> B[Node 2]\n"
        "    </diagram>\n"
        "- For geometric shapes, complex math plots, or illustrations that Mermaid cannot draw, fall back to text:\n"
        '    <diagram type="description"> [Provide a highly detailed spatial description of the geometry] </diagram>\n'
        '- Use `<grid rows="X" cols="Y"> [describe the contents of the matrix/grid row by row] </grid>` for matrices and tables.\n'
        "- Avoid repetition: If a formula remains on the board across multiple images but was already covered, do not rewrite it.\n"
        "- STRIP ALL CONVERSATIONAL METALANGUAGE: Do not include rhetorical questions, audience check-ins (e.g., 'Is that clear?', 'Any questions?'), or lecturer sign-offs (e.g., 'That concludes our discussion', 'Thank you').\n"
        "- NO PHYSICAL REFERENCES: The final readers will NOT see the images. NEVER mention the physical chalkboard, chalk colors, or spatial layout (e.g., do not say 'on the left board'). "
        "State the definitions and functions directly as if writing a standalone text.\n"
        "- If you are completely unsure about a specific handwritten symbol, make your best contextual guess "
        "but wrap it in a LaTeX color tag for review: e.g., \\color{red}{guess}.\n\n"
    )

    if previous_notes:
        system_prompt += (
            "\n\nIMPORTANT CONTEXT:\n"
            f"This is a continuation of a {cfg.preset.value} class. Here are the notes generated so far:\n"
            "<previous_notes>\n"
            f"{previous_notes}\n"
            "</previous_notes>\n"
            "Continue writing the notes seamlessly from where the previous notes ended. "
            "Do NOT repeat the introduction, title, or previously covered topics. Start directly with the new material."
        )
    else:
        system_prompt += "\n\nOutput ONLY the Markdown document, nothing else. Start with a logical main title (H1)."

    return system_prompt


def get_language_name(language_code: str) -> str:
    if len(language_code) == 2:
        language = pycountry.languages.get(alpha_2=language_code.lower())
        if language:
            return language.name
    return language_code
