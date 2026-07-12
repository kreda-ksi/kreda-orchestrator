import re
from pathlib import Path


def run(markdown_text: str, out_path: Path):
    typst_preamble = (
        '#import "@preview/merman:0.1.0": mermaid\n'
        '#import "@preview/mitex:0.2.7": *\n'
        '#set page(paper: "a4", margin: 1.5in)\n'
        '#set text(font: "New Computer Modern", size: 11pt)\n'
        '#set heading(numbering: "1.1.")\n'
        "#set par(justify: true)\n\n"
    )

    text = markdown_text

    text = re.sub(r"^#### (.*)", r"==== \1", text, flags=re.M)
    text = re.sub(r"^### (.*)", r"=== \1", text, flags=re.M)
    text = re.sub(r"## (.*)", r"== \1", text, flags=re.M)
    text = re.sub(r"# (.*)", r"= \1", text, flags=re.M)

    text = re.sub(r"\*\*(.*?)\*\*", r"#strong[\1]", text, flags=re.S)
    text = re.sub(
        r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"#emph[\1]", text, flags=re.S
    )

    text = re.sub(
        r'<diagram type="mermaid">\s*(.*?)\s*</diagram>',
        r'#mermaid("\n\1\n")',
        text,
        flags=re.S,
    )

    text = re.sub(
        r'<diagram type="description">\s*(.*?)\s*</diagram>',
        r"#rect(fill: luma(240), stroke: 1pt + black, width: 100%, inset: 1em)[\1]",
        text,
        flags=re.S,
    )

    text = re.sub(r"\$\$(.*?)\$\$", r"#mitex(`\1`)", text, flags=re.S)
    text = re.sub(r"\$([^$]+)\$", r"#mi(`\1`)", text)

    typst = typst_preamble + text

    with open(out_path, "w") as f:
        f.write(typst)
