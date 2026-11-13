"""Build the arXiv source and PDF from docs/paper/draft.md (pdfLaTeX, as arXiv compiles).

  uv run python scripts/build_paper_latex.py            # writes docs/paper/arxiv/{main.tex, figures/, main.pdf, arxiv-source.tar.gz}

The Markdown stays the single source: title, author and abstract are lifted into the LaTeX preamble, section numbers
are kept as written (secnumdepth 0, so "Section 5.3" in the text matches the heading), figure captions keep their own
"Figure N." text, and every non-ASCII character is mapped for pdfLaTeX with newunicodechar.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/paper/draft.md"
OUT = ROOT / "docs/paper/arxiv"

UNI = {
    "−": r"\ensuremath{-}", "×": r"\ensuremath{\times}", "→": r"\ensuremath{\rightarrow}", "±": r"\ensuremath{\pm}",
    "·": r"\ensuremath{\cdot}", "‖": r"\ensuremath{\|}", "≈": r"\ensuremath{\approx}", "≥": r"\ensuremath{\geq}",
    "≤": r"\ensuremath{\leq}", "√": r"\ensuremath{\surd}", "ρ": r"\ensuremath{\rho}", "κ": r"\ensuremath{\kappa}",
    "μ": r"\ensuremath{\mu}", "η": r"\ensuremath{\eta}", "λ": r"\ensuremath{\lambda}",
    "⁻": r"\textsuperscript{\ensuremath{-}}", "¹": r"\textsuperscript{1}", "²": r"\textsuperscript{2}",
    "³": r"\textsuperscript{3}", "⁴": r"\textsuperscript{4}", "⁵": r"\textsuperscript{5}", "⁶": r"\textsuperscript{6}",
    "⁷": r"\textsuperscript{7}", "⁸": r"\textsuperscript{8}", "⁹": r"\textsuperscript{9}", "⁰": r"\textsuperscript{0}",
}
NATIVE = set("—…çíöÉéáóúñü")  # handled by inputenc/T1

HEADER = r"""
\usepackage{newunicodechar}
%UNICODE%
\setcounter{secnumdepth}{0}
\usepackage{caption}
\captionsetup[figure]{labelformat=empty}
\usepackage[margin=2.5cm]{geometry}
\usepackage{microtype}
\usepackage{xurl}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\renewcommand{\arraystretch}{1.15}
\setlength{\emergencystretch}{3em}
"""


def main() -> None:
    md = SRC.read_text()
    title = re.match(r"# (.+)\n", md).group(1).strip()
    author = re.search(r"\*\*(Javier[^*]+)\*\*\n(.+)\n", md)
    affil = author.group(2).strip()
    abs_m = re.search(r"## Abstract\n\n(.+?)\n\n\*\*Keywords:\*\*(.+?)\n", md, re.S)
    abstract, keywords = abs_m.group(1).strip(), abs_m.group(2).strip()
    body = md[abs_m.end():].lstrip()
    if body.startswith("---"):
        body = body[3:].lstrip()
    extra = sorted({c for c in md if ord(c) > 127} - set(UNI) - NATIVE)
    if extra:
        sys.exit(f"unmapped characters: {extra}")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "docs/paper/figures", OUT / "figures", dirs_exist_ok=True)
    header = HEADER.replace("%UNICODE%", "\n".join(rf"\newunicodechar{{{k}}}{{{v}}}" for k, v in UNI.items()))
    (OUT / "header.tex").write_text(header)
    meta = (OUT / "meta.yaml")
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    meta.write_text(f'title: "{esc(title)}"\nauthor: |\n  {author.group(1).strip()}\\\n  {affil}\n'
                    f'date: "November 2026"\nabstract: |\n'
                    + "".join(f"  {ln}\n" for ln in (abstract + f"\n\n**Keywords:** {keywords}").splitlines()))
    (OUT / "body.md").write_text(body)
    subprocess.run(["pandoc", str(OUT / "body.md"), "--metadata-file", str(meta), "-s", "-o", str(OUT / "main.tex"),
                    "--from", "markdown+autolink_bare_uris", "--lua-filter", str(ROOT / "scripts/paper_code_path.lua"), "--resource-path", str(OUT), "-H", str(OUT / "header.tex"),
                    "-V", "documentclass=article", "-V", "fontsize=10pt", "-V", "colorlinks=true",
                    "-V", "linkcolor=blue", "-V", "urlcolor=blue", "--columns=40"], check=True)
    (OUT / "body.md").unlink()
    meta.unlink()
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=OUT,
                       capture_output=True, text=True)
    log = (OUT / "main.log").read_text(errors="ignore") if (OUT / "main.log").exists() else r.stdout
    over = [ln for ln in log.splitlines() if ln.startswith("Overfull \\hbox")]
    print(f"latexmk exit {r.returncode}; overfull hboxes {len(over)}")
    for ln in over[:15]:
        print("  ", ln)
    if r.returncode:
        print(r.stdout[-3000:])
        sys.exit(1)
    subprocess.run(["tar", "czf", "arxiv-source.tar.gz", "main.tex", "figures"], cwd=OUT, check=True)
    print(f"arXiv source: {OUT / 'arxiv-source.tar.gz'}; PDF: {OUT / 'main.pdf'}")


if __name__ == "__main__":
    main()
