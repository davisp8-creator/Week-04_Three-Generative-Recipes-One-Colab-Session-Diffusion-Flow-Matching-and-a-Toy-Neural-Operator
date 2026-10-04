"""Build the Colab notebook and the simplified script from the full assignment script.

The full script follows the course layout: a SCRIPT DETAILS region, then one banner per
"Step N: Title" whose comment lines are the step's explanation and whose docstring lists
its references. This turns each step into a markdown cell (title, explanation,
references) plus a code cell, and writes a simplified script with the banners,
explanations and references stripped.

The Step 1 block between the INLINE-ENV-SETUP markers is replaced by env_setup.py's
source in the notebook, so the notebook runs on Colab with nothing else uploaded.

Usage:  python scripts/build_notebook.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "Assignment_04_Three_Generative_Recipes.py"
SIMPLIFIED = ROOT / "scripts" / "Assignment_04_Three_Generative_Recipes_Simplified.py"
ENV_SETUP = ROOT / "env_setup.py"
NOTEBOOK = ROOT / "Assignment_04_Three_Generative_Recipes_Colab.ipynb"
RULE = re.compile(r"^# =+$")
STEP = re.compile(r"^# Step (\d+): (.+)$")


def uncomment(lines):
    return [re.sub(r"^# ?", "", ln) for ln in lines]


def parse(text):
    """Split the script into (header_lines, header_refs, [step dicts])."""
    lines = text.split("\n")
    starts = [i for i, ln in enumerate(lines) if STEP.match(ln)]
    header = lines[: starts[0]]
    steps = []
    for k, s in enumerate(starts):
        end = starts[k + 1] - 1 if k + 1 < len(starts) else len(lines)
        num, title = STEP.match(lines[s]).groups()
        i = s + 1
        desc = []
        while not RULE.match(lines[i]):
            desc.append(lines[i])
            i += 1
        i += 1
        while i < end and not lines[i].strip():
            i += 1
        refs = []
        if i < end and lines[i] == '"""':
            i += 1
            while lines[i] != '"""':
                refs.append(lines[i].strip())
                i += 1
            i += 1
        body = lines[i:end]
        while body and (not body[-1].strip() or RULE.match(body[-1]) or body[-1] == "# endregion"):
            body.pop()
        steps.append(dict(num=int(num), title=title, desc=uncomment(desc),
                          refs=[r for r in refs if r], code="\n".join(body).strip("\n")))
    return header, steps


def header_markdown(header):
    """Course header + overview + APA references -> the notebook's title cell."""
    comments, refs, in_doc = [], [], False
    for ln in header:
        if ln == '"""':
            in_doc = not in_doc
        elif in_doc:
            refs.append(ln)
        elif ln.startswith("#") and not RULE.match(ln) and "region SCRIPT DETAILS" not in ln:
            comments.append(ln)
    body = uncomment(comments)
    course, week, objective = body[0], body[1], body[2]
    entries = [" ".join(e.split()) for e in "\n".join(refs).split("\n\n") if e.strip()]
    return ([f"# {week}", f"*{course}*", "", f"**{objective}**"] + body[3:]
            + ["", "**References**", ""] + [f"- {e}" for e in entries])


def inline_env_setup(code):
    src = ENV_SETUP.read_text(encoding="utf-8")
    src = src[: src.index('\nif __name__ == "__main__":')]
    src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)  # drop module docstring
    inlined = ("# ── env_setup.py (inlined by scripts/build_notebook.py; edit the .py, then rebuild) ──\n"
               + src.strip() + "\n# ── end env_setup.py ──\n\nROOT = Path.cwd()")
    return re.sub(r"# Local runs import env_setup.*?# INLINE-ENV-SETUP-END", lambda _: inlined,
                  code, flags=re.S)


def cell(kind, lines):
    src = [ln + "\n" for ln in lines]
    if src:
        src[-1] = src[-1].rstrip("\n")
    c = dict(cell_type=kind, metadata={}, source=src)
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build_notebook(header, steps):
    cells = [cell("markdown", header_markdown(header))]
    for st in steps:
        md = [f"## Step {st['num']}: {st['title']}", ""] + st["desc"]
        if st["refs"]:
            md += ["", "References: " + " · ".join(f"[{n}]({u})" for n, u in enumerate(st["refs"], 1))]
        cells.append(cell("markdown", md))
        if st["code"]:
            code = inline_env_setup(st["code"]) if "INLINE-ENV-SETUP-BEGIN" in st["code"] else st["code"]
            cells.append(cell("code", code.split("\n")))
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
              metadata=dict(accelerator="GPU", colab=dict(gpuType="T4", provenance=[]),
                            kernelspec=dict(display_name="Python 3", name="python3"),
                            language_info=dict(name="python")))
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(cells)


def build_simplified(steps):
    parts = []
    for st in steps:
        if not st["code"]:
            continue
        code = re.sub(r"# Local runs import env_setup.*?\n# INLINE-ENV-SETUP-BEGIN\n", "", st["code"], flags=re.S)
        code = re.sub(r"\n*# INLINE-ENV-SETUP-END\n", "\n", code)
        parts.append(f"# Step {st['num']}: {st['title']}\n{code}")
    SIMPLIFIED.write_text("\n\n\n".join(parts) + "\n", encoding="utf-8")


def build():
    header, steps = parse(SRC.read_text(encoding="utf-8"))
    n = build_notebook(header, steps)
    build_simplified(steps)
    print(f"{len(steps)} steps -> {NOTEBOOK.name} ({n} cells), {SIMPLIFIED.name}")


if __name__ == "__main__":
    build()
