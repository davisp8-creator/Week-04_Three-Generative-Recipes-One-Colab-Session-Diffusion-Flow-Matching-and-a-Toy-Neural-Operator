# =============================================================================
# region SCRIPT DETAILS
# =============================================================================
# CSP CST 627 100  Deep Learning Neural Networks
# Week 04 Assignment: Three Generative Recipes, One Colab Session
# Objective: Write the VM-CPU vs Colab-T4 comparison into README.md and,
# optionally, push it to GitHub with a fine-grained personal access token.
#
# Usage:
#   python scripts/update_readme.py            # re-render the README results section
#   python scripts/update_readme.py --push     # also commit + push (prompts for a token)
#   python scripts/update_readme.py --push --tag t4gpu
# The last step of the assignment notebook calls offer_publish(), which asks first.
# =============================================================================
# endregion
# =============================================================================
# Step 1: Render the Results Section
# Reads results/results_<tag>.json + manifest_<tag>.json for every tag present
# (cpu = the VM, t4gpu = Colab T4, gpu = any other GPU) and renders side-by-side
# hardware, wall-clock/VRAM and quality tables between the README markers.
# =============================================================================
"""
https://docs.python.org/3/library/json.html
https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/organizing-information-with-tables
"""

import argparse
import base64
import getpass
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
TAG_ORDER = ["cpu", "t4gpu", "gpu"]
TAG_LABEL = {"cpu": "VM · CPU only", "t4gpu": "Colab · T4 GPU", "gpu": "Other GPU"}


def load_runs(root):
    runs = {}
    for tag in TAG_ORDER:
        r, m = (
            root / "results" / f"results_{tag}.json",
            root / "results" / f"manifest_{tag}.json",
        )
        if r.exists() and m.exists():
            runs[tag] = (
                json.loads(r.read_text(encoding="utf-8")),
                json.loads(m.read_text(encoding="utf-8")),
            )
    return runs


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def fmt(x, kind):
    x = _num(x)
    if x is None:
        return "n/a"
    if kind == "ms" and x >= 10:
        kind = "s"
    return {
        "s": f"{x:,.1f} s",
        "ms": f"{x * 1e3:,.1f} ms",
        "mb": f"{x:,.0f} MB",
        "p": f"{x:.3f}",
        "l2": f"{x:.4f}",
        "int": f"{int(x)}",
    }[kind]


def device_label(m):
    h = m["hardware"]
    if h.get("gpu_name"):
        return f"{h['gpu_name']} ({h['vram_gb']} GB, sm_{str(h['compute_capability']).replace('.', '')})"
    return f"CPU only: {h['cpu']}, {h['cpu_count']} cores"


def render_section(root=ROOT):
    runs = load_runs(root)
    if not runs:
        return f"{BEGIN}\n_No full-run results in `results/` yet._\n{END}"
    tags = list(runs)
    has_cmp = "cpu" in runs and any(t != "cpu" for t in runs)
    gpu_tag = next((t for t in tags if t != "cpu"), None)
    head = (
        "| | "
        + " | ".join(TAG_LABEL[t] for t in tags)
        + (" | T4 speedup |" if has_cmp else " |")
    )
    sep = "|---|" + "---:|" * (len(tags) + has_cmp)

    def row(label, getter, kind, speedup=False):
        vals = [getter(*runs[t]) for t in tags]
        cells = [fmt(v, kind) for v in vals]
        extra = []
        if has_cmp:
            c, g = _num(getter(*runs["cpu"])), _num(getter(*runs[gpu_tag]))
            extra = [f"{c / g:.1f}x" if speedup and c and g else ""]
        return "| " + " | ".join([label] + cells + extra) + " |"

    hw = [
        head,
        sep,
        "| Device | "
        + " | ".join(device_label(runs[t][1]) for t in tags)
        + (" | |" if has_cmp else " |"),
        "| Platform / Python | "
        + " | ".join(
            f"{runs[t][1]['hardware']['platform']} / {runs[t][1]['hardware']['python']}"
            for t in tags
        )
        + (" | |" if has_cmp else " |"),
        "| torch | "
        + " | ".join(f"`{runs[t][1]['hardware']['torch']}`" for t in tags)
        + (" | |" if has_cmp else " |"),
        row("**Total notebook runtime**", lambda r, m: m["total_runtime_s"], "s", True),
    ]
    clock = [
        head,
        sep,
        row(
            "Flow-matching training (20k steps, incl. checkpoints)",
            lambda r, m: r["fm_train"]["seconds"],
            "s",
            True,
        ),
        row(
            "DDPM training (20k steps, incl. checkpoints)",
            lambda r, m: r["ddpm_train"]["seconds"],
            "s",
            True,
        ),
        row(
            "FNO training (150 epochs)",
            lambda r, m: r["fno_train"]["seconds"],
            "s",
            True,
        ),
        row(
            "Burgers data generation (1,200 solves)",
            lambda r, m: r["datagen_s"],
            "s",
            True,
        ),
        row(
            "DDPM 1000-step sampling, 10k samples",
            lambda r, m: r["ddpm1000"]["latency_s"],
            "ms",
            True,
        ),
        row(
            "DDIM sampling at target NFE, 10k samples",
            lambda r, m: r["ddim_min"][1],
            "ms",
            True,
        ),
        row(
            "Flow-matching sampling at target NFE, 10k samples",
            lambda r, m: r["fm_min"][1],
            "ms",
            True,
        ),
        row("FD solver, 200 cases @512", lambda r, m: r["fno"]["solver_s"], "ms", True),
        row(
            "FNO inference, 200 cases @512",
            lambda r, m: r["fno"]["infer_s"],
            "ms",
            True,
        ),
        row(
            "Peak VRAM: flow-matching training",
            lambda r, m: r["fm_train"]["peak_mb"],
            "mb",
        ),
        row("Peak VRAM: DDPM training", lambda r, m: r["ddpm_train"]["peak_mb"], "mb"),
        row("Peak VRAM: FNO training", lambda r, m: r["fno_train"]["peak_mb"], "mb"),
    ]
    head_q = (
        "| | " + " | ".join(TAG_LABEL[t] for t in tags) + (" | |" if has_cmp else " |")
    )
    quality = [
        head_q,
        sep,
        row(
            "Flow matching: best precision (real data 0.950)",
            lambda r, m: r["fm_best"],
            "p",
        ),
        row("DDIM: best precision", lambda r, m: r["ddim_best"], "p"),
        row("DDPM 1000-step: precision", lambda r, m: r["ddpm1000"]["precision"], "p"),
        row("DDPM 1000-step: recall", lambda r, m: r["ddpm1000"]["recall"], "p"),
        row(
            "Network calls to target: flow matching", lambda r, m: r["fm_min"][0], "int"
        ),
        row("Network calls to target: DDIM", lambda r, m: r["ddim_min"][0], "int"),
        row(
            "FNO rel-L2 @128 (training grid)",
            lambda r, m: r["fno"]["rel_l2"]["128"],
            "l2",
        ),
        row(
            "FNO rel-L2 @512 (zero-shot)", lambda r, m: r["fno"]["rel_l2"]["512"], "l2"
        ),
        row(
            "FNO with coordinate bug (F3) @512",
            lambda r, m: r["fno"]["rel_l2_bug"]["512"],
            "l2",
        ),
        row(
            "Heat-only (linear physics) baseline",
            lambda r, m: r["fno"]["heat"]["512"],
            "l2",
        ),
    ]
    fig_tag = gpu_tag or "cpu"
    logs = " · ".join(f"[{TAG_LABEL[t]}](results/failure_log_{t}.md)" for t in tags)
    note = (
        ""
        if has_cmp
        else "\n> Only the VM CPU run is in `results/` so far. The Colab T4 column and the speedup column "
        "appear after the T4 run (see COLAB.md).\n"
    )
    return "\n".join(
        [
            BEGIN,
            "<!-- Generated by scripts/update_readme.py from results/*.json; edits inside these markers are overwritten. -->",
            "## Results: VM CPU vs Colab T4",
            note,
            "Same code, same seeds, same budgets on every machine. **Quality numbers should match across hardware** "
            "(small drift comes from GPU vs CPU floating-point order); **wall-clock and VRAM are what the hardware changes.** "
            "Speedup = CPU time ÷ GPU time.",
            "",
            "### Hardware",
            "",
            *hw,
            "",
            "### Wall-clock and memory (lower is better)",
            "",
            *clock,
            "",
            "### Quality (should agree across hardware)",
            "",
            *quality,
            "",
            f"Target = precision and recall within 0.05 of the 1000-step DDPM. Failure-and-fix logs: {logs}.",
            "",
            f"![Samples vs NFE](results/figs_{fig_tag}/A5_samples_grid.png)",
            "",
            f"![Quality vs NFE](results/figs_{fig_tag}/A5_quality_vs_nfe.png)",
            "",
            "<details>",
            f"<summary>Failure figures F1–F4 ({TAG_LABEL[fig_tag]})</summary>",
            "",
            *[
                f"![{n}](results/figs_{fig_tag}/{n}.png)\n"
                for n in (
                    "F1_exploding_lr",
                    "F2_noise_schedule",
                    "F3_resolution",
                    "F4_metric_blind_spot",
                )
            ],
            "</details>",
            END,
        ]
    )


def write_readme(root=ROOT):
    path = root / "README.md"
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(
            f"README.md has no {BEGIN} ... {END} markers; add them where the results should go."
        )
    new = re.sub(
        re.escape(BEGIN) + r".*?" + re.escape(END),
        lambda _: render_section(root),
        text,
        flags=re.S,
    )
    changed = new != text
    if changed:
        path.write_text(new, encoding="utf-8")
    return changed


# =============================================================================
# Step 2: Commit and Push with a Fine-Grained Token
# The token comes from the Colab secret GITHUB_TOKEN if it exists, otherwise from a
# hidden getpass prompt (blank = use this machine's existing git credentials). It is
# sent per command as an HTTP auth header: never printed, never written to
# .git/config, never put in the remote URL. Results commit first, then a rebase pull
# (our results win on conflict), then the README is re-rendered so the comparison
# includes the latest results from GitHub (e.g. a newer VM CPU run), then push.
# =============================================================================
"""
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token
https://git-scm.com/docs/git-config#Documentation/git-config.txt-httpextraHeader
https://git-scm.com/docs/git-pull#Documentation/git-pull.txt---rebase
https://docs.python.org/3/library/getpass.html
"""


class GitError(RuntimeError):
    pass


def _git(root, *args, token=None, check=True):
    cmd = ["git"]
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += [
            "-c",
            f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}",
        ]
    out = subprocess.run(cmd + list(args), cwd=root, capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    if token:
        text = text.replace(token, "***").replace(basic, "***")
    if check and out.returncode != 0:
        raise GitError(f"git {args[0]} failed:\n{text}")
    return out.returncode, text


def colab_secret_token():
    try:
        from google.colab import userdata

        return userdata.get("GITHUB_TOKEN") or None
    except Exception:  # not on Colab, secret missing, or notebook access not granted
        return None


def get_token(ask_secret=getpass.getpass):
    token = colab_secret_token()
    if token:
        print("[publish] using the Colab secret GITHUB_TOKEN")
        return token
    print(
        "[publish] Fine-grained token: this repo only, Repository permissions > Contents: Read and write."
    )
    token = ask_secret(
        "[publish] GitHub token (hidden; blank = use existing git credentials): "
    ).strip()
    return token or None


def ensure_identity(root, ask=input):
    for key, prompt in (
        ("user.name", "Your name for the commit: "),
        ("user.email", "Your email for the commit: "),
    ):
        if not _git(root, "config", key, check=False)[1]:
            _git(root, "config", key, ask(f"[publish] {prompt}").strip())


def require_clone(root):
    code, top = _git(root, "rev-parse", "--show-toplevel", check=False)
    if code != 0 or Path(top).resolve() != Path(root).resolve():
        raise GitError(
            f"{root} is not the top of a git clone. On Colab: !git clone <repo> then %cd into it (COLAB.md)."
        )


def push_results(root, tag, token=None, ask=input):
    require_clone(root)
    ensure_identity(root, ask)
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")[1]
    result_paths = [
        p
        for p in (
            f"results/failure_log_{tag}.md",
            f"results/results_{tag}.json",
            f"results/manifest_{tag}.json",
            f"results/figs_{tag}",
        )
        if (root / p).exists()
    ]
    _git(
        root, "checkout", "--", "README.md"
    )  # regenerated after the pull, so drop the local render
    _git(root, "add", "--", *result_paths)
    if _git(root, "diff", "--cached", "--quiet", check=False)[0]:
        _git(root, "commit", "-m", f"Add {TAG_LABEL.get(tag, tag)} results ({tag})")
    print(f"[publish] pulling origin/{branch} ...")
    code, text = _git(
        root,
        "pull",
        "--rebase",
        "-X",
        "theirs",
        "origin",
        branch,
        token=token,
        check=False,
    )
    if code != 0:
        _git(root, "rebase", "--abort", check=False)
        raise GitError(f"pull --rebase failed, nothing was pushed:\n{text}")
    write_readme(root)
    _git(root, "add", "README.md")
    if _git(root, "diff", "--cached", "--quiet", check=False)[0]:
        _git(
            root, "commit", "-m", "Update README results: VM CPU vs Colab T4 comparison"
        )
    print(f"[publish] pushing to origin/{branch} ...")
    _git(root, "push", "origin", f"HEAD:{branch}", token=token)
    print(
        f"[publish] done: README.md and {len(result_paths)} result path(s) pushed to origin/{branch}"
    )


# =============================================================================
# Step 3: Prompt at the End of a Run
# Called by the notebook's last step: asks before touching README.md and asks again
# before pushing. Non-interactive runs only print how to do it later.
# =============================================================================


def _yes(answer):
    return answer.strip().lower() in ("y", "yes")


def offer_publish(root, tag, interactive, ask=input, ask_secret=getpass.getpass):
    later = "python scripts/update_readme.py --push"
    if not interactive:
        print(
            f"[publish] non-interactive run: to write README.md and push later, run  {later}"
        )
        return
    try:
        if not _yes(
            ask(
                "Write these results into README.md (VM CPU vs Colab T4 comparison)? [y/N] "
            )
        ):
            print(f"[publish] skipped. Later: {later}")
            return
        changed = write_readme(root)
        print(
            f"[publish] README.md {'updated' if changed else 'already up to date'} (results section only)"
        )
        if not _yes(
            ask(f"Commit and push README.md + results/*_{tag}* to GitHub? [y/N] ")
        ):
            print("[publish] not pushed. README.md is updated locally only.")
            return
        require_clone(root)  # fail before asking for a token that could not be used
        push_results(root, tag, token=get_token(ask_secret), ask=ask)
    except (EOFError, KeyboardInterrupt):
        print(f"\n[publish] cancelled. Later: {later}")
    except GitError as err:
        print(f"[publish] {err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Render the README results section; optionally commit and push."
    )
    ap.add_argument(
        "--push",
        action="store_true",
        help="commit results + README and push (prompts for a token)",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="results tag to commit (default: t4gpu if present, else cpu)",
    )
    args = ap.parse_args()
    if args.push:
        runs = load_runs(ROOT)
        tag = args.tag or ("t4gpu" if "t4gpu" in runs else "cpu")
        try:
            require_clone(ROOT)
            push_results(ROOT, tag, token=get_token())
        except GitError as err:
            sys.exit(f"[publish] {err}")
    else:
        print(f"README.md {'updated' if write_readme(ROOT) else 'already up to date'}")
