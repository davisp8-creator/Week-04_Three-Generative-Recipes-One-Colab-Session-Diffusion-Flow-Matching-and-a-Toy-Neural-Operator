# Running the notebook on a Colab T4 GPU

This produces the graded results, tagged `t4gpu`, and at the end offers to write a **VM-CPU vs Colab-T4
comparison** into `README.md` and push it to GitHub with a fine-grained token. The VM's CPU results
(`results/*_cpu.*`) are never overwritten; they become the left-hand column of the comparison.

## 1. One-time: create a fine-grained token

GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token:

- **Repository access:** *Only select repositories* → this Week-04 repo
- **Permissions:** Repository permissions → **Contents: Read and write** (nothing else)
- **Expiration:** short (7–30 days); revoke it when the assignment is done

Optional, recommended: in Colab open the 🔑 **Secrets** panel, add a secret named `GITHUB_TOKEN` with the token
as its value, and turn on *Notebook access*. The notebook then uses it automatically and never asks you to paste it.
Without the secret, you get a hidden prompt at the end (the token isn't echoed or saved).

## 2. Runtime and clone

**Runtime → Change runtime type → T4 GPU → Save.** Then, in the notebook's first cell (or a scratch cell above it):

```python
!git clone https://github.com/davisp8-creator/Week-04_Three-Generative-Recipes-One-Colab-Session-Diffusion-Flow-Matching-and-a-Toy-Neural-Operator.git
%cd Week-04_Three-Generative-Recipes-One-Colab-Session-Diffusion-Flow-Matching-and-a-Toy-Neural-Operator
```

Results land in the clone's `results/` folder, which is what makes the end-of-run push possible. If you skip the
clone, everything still runs, Step 1 warns you, and Step 25 can only offer the zip download.

## 3. Run all

**Runtime → Run all.** Step 1 should print:

```text
[setup] platform=colab  python=3.12.x  torch=2.x.x+cu12x  device=cuda  gpu=Tesla T4 (14.7 GB, sm_75)  results tag='t4gpu'
```

Check for **`gpu=Tesla T4`** and **`results tag='t4gpu'`**. If you see:

| Message | Meaning | Fix |
|---|---|---|
| `Colab is on a CPU-only runtime` | Step 2 was skipped | Change runtime type to T4 GPU, then Run all again |
| `nvidia-smi sees a GPU ... but torch cannot use it` | A CPU torch was installed over Colab's CUDA build | Runtime → Disconnect and delete runtime, then run again without installing anything |
| `GPU is <name>, not a T4` | Colab assigned a different GPU | Fine; results are tagged `gpu` so they aren't mistaken for T4 numbers |
| `not running inside a git clone` | Step 2's `%cd` didn't happen | Clone and `%cd` (step 2), then Run all |

Don't `pip install -r requirements.txt` on Colab. It pins the VM's CPU torch and would hide the T4 from torch.

Expect about 10–15 minutes in total.

## 4. The end-of-run prompts (Step 25)

```text
Write these results into README.md (VM CPU vs Colab T4 comparison)? [y/N] y
[publish] README.md updated (results section only)
Commit and push README.md + results/*_t4gpu* to GitHub? [y/N] y
[publish] GitHub token (hidden; blank = use existing git credentials): ••••••••
Your name for the commit: ...          ← only asked if git has no identity configured
Your email for the commit: ...
[publish] pulling origin/main ...
[publish] pushing to origin/main ...
[publish] done: README.md and 4 result path(s) pushed to origin/main
```

What the push does, in order:

1. Commits `results/failure_log_t4gpu.md`, `results_t4gpu.json`, `manifest_t4gpu.json` and `figs_t4gpu/`.
2. Pulls `origin/main` with a rebase. If the VM pushed a newer CPU run after you cloned, it comes in here. If an
   older T4 run conflicts, this run's results win.
3. Re-renders the README results section from **all** results now in the clone (VM CPU + this T4 run) and commits it.
4. Pushes.

The token is sent per command as an HTTP auth header. It is never printed, never written to `.git/config`, never
added to the remote URL, and never saved in the notebook. Answer **N** to either question to skip; nothing is
pushed without your second yes.

Only the section between `<!-- RESULTS:BEGIN -->` and `<!-- RESULTS:END -->` in README.md is rewritten.

## 5. If you skipped the push

- **Later from Colab** (same session): `!python scripts/update_readme.py --push`
- **From the VM:** download the zip from Step 24, unzip it into `results/`, then
  `python scripts/update_readme.py --push --tag t4gpu`. It uses the VM's normal git credentials if you leave the
  token prompt blank.
- **Re-render only, no git:** `python scripts/update_readme.py`

## What the comparison shows

| Table | Contents |
|---|---|
| Hardware | Device (CPU model and cores vs T4 + VRAM + compute capability), platform, torch build, total runtime |
| Wall-clock and memory | Each training stage, Burgers data generation, sampling latency for DDPM-1000 / DDIM / flow matching, FD solver vs FNO inference, peak VRAM, plus a **T4 speedup** column (CPU time ÷ GPU time) |
| Quality | Precision/recall, network calls to target, FNO rel-L2. These should agree across hardware, so a big gap is a bug to investigate, not a hardware effect |

The 2D generative models are tiny, so expect modest T4 speedups on their training. Expect the big gains on
DDPM's 1000-step sampling and FNO training.
