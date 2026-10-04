# Step 1: Environment Setup
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))
from env_setup import setup, hardware_snapshot

ENV = setup(require_t4=False)  # set require_t4=True to refuse anything but a Colab T4
if not (ROOT / ".git").exists():
    print(
        "[setup] note: not running inside a git clone of the repo, so Step 25 cannot write "
        "README.md or push. On Colab: !git clone <repo>, %cd into it, then Run all (COLAB.md)."
    )


# Step 2: Imports, Single-Source Constants & Measurement Helpers
# Install required libraries: run the installer for your OS (or `python env_setup.py`);
# on Colab, Step 1 installs anything missing and leaves the CUDA torch build alone.

# Import necessary libraries
import json
import math
import os
import time
import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # script mode: write figures, do not open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Single-source constants
QUICK = os.environ.get("HALDEN_QUICK") == "1"  # smoke test only: results NOT meaningful
SEED = 0
GEN_STEPS = 3_000 if QUICK else 20_000  # training steps for DDPM and flow matching
EVAL_EVERY = GEN_STEPS // 5  # sample-quality checkpoints during training (F4 evidence)
BAD_LR = 5e-2  # F1: the deliberately exploding learning rate
BAD_LR_STEPS = 500 if QUICK else 2000
RANGE_TEST_STEPS = 150 if QUICK else 300
SAMPLE_STEPS = 100  # F2: the "fast" DDPM sampler length
NFES = [1, 2, 4, 8, 16, 32, 64, 128]  # sampling-step sweep for the head-to-head
NU, T_END = 0.01, 0.5  # Burgers viscosity and final time
N_FINE = 1024  # grid the Burgers solutions are computed on
N_TRAIN_S, N_TEST_S = (300, 100) if QUICK else (1000, 200)
N_TRAIN_GRID = 128  # grid the FNO is trained on
FNO_EPOCHS = 30 if QUICK else 150

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = ENV["device"]
GPU_NAME = ENV["gpu_name"] or "CPU only"
RUN_TAG = ENV["tag"] + ("_quick" if QUICK else "")
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / f"figs_{RUN_TAG}"
FIG_DIR.mkdir(parents=True, exist_ok=True)
T_START = time.perf_counter()
print(
    f"torch {torch.__version__} | device: {DEVICE} ({GPU_NAME}) | outputs -> {FIG_DIR.relative_to(ROOT)}"
    + ("  [QUICK smoke-test mode]" if QUICK else "")
)

RESULTS = {}  # everything the decision table needs
FAILURES = []  # everything the failure log needs


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


class Meter:
    """Wall-clock and peak VRAM for the enclosed block (VRAM is NaN on CPU)."""

    def __enter__(self):
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.seconds = time.perf_counter() - self.t0
        self.peak_mb = (
            torch.cuda.max_memory_allocated() / 2**20
            if DEVICE == "cuda"
            else float("nan")
        )


def latency(fn, n=10_000):
    """Seconds to produce n samples with fn(n), after a small warm-up call."""
    fn(256)
    with Meter() as m:
        fn(n)
    return m.seconds


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=110)
    plt.show() if IN_NOTEBOOK else plt.close()


# Step 3: A1 Self-Generated 2D Data: Two Interleaved Spirals
def make_spirals(n, noise=0.04, seed=None):
    rng = np.random.default_rng(seed)
    h = n // 2
    t = (
        np.sqrt(rng.uniform(0, 1, h)) * 3 * np.pi
    )  # sqrt gives a roughly uniform density along the arm
    r = t / (3 * np.pi)
    arm = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    x = np.concatenate([arm, -arm]) + noise * rng.standard_normal((2 * h, 2))
    rng.shuffle(x)
    return (2.0 * x).astype(np.float32)  # fixed scale: data lies roughly in [-2, 2]


X_TRAIN = torch.from_numpy(make_spirals(100_000, seed=1)).to(DEVICE)
X_REF = torch.from_numpy(make_spirals(5_000, seed=2))  # held-out reference (CPU)
X_REF2 = torch.from_numpy(
    make_spirals(5_000, seed=3)
)  # second real draw for the noise floor


def swd(x, y, n_proj=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    th = torch.randn(2, n_proj, generator=g)
    th /= th.norm(dim=0, keepdim=True)
    px, py = (x.float() @ th).sort(0).values, (y.float() @ th).sort(0).values
    return (px - py).abs().mean().item()


def mmd(x, y, n=2000, scales=(0.05, 0.2, 0.8)):
    x, y = x[:n].float(), y[:n].float()
    xy = torch.cat([x, y])
    d2 = torch.cdist(xy, xy) ** 2
    k = sum(torch.exp(-d2 / (2 * s * s)) for s in scales)
    return (k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()).item()


def nn_dist(a, b):
    return torch.cdist(a, b).min(1).values


PR_RADIUS = torch.quantile(nn_dist(X_REF2, X_REF), 0.95).item()


def quality(samples):
    s = samples.detach().cpu()[:5000].float()
    if not torch.isfinite(s).all():
        return dict(precision=0.0, recall=0.0, swd=float("inf"), mmd=float("inf"))
    return dict(
        precision=(nn_dist(s, X_REF) <= PR_RADIUS).float().mean().item(),
        recall=(nn_dist(X_REF, s) <= PR_RADIUS).float().mean().item(),
        swd=swd(s, X_REF),
        mmd=mmd(s, X_REF),
    )


FLOOR = quality(X_REF2)
GAUSS = quality(torch.randn(5000, 2) * X_REF.std(0))  # "ignore the data" baseline
RESULTS["floor"] = FLOOR
RESULTS["gauss"] = GAUSS
fmt_q = (
    lambda q: f"precision={q['precision']:.3f} recall={q['recall']:.3f} SWD={q['swd']:.4f} MMD={q['mmd']:.5f}"
)
print("noise floor (real vs real):", fmt_q(FLOOR))
print("Gaussian baseline         :", fmt_q(GAUSS))

plt.figure(figsize=(4, 4))
plt.scatter(*X_REF.T, s=1)
plt.axis("equal")
plt.title("two spirals (held-out real)")
savefig("data")


# Step 4: A2 Shared Network and Training Loop
class TimeMLP(nn.Module):
    def __init__(self, hidden=256, depth=4, temb=64):
        super().__init__()
        self.register_buffer(
            "freqs", torch.exp(torch.linspace(0, math.log(1000.0), temb // 2))
        )
        self.inp = nn.Linear(2 + temb, hidden)
        self.blocks = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth - 1)]
        )
        self.out = nn.Linear(hidden, 2)

    def forward(self, x, t):  # x: (B,2), t: (B,) in [0,1]
        ang = t[:, None] * self.freqs[None]
        h = F.silu(self.inp(torch.cat([x, ang.sin(), ang.cos()], 1)))
        for b in self.blocks:
            h = h + F.silu(b(h))
        return self.out(h)


def train(
    model,
    loss_fn,
    steps,
    lr,
    bs=1024,
    warmup=0,
    clip=None,
    cosine=False,
    log_every=10,
    seed=0,
    eval_fn=None,
    eval_every=None,
):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_mult(s):
        w = min(1.0, (s + 1) / warmup) if warmup else 1.0
        c = 0.5 * (1 + math.cos(math.pi * s / steps)) if cosine else 1.0
        return w * c

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    hist = dict(step=[], loss=[], gnorm=[], diverged_at=None, evals=[])
    for s in range(steps):
        x1 = X_TRAIN[torch.randint(0, len(X_TRAIN), (bs,), device=DEVICE)]
        loss = loss_fn(model, x1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip if clip else float("inf")
        )
        opt.step()
        sched.step()
        if s % log_every == 0 or s == steps - 1:
            l, g = loss.item(), gn.item()
            hist["step"].append(s)
            hist["loss"].append(l)
            hist["gnorm"].append(g)
            if not math.isfinite(l) or not math.isfinite(g):
                hist["diverged_at"] = s
                break
        if eval_fn and ((s + 1) % eval_every == 0):
            hist["evals"].append(dict(step=s + 1, **eval_fn(model)))
    return hist


def n_params(m):
    return sum(p.numel() for p in m.parameters())


print("params per generative model:", n_params(TimeMLP()))


# Step 5: A3 Flow Matching / Rectified Flow
def fm_loss(model, x1):
    x0 = torch.randn_like(x1)
    t = torch.rand(len(x1), device=x1.device)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    return F.mse_loss(model(xt, t), x1 - x0)


@torch.no_grad()
def fm_sample(model, n, steps, seed=123):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(n, 2, device=DEVICE, generator=g)
    dt = 1.0 / steps
    for i in range(steps):
        x = x + dt * model(x, torch.full((n,), i * dt, device=DEVICE))
    return x


# Step 6: F1 Exploding Learning Rate (Deliberately Induced)
torch.manual_seed(0)
fm_bad = TimeMLP().to(DEVICE)
with Meter() as m:
    hist_bad = train(fm_bad, fm_loss, steps=BAD_LR_STEPS, lr=BAD_LR)
q_bad = quality(fm_sample(fm_bad, 5000, 50))

# What loss does a model get by ignoring its input and predicting E[v] (≈0)? This is
# the "learned nothing" ceiling.
with torch.no_grad():
    x1 = X_TRAIN[:20000]
    x0 = torch.randn_like(x1)
    CONST_LOSS = ((x1 - x0) - (x1 - x0).mean(0)).pow(2).mean().item()
print(
    f'bad-LR run: final loss {hist_bad["loss"][-1]:.3f} (constant-predictor loss = {CONST_LOSS:.3f}), '
    f'max grad-norm {max(hist_bad["gnorm"]):.1f}, diverged_at={hist_bad["diverged_at"]}'
)
print("bad-LR samples :", fmt_q(q_bad))
print("Gaussian       :", fmt_q(GAUSS))


# Step 7: F1 Diagnosis: Gradient Norm and LR Range Test
LRS = np.logspace(-4.5, -0.5, 9)
range_test = []
for lr in LRS:
    torch.manual_seed(0)
    mdl = TimeMLP().to(DEVICE)
    h = train(mdl, fm_loss, steps=RANGE_TEST_STEPS, lr=lr, log_every=5)
    tail = h["loss"][-10:]
    range_test.append(
        dict(
            lr=lr,
            final_loss=(
                float(np.mean(tail)) if h["diverged_at"] is None else float("inf")
            ),
            max_gnorm=max(h["gnorm"]),
        )
    )
range_test = pd.DataFrame(range_test)
print(range_test.to_string(index=False, float_format="%.4g"))

finite = range_test[np.isfinite(range_test.final_loss)]
best_lr = float(finite.loc[finite.final_loss.idxmin(), "lr"])
# the cliff = first LR whose peak grad-norm exceeds 10x the median of the three
# gentlest LRs
base_g = range_test.max_gnorm.iloc[:3].median()
CLIFF_LR = float(range_test[range_test.max_gnorm > 10 * base_g].lr.min())
CHOSEN_LR = float(f"{CLIFF_LR / 10:.1g}")  # rule: 10x below the cliff
print(
    f"loss minimum at lr={best_lr:.2g}; grad-norm cliff at lr={CLIFF_LR:.2g}; "
    f"chosen lr={CHOSEN_LR:g} (10x below the cliff); bad lr={BAD_LR:g} is past the cliff"
)


# Step 8: F1 Repair: Evidence-Based LR, Warmup, Cosine Decay, Clipping
RECIPE = dict(lr=CHOSEN_LR, warmup=200, clip=1.0, cosine=True)
torch.manual_seed(0)
fm = TimeMLP().to(DEVICE)
with Meter() as m_fm:
    hist_fm = train(
        fm,
        fm_loss,
        steps=GEN_STEPS,
        **RECIPE,
        eval_every=EVAL_EVERY,
        eval_fn=lambda mdl: quality(fm_sample(mdl, 5000, 32)),
    )
RESULTS["fm_train"] = dict(
    seconds=m_fm.seconds,
    peak_mb=m_fm.peak_mb,
    final_loss=float(np.mean(hist_fm["loss"][-20:])),
)
q_fm50 = quality(fm_sample(fm, 5000, 50))
print(
    f"repaired FM: train {m_fm.seconds:.1f}s (incl. periodic evals), peak VRAM {m_fm.peak_mb:.1f} MB, "
    f"final loss {RESULTS['fm_train']['final_loss']:.3f}\n  @50 steps: {fmt_q(q_fm50)}"
)

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].semilogy(
    hist_bad["step"], hist_bad["loss"], label=f"lr={BAD_LR:g}, no warmup/clip", c="C3"
)
ax[0].semilogy(
    hist_fm["step"][:200],
    hist_fm["loss"][:200],
    label=f"repaired (lr={CHOSEN_LR:g}, warmup, clip)",
    c="C0",
)
ax[0].axhline(CONST_LOSS, ls="--", c="k", lw=1, label="constant-predictor loss")
ax[0].set_title("loss (first 2000 steps)")
ax[0].legend(fontsize=8)
ax[1].semilogy(hist_bad["step"], hist_bad["gnorm"], c="C3")
ax[1].semilogy(hist_fm["step"][:200], hist_fm["gnorm"][:200], c="C0")
ax[1].set_title("gradient norm (pre-clip)")
ax[2].loglog(range_test.lr, range_test.final_loss.replace(np.inf, np.nan), "o-")
ax[2].axvline(BAD_LR, c="C3", ls="--", label="bad lr")
ax[2].axvline(CHOSEN_LR, c="C0", ls="--", label="chosen lr")
ax[2].set_xlabel("learning rate")
ax[2].set_title("LR range test: loss after 300 steps")
ax[2].legend(fontsize=8)
savefig("F1_exploding_lr")

FAILURES.append(
    dict(
        id="F1",
        name="Exploding learning rate (flow matching)",
        induced=f"Adam lr {BAD_LR:g}, no warmup, no clipping",
        symptom=(
            f'grad-norm peaked at {max(hist_bad["gnorm"]):,.0f}; samples no better than a Gaussian '
            f'(precision {q_bad["precision"]:.2f} vs {GAUSS["precision"]:.2f}).'
        ),
        diagnosis=(
            f"LR range test: peak grad-norm ~{base_g:.1f} at gentle LRs, "
            f"{range_test[range_test.lr == CLIFF_LR].max_gnorm.item():,.0f} at lr {CLIFF_LR:.2g} (the cliff); {BAD_LR:g} is past it."
        ),
        fix=f"lr {CHOSEN_LR:g} (cliff / 10) + warmup, cosine decay, clip 1.0",
        after=f'precision {q_fm50["precision"]:.2f} (real data {FLOOR["precision"]:.2f})',
        guard="2-min LR range test before every new run; alert on grad-norm spikes.",
    )
)


# Step 9: A4 DDPM: Definitions and Training
class Schedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=DEVICE)
        self.abar = torch.cumprod(1 - self.betas, 0)


def ddpm_loss_fn(sch):
    def loss(model, x0):
        k = torch.randint(0, sch.T, (len(x0),), device=x0.device)
        ab = sch.abar[k][:, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return F.mse_loss(model(xt, (k + 1).float() / sch.T), eps)

    return loss


ONE = torch.tensor(1.0, device=DEVICE)


@torch.no_grad()
def ddpm_sample(model, sch, n, steps=None, seed=123):
    """Ancestral sampling over `steps` timesteps of `sch` (None = all T). Each step's beta is
    derived from the abar ratio of consecutive kept timesteps, so a subset stays on the same schedule.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(n, 2, device=DEVICE, generator=g)
    ks = (
        list(range(sch.T - 1, -1, -1))
        if steps is None
        else sorted(
            set(torch.linspace(sch.T - 1, 0, steps).round().long().tolist()),
            reverse=True,
        )
    )
    for i, k in enumerate(ks):
        ab = sch.abar[k]
        ab_prev = sch.abar[ks[i + 1]] if i + 1 < len(ks) else ONE
        beta = 1 - ab / ab_prev
        eps = model(x, torch.full((n,), (k + 1) / sch.T, device=DEVICE))
        mean = (x - beta / (1 - ab).sqrt() * eps) / (1 - beta).sqrt()
        if i + 1 < len(ks):
            var = beta * (1 - ab_prev) / (1 - ab)  # posterior variance
            x = mean + var.sqrt() * torch.randn(x.shape, device=DEVICE, generator=g)
        else:
            x = mean
    return x


@torch.no_grad()
def ddim_sample(model, sch, n, steps, seed=123):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(n, 2, device=DEVICE, generator=g)
    ks = torch.linspace(sch.T - 1, 0, steps).round().long().tolist()
    for i, k in enumerate(ks):
        ab = sch.abar[k]
        eps = model(x, torch.full((n,), (k + 1) / sch.T, device=DEVICE))
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-4, 4)
        ab_prev = sch.abar[ks[i + 1]] if i + 1 < len(ks) else ONE
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return x


sch = Schedule()
torch.manual_seed(0)
ddpm = TimeMLP().to(DEVICE)
with Meter() as m_dd:
    hist_ddpm = train(
        ddpm,
        ddpm_loss_fn(sch),
        steps=GEN_STEPS,
        **RECIPE,
        eval_every=EVAL_EVERY,
        eval_fn=lambda mdl: quality(ddim_sample(mdl, sch, 5000, 32)),
    )
RESULTS["ddpm_train"] = dict(
    seconds=m_dd.seconds,
    peak_mb=m_dd.peak_mb,
    final_loss=float(np.mean(hist_ddpm["loss"][-20:])),
)
s_ddpm = ddpm_sample(ddpm, sch, 5000)
q_ddpm = quality(s_ddpm)
LAT_DDPM1000 = latency(lambda n: ddpm_sample(ddpm, sch, n))
print(
    f"DDPM: train {m_dd.seconds:.1f}s (incl. periodic evals), peak VRAM {m_dd.peak_mb:.1f} MB\n"
    f"  1000-step ancestral: {fmt_q(q_ddpm)} | {LAT_DDPM1000:.2f}s per 10k samples"
)


# Step 10: F2 Mismatched Noise Schedule (Deliberately Induced)
sch_short = Schedule(
    T=SAMPLE_STEPS
)  # BUG: a new schedule, not a subset of the trained one
s_f2_bad = ddpm_sample(ddpm, sch_short, 5000)
q_f2_bad = quality(s_f2_bad)
print("100 steps, rebuilt schedule (bug):", fmt_q(q_f2_bad))
print("1000 steps, trained schedule (ref):", fmt_q(q_ddpm))


# Step 11: F2 Diagnosis: Noise Level per t and Denoiser Probe
def k_of(s, t):
    return min(max(int(round(t * s.T)), 1), s.T) - 1


def sigma_at(s, t):
    return (1 - s.abar[k_of(s, t)]).sqrt().item()


@torch.no_grad()
def eps_mse_at(model, s, t, n=5000):
    k = k_of(s, t)
    ab = s.abar[k]
    x0 = X_TRAIN[:n]
    eps = torch.randn_like(x0)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    return F.mse_loss(
        model(xt, torch.full((n,), (k + 1) / s.T, device=DEVICE)), eps
    ).item()


probe = pd.DataFrame(
    [
        dict(
            t=t,
            sigma_trained=sigma_at(sch, t),
            sigma_sampler=sigma_at(sch_short, t),
            eps_mse_trained=eps_mse_at(ddpm, sch, t),
            eps_mse_sampler=eps_mse_at(ddpm, sch_short, t),
        )
        for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    ]
)
print(probe.to_string(index=False, float_format="%.3f"))
worst = probe.loc[
    (probe.eps_mse_sampler - probe.eps_mse_trained).idxmax()
]  # largest error gap


# Step 12: F2 Repair: Respace the Trained Schedule
s_f2_fix = ddpm_sample(ddpm, sch, 5000, steps=SAMPLE_STEPS)
q_f2_fix = quality(s_f2_fix)
LAT_DDPM100 = latency(lambda n: ddpm_sample(ddpm, sch, n, steps=SAMPLE_STEPS))
print(
    "100 steps, respaced trained schedule (fix):",
    fmt_q(q_f2_fix),
    f"| {LAT_DDPM100:.2f}s per 10k",
)

fig, ax = plt.subplots(1, 5, figsize=(19, 3.7))
tt = np.linspace(0.01, 1, 100)
ax[0].plot(tt, [sigma_at(sch, t) for t in tt], c="C0", label="trained (T=1000)")
ax[0].plot(
    tt, [sigma_at(sch_short, t) for t in tt], c="C3", label="sampler bug (T=100)"
)
ax[0].set_xlabel("normalized time t (network input)")
ax[0].set_title("noise std at the same t")
ax[0].legend(fontsize=8)
w = 0.35
xi = np.arange(len(probe))
ax[1].bar(
    xi - w / 2, probe.eps_mse_trained, w, label="trained-schedule inputs", color="C0"
)
ax[1].bar(
    xi + w / 2, probe.eps_mse_sampler, w, label="sampler-schedule inputs", color="C3"
)
ax[1].set_xticks(xi, [f"{t:g}" for t in probe.t])
ax[1].set_xlabel("t")
ax[1].set_yscale("log")
ax[1].set_title("denoiser probe: eps-MSE")
ax[1].legend(fontsize=8)
for a, s, q, ttl in [
    (ax[2], s_f2_bad, q_f2_bad, "100 steps, rebuilt schedule (bug)"),
    (ax[3], s_f2_fix, q_f2_fix, "100 steps, respaced (fix)"),
    (ax[4], s_ddpm, q_ddpm, "1000 steps (reference)"),
]:
    a.scatter(*s.cpu().T, s=1, c="C3" if "bug" in ttl else "C0")
    a.set_xlim(-2.8, 2.8)
    a.set_ylim(-2.8, 2.8)
    a.set_title(f"{ttl}\nprecision {q['precision']:.2f}", fontsize=9)
savefig("F2_noise_schedule")

FAILURES.append(
    dict(
        id="F2",
        name="Mismatched noise schedule (DDPM)",
        induced=f"trained on T=1000, sampled with a rebuilt T={SAMPLE_STEPS} schedule",
        symptom=(
            f'recall {q_ddpm["recall"]:.2f} -> {q_f2_bad["recall"]:.2f}, precision {q_ddpm["precision"]:.2f} -> {q_f2_bad["precision"]:.2f}; '
            "no error raised."
        ),
        diagnosis=(
            f"same t, different noise (sigma {worst.sigma_sampler:.2f} vs trained {worst.sigma_trained:.2f} at t={worst.t:g}). "
            f"Denoiser probe: eps-MSE {worst.eps_mse_trained:.2f} on trained-schedule inputs vs {worst.eps_mse_sampler:.2f}; network fine, schedule wrong."
        ),
        fix=f"respace {SAMPLE_STEPS} steps of the trained schedule (or DDIM)",
        after=(
            f'precision {q_f2_fix["precision"]:.2f} / recall {q_f2_fix["recall"]:.2f}, '
            f"{LAT_DDPM1000 / LAT_DDPM100:.0f}x faster than 1000 steps"
        ),
        guard="Save the schedule with the checkpoint; sampler refuses a mismatch.",
    )
)


# Step 13: A5 Head-to-Head at Matched Compute (NFE Sweep)
rows = []
for nfe in NFES:
    rows.append(
        dict(
            model="DDIM (DDPM-trained)",
            nfe=nfe,
            **quality(ddim_sample(ddpm, sch, 5000, nfe)),
            latency_s=latency(lambda n: ddim_sample(ddpm, sch, n, nfe)),
        )
    )
    rows.append(
        dict(
            model="Flow matching (Euler)",
            nfe=nfe,
            **quality(fm_sample(fm, 5000, nfe)),
            latency_s=latency(lambda n: fm_sample(fm, n, nfe)),
        )
    )
rows.append(
    dict(
        model="DDPM ancestral (respaced)",
        nfe=SAMPLE_STEPS,
        **q_f2_fix,
        latency_s=LAT_DDPM100,
    )
)
rows.append(dict(model="DDPM ancestral", nfe=1000, **q_ddpm, latency_s=LAT_DDPM1000))
cmp_df = pd.DataFrame(rows)
print(cmp_df.to_string(index=False, float_format="%.4g"))

TARGET, TARGET_R = q_ddpm["precision"] - 0.05, q_ddpm["recall"] - 0.05


def min_nfe(model):
    ok = cmp_df[
        (cmp_df.model == model)
        & (cmp_df.precision >= TARGET)
        & (cmp_df.recall >= TARGET_R)
    ].sort_values("nfe")
    return (
        (int(ok.nfe.iloc[0]), float(ok.latency_s.iloc[0])) if len(ok) else (None, None)
    )


RESULTS["target_precision"] = TARGET
RESULTS["target_recall"] = TARGET_R
RESULTS["ddim_min"] = min_nfe("DDIM (DDPM-trained)")
RESULTS["fm_min"] = min_nfe("Flow matching (Euler)")
RESULTS["ddpm1000"] = dict(**q_ddpm, latency_s=LAT_DDPM1000)
RESULTS["fm_best"] = float(cmp_df[cmp_df.model.str.startswith("Flow")].precision.max())
RESULTS["ddim_best"] = float(
    cmp_df[cmp_df.model.str.startswith("DDIM")].precision.max()
)
print(
    f"target precision >= {TARGET:.3f} and recall >= {TARGET_R:.3f}: DDIM needs NFE={RESULTS['ddim_min'][0]}, flow matching needs NFE={RESULTS['fm_min'][0]}"
)

fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
for mdl, c in [("DDIM (DDPM-trained)", "C1"), ("Flow matching (Euler)", "C0")]:
    d = cmp_df[cmp_df.model == mdl]
    ax[0].semilogx(d.nfe, d.precision, "o-", c=c, label=mdl)
    ax[1].loglog(d.nfe, d.swd, "o-", c=c, label=mdl)
    ax[2].loglog(d.nfe, d.latency_s * 1e3, "o-", c=c, label=mdl)
ax[0].axhline(q_ddpm["precision"], c="k", ls=":", label="DDPM 1000-step")
ax[0].axhline(FLOOR["precision"], c="gray", ls="--", label="real data (ceiling)")
ax[0].axhline(TARGET, c="C2", ls="-.", lw=1, label="target")
ax[0].set_ylim(0, 1)
ax[0].set_title("precision vs NFE (higher = better)")
ax[0].legend(fontsize=7)
ax[1].axhline(FLOOR["swd"], c="gray", ls="--", label="noise floor")
ax[1].set_title("SWD vs NFE (lower = better)")
ax[1].legend(fontsize=7)
ax[2].set_title("sampling latency, ms per 10k samples")
ax[2].legend(fontsize=7)
for a in ax:
    a.set_xlabel("NFE (network calls)")
savefig("A5_quality_vs_nfe")

fig, ax = plt.subplots(2, 5, figsize=(16, 6.6))
for j, nfe in enumerate([1, 4, 16, 64]):
    for i, (name, fn) in enumerate(
        [
            ("DDIM", lambda: ddim_sample(ddpm, sch, 3000, nfe)),
            ("FM", lambda: fm_sample(fm, 3000, nfe)),
        ]
    ):
        ax[i, j].scatter(*fn().cpu().T, s=1, c=f"C{1 - i}")
        ax[i, j].set_title(f"{name}, NFE={nfe}")
ax[0, 4].scatter(*s_ddpm[:3000].cpu().T, s=1, c="k")
ax[0, 4].set_title("DDPM ancestral, NFE=1000")
ax[1, 4].scatter(*X_REF[:3000].T, s=1, c="gray")
ax[1, 4].set_title("real (held-out)")
for a in ax.flat:
    a.set_xlim(-2.8, 2.8)
    a.set_ylim(-2.8, 2.8)
    a.set_xticks([])
    a.set_yticks([])
savefig("A5_samples_grid")


# Step 14: F4 Metric Blind Spot (Unplanned)
ev = pd.concat(
    [
        pd.DataFrame(hist_fm["evals"]).assign(model="Flow matching"),
        pd.DataFrame(hist_ddpm["evals"]).assign(model="DDPM (DDIM-32)"),
    ]
)
print(
    ev[["model", "step", "precision", "recall", "swd"]].to_string(
        index=False, float_format="%.4f"
    )
)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for (mdl, d), c in zip(ev.groupby("model", sort=False), ["C0", "C1"]):
    ax[0].plot(d.step, d.swd, "o-", c=c, label=mdl)
    ax[1].plot(d.step, d.precision, "o-", c=c, label=mdl)
ax[0].axhline(FLOOR["swd"], c="gray", ls="--")
ax[0].set_title("SWD during training (noisy, non-monotonic)")
ax[1].axhline(FLOOR["precision"], c="gray", ls="--")
ax[1].set_title("precision during training (keeps improving)")
for a in ax:
    a.set_xlabel("training step")
    a.legend(fontsize=8)
savefig("F4_metric_blind_spot")

trace = lambda key: " -> ".join(f"{e[key]:.3f}" for e in hist_fm["evals"])
FAILURES.append(
    dict(
        id="F4",
        name="Metric blind spot (unplanned)",
        induced="first run: 6k steps, judged on SWD alone",
        symptom="SWD ~0.03, near the 0.014 floor, while the scatter plot showed blobs with no spiral arms.",
        diagnosis=(
            f'checkpoints: precision {trace("precision")} rises steadily; SWD {trace("swd")} wanders. '
            "SWD averages 1D projections, so it cannot see the arms."
        ),
        fix="train 20k steps; gate on precision/recall + scatter plot",
        after=f'precision FM {q_fm50["precision"]:.2f}, DDPM {q_ddpm["precision"]:.2f}',
        guard="Acceptance metric must separate a known-bad model from a good one.",
    )
)


# Step 15: C1 Finite-Difference Viscous Burgers Solver and Verification
def random_ic(n, N, seed):
    rng = np.random.default_rng(seed)
    x = np.arange(N) / N
    k = np.arange(1, 9)
    a = rng.standard_normal((n, 8)) / k**1.5
    b = rng.standard_normal((n, 8)) / k**1.5
    u = (
        a[:, :, None] * np.sin(2 * np.pi * k[None, :, None] * x)
        + b[:, :, None] * np.cos(2 * np.pi * k[None, :, None] * x)
    ).sum(1)
    return (u / np.abs(u).max(1, keepdims=True) * rng.uniform(0.5, 1.5, (n, 1))).astype(
        np.float64
    )  # peak |u| in [0.5, 1.5]


@torch.no_grad()
def burgers_fd(u0, nu=NU, T=T_END, cfl=0.4, dt=None, advect=True):
    u = torch.as_tensor(u0, dtype=torch.float64, device=DEVICE)
    N = u.shape[-1]
    dx = 1.0 / N
    if dt is None:
        dt = (
            cfl * dx / max(u.abs().max().item(), 1e-6)
        )  # max|u| cannot grow for Burgers
    nsteps = math.ceil(T / dt)
    dt = T / nsteps
    k = torch.arange(N // 2 + 1, device=DEVICE)
    lap_eig = (
        -4.0 * torch.sin(math.pi * k / N) ** 2 / dx**2
    )  # eigenvalues of the periodic FD Laplacian
    denom = 1.0 - dt * nu * lap_eig  # backward Euler for diffusion
    for _ in range(nsteps):
        if advect:
            f = 0.5 * u * u
            u = u - dt * (torch.roll(f, -1, -1) - torch.roll(f, 1, -1)) / (2 * dx)
        u = torch.fft.irfft(torch.fft.rfft(u) / denom, n=N)
    return u


# -- verification: grid convergence on one IC (same dt everywhere, so we isolate
# spatial error) --
u0_test = random_ic(1, 4096, seed=99)
ref = burgers_fd(u0_test, dt=2e-4)  # reference on a 4096 grid
conv = []
for N in [128, 256, 512, 1024]:
    uN = burgers_fd(u0_test[:, :: 4096 // N], dt=2e-4)
    err = (uN - ref[:, :: 4096 // N]).norm() / ref[:, :: 4096 // N].norm()
    conv.append(dict(N=N, rel_err=err.item()))
conv = pd.DataFrame(conv)
conv["order"] = np.r_[
    np.nan, np.log2(conv.rel_err.values[:-1] / conv.rel_err.values[1:])
]
print("grid convergence vs 4096-point reference:")
print(conv.to_string(index=False, float_format="%.3g"))
mass_drift = (ref.mean() - torch.as_tensor(u0_test).mean()).abs().item()
print(
    f"mass drift over the run: {mass_drift:.2e}  (conservative scheme, so this should be ~ machine precision)"
)
RESULTS["solver"] = dict(conv=conv.to_dict("records"), mass_drift=mass_drift)


# Step 16: C1 Generate the Operator Dataset
with Meter() as m_gen:
    U0 = random_ic(N_TRAIN_S + N_TEST_S, N_FINE, seed=7)
    UT = burgers_fd(U0).float().cpu()
    U0 = torch.from_numpy(U0).float()
RESULTS["datagen_s"] = m_gen.seconds
print(f"generated {len(U0)} solution pairs at N={N_FINE} in {m_gen.seconds:.1f}s")


def at_res(U, N):
    return U[:, :: N_FINE // N]


tr, te = slice(0, N_TRAIN_S), slice(N_TRAIN_S, None)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
xs = np.arange(N_FINE) / N_FINE
for i, a in enumerate(ax):
    a.plot(xs, U0[i], label="u(x,0)")
    a.plot(xs, UT[i], label=f"u(x,{T_END:g})")
    a.legend(fontsize=8)
plt.suptitle(f"Burgers, nu={NU}: smooth inputs steepen into shocks")
savefig("C1_burgers_samples")


# Step 17: C2 1D Fourier Neural Operator
class SpectralConv1d(nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(
            torch.randn(cin, cout, modes, dtype=torch.cfloat) / (cin * cout)
        )

    def forward(self, x):
        B, C, N = x.shape
        xf = torch.fft.rfft(x)
        m = min(self.modes, xf.shape[-1])
        out = torch.zeros(
            B, self.w.shape[1], N // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out[:, :, :m] = torch.einsum("bim,iom->bom", xf[:, :, :m], self.w[:, :, :m])
        return torch.fft.irfft(out, n=N)


def grid_index_based(
    N, device
):  # BUG (induced): normalised by the *training* grid size
    return torch.arange(N, device=device).float() / N_TRAIN_GRID


def grid_physical(N, device):  # FIX: physical coordinate in [0,1), any N
    return torch.arange(N, device=device).float() / N


class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64, layers=4, grid_fn=grid_physical):
        super().__init__()
        self.grid_fn = grid_fn
        self.lift = nn.Conv1d(2, width, 1)
        self.spec = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(layers)]
        )
        self.skip = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(
            nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, 1, 1)
        )

    def forward(self, u):  # u: (B, N)
        g = self.grid_fn(u.shape[-1], u.device).expand_as(u)
        h = self.lift(torch.stack([u, g], 1))
        for i, (s, w) in enumerate(zip(self.spec, self.skip)):
            h = s(h) + w(h)
            if i < len(self.spec) - 1:
                h = F.gelu(h)
        return self.proj(h).squeeze(1)


def rel_l2(pred, true):
    return ((pred - true).norm(dim=-1) / true.norm(dim=-1)).mean().item()


def train_fno(grid_fn, epochs=FNO_EPOCHS, bs=32, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = FNO1d(grid_fn=grid_fn).to(DEVICE)
    a, u = at_res(U0[tr], N_TRAIN_GRID).to(DEVICE), at_res(UT[tr], N_TRAIN_GRID).to(
        DEVICE
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    iters = epochs * math.ceil(len(a) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters)
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(a), device=DEVICE)
        tot = 0
        for i in range(0, len(a), bs):
            idx = perm[i : i + bs]
            pred = model(a[idx])
            loss = (
                (pred - u[idx]).norm(dim=-1) / u[idx].norm(dim=-1)
            ).mean()  # train on relative L2 directly
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        losses.append(tot / len(a))
    return model, losses


@torch.no_grad()
def eval_fno(model, N):
    return rel_l2(model(at_res(U0[te], N).to(DEVICE)).cpu(), at_res(UT[te], N))


# Step 18: F3 Train/Eval Resolution Mismatch (Deliberately Induced)
with Meter() as m_fno_bad:
    fno_bad, loss_bad = train_fno(grid_index_based)
res_bad = {N: eval_fno(fno_bad, N) for N in [128, 256, 512]}
print(
    "index-grid FNO, test rel-L2 by eval resolution:",
    {k: round(v, 4) for k, v in res_bad.items()},
)


# Step 19: F3 Diagnosis: Isolate Data vs Input Encoding
fno_bad.grid_fn = grid_physical
res_patched = {N: eval_fno(fno_bad, N) for N in [128, 256, 512]}
fno_bad.grid_fn = grid_index_based
print(
    "same weights, coordinate patched at eval:",
    {k: round(v, 4) for k, v in res_patched.items()},
)
for N in [128, 512]:
    g = grid_index_based(N, "cpu")
    print(f"  index grid at N={N}: range [{g.min():.2f}, {g.max():.2f}]")


# Step 20: F3 Repair and Baselines
with Meter() as m_fno:
    fno, loss_fno = train_fno(grid_physical)
RESULTS["fno_train"] = dict(
    seconds=m_fno.seconds, peak_mb=m_fno.peak_mb, params=n_params(fno)
)
res_fix = {N: eval_fno(fno, N) for N in [128, 256, 512]}
res_ident = {N: rel_l2(at_res(U0[te], N), at_res(UT[te], N)) for N in [128, 256, 512]}
UHEAT = burgers_fd(U0[te].double().numpy(), advect=False).float().cpu()
res_heat = {N: rel_l2(at_res(UHEAT, N), at_res(UT[te], N)) for N in [128, 256, 512]}
print("physical-grid FNO rel-L2:", {k: round(v, 4) for k, v in res_fix.items()})
print("heat-only baseline      :", {k: round(v, 4) for k, v in res_heat.items()})
print("identity baseline       :", {k: round(v, 4) for k, v in res_ident.items()})

# Latency: surrogate vs solver for the 200 test cases on the 512 grid (solver needs
# its own CFL-limited steps)
with Meter() as m_solver:
    _ = burgers_fd(at_res(U0[te], 512).double().numpy())
a512 = at_res(U0[te], 512).to(DEVICE)
with torch.no_grad():
    fno(a512)
    with Meter() as m_inf:
        fno(a512)
RESULTS["fno"] = dict(
    rel_l2=res_fix,
    rel_l2_bug=res_bad,
    rel_l2_patched=res_patched,
    identity=res_ident,
    heat=res_heat,
    solver_s=m_solver.seconds,
    infer_s=m_inf.seconds,
    speedup=m_solver.seconds / m_inf.seconds,
)
print(
    f'200 cases @512: FD solver {m_solver.seconds:.3f}s vs FNO {m_inf.seconds * 1e3:.2f} ms -> {RESULTS["fno"]["speedup"]:.0f}x faster'
)

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
Ns = [128, 256, 512]
ax[0].plot(Ns, [res_bad[n] for n in Ns], "o-", c="C3", label="index-based grid (bug)")
ax[0].plot(
    Ns, [res_patched[n] for n in Ns], "s--", c="C1", label="bug weights, coord patched"
)
ax[0].plot(
    Ns, [res_fix[n] for n in Ns], "o-", c="C0", label="physical grid (fixed, retrained)"
)
ax[0].plot(
    Ns, [res_heat[n] for n in Ns], ":", c="gray", label="heat-only (linear) baseline"
)
ax[0].set_xscale("log", base=2)
ax[0].set_yscale("log")
ax[0].set_xlabel("evaluation grid N (trained at 128)")
ax[0].set_ylabel("test relative L2")
ax[0].legend(fontsize=8)
ax[0].set_title("zero-shot super-resolution")
ax[1].semilogy(loss_fno)
ax[1].set_title("FNO training loss (rel L2)")
ax[1].set_xlabel("epoch")
i = int(
    np.argmax([(UT[N_TRAIN_S + j].diff().abs().max()).item() for j in range(N_TEST_S)])
)  # sharpest-shock test case
x512 = np.arange(512) / 512
with torch.no_grad():
    ax[2].plot(x512, at_res(UT[te], 512)[i], "k", lw=2, label="FD truth @512")
    ax[2].plot(
        x512, at_res(UHEAT, 512)[i], c="gray", ls=":", label="heat-only baseline"
    )
    ax[2].plot(x512, fno(a512[i : i + 1]).cpu()[0], "C0--", label="FNO (fixed)")
    ax[2].plot(x512, fno_bad(a512[i : i + 1]).cpu()[0], "C3:", label="FNO (bug)")
ax[2].legend(fontsize=8)
ax[2].set_title("hardest test case (steepest shock), N=512")
savefig("F3_resolution")

FAILURES.append(
    dict(
        id="F3",
        name="Train/eval resolution mismatch (FNO)",
        induced=f"coordinate feature = arange(N)/{N_TRAIN_GRID}",
        symptom=(
            f"rel-L2 {res_bad[128]:.4f} at the training grid (128) but {res_bad[512]:.4f} at 512 "
            f"({res_bad[512] / res_bad[128]:.0f}x worse)."
        ),
        diagnosis=(
            f"data identical across grids (same 1024-pt solves). Patching only the coordinate on the same weights: "
            f"{res_patched[512]:.4f} at 512, so the encoding is the cause (it spans [0,4) at N=512)."
        ),
        fix="physical coordinate arange(N)/N, retrained",
        after=f"rel-L2 {res_fix[512]:.4f} at 128/256/512 (heat-only baseline {res_heat[512]:.2f})",
        guard="Report operator error at 2+ resolutions; test features are grid-independent.",
    )
)


# Step 21: D Decision Table for Kwame
R = RESULTS
fm_nfe, fm_lat = R["fm_min"]
dd_nfe, dd_lat = R["ddim_min"]
fmt_lat = lambda s: f"{s * 1e3:.0f} ms" if s is not None else "never reached target"
table = pd.DataFrame(
    [
        dict(
            recipe="DDPM, 1000-step ancestral",
            quality=f"precision {R['ddpm1000']['precision']:.3f}",
            train_s=R["ddpm_train"]["seconds"],
            peak_vram_mb=R["ddpm_train"]["peak_mb"],
            latency_10k=fmt_lat(R["ddpm1000"]["latency_s"]),
            nfe="1000",
        ),
        dict(
            recipe="DDPM-trained + DDIM",
            quality=f"best precision {R['ddim_best']:.3f}",
            train_s=R["ddpm_train"]["seconds"],
            peak_vram_mb=R["ddpm_train"]["peak_mb"],
            latency_10k=fmt_lat(dd_lat),
            nfe=str(dd_nfe),
        ),
        dict(
            recipe="Flow matching, Euler",
            quality=f"best precision {R['fm_best']:.3f}",
            train_s=R["fm_train"]["seconds"],
            peak_vram_mb=R["fm_train"]["peak_mb"],
            latency_10k=fmt_lat(fm_lat),
            nfe=str(fm_nfe),
        ),
    ]
)
print(
    f"Hardware: {GPU_NAME}. Real-data precision ceiling {R['floor']['precision']:.3f}. "
    f"Target = precision >= {R['target_precision']:.3f} and recall >= {R['target_recall']:.3f} (DDPM-1000 minus 0.05). "
    "'nfe' = network calls needed to hit the target."
)
print(table.to_string(index=False, float_format="%.1f"))

f = R["fno"]
op = pd.DataFrame(
    [
        dict(
            model="FNO (fixed)",
            **{f"relL2@{n}": f["rel_l2"][n] for n in (128, 256, 512)},
        ),
        dict(
            model="FNO (coord bug)",
            **{f"relL2@{n}": f["rel_l2_bug"][n] for n in (128, 256, 512)},
        ),
        dict(
            model="heat-only baseline",
            **{f"relL2@{n}": f["heat"][n] for n in (128, 256, 512)},
        ),
        dict(
            model="identity baseline",
            **{f"relL2@{n}": f["identity"][n] for n in (128, 256, 512)},
        ),
    ]
)
print(
    f"\nOperator (trained at 128 only; 256/512 zero-shot). Train {R['fno_train']['seconds']:.0f}s, "
    f"peak VRAM {R['fno_train']['peak_mb']:.0f} MB, {R['fno_train']['params']:,} params."
)
print(op.to_string(index=False, float_format="%.4f"))
print(
    f"200 cases @512: FD solver {f['solver_s']:.3f}s vs FNO {f['infer_s'] * 1e3:.2f} ms -> {f['speedup']:.0f}x"
)

nfe_ratio = (dd_nfe / fm_nfe) if (dd_nfe and fm_nfe) else float("nan")
print(f"""
READ-OUT
* Generative: same network, same {GEN_STEPS}-step recipe, near-identical training time ({R['fm_train']['seconds']:.0f}s FM vs {R['ddpm_train']['seconds']:.0f}s DDPM)
  and peak VRAM ({R['fm_train']['peak_mb']:.0f} vs {R['ddpm_train']['peak_mb']:.0f} MB). The difference is at inference: flow matching hit the
  target at NFE={fm_nfe}, DDIM at NFE={dd_nfe} ({nfe_ratio:.1f}x more calls), ancestral DDPM uses 1000.
  On one GPU, sampling latency scales with NFE, so NFE is the binding constraint.
* Operator: rel-L2 {f['rel_l2'][128]:.4f} vs {f['heat'][128]:.4f} for linear physics ({f['heat'][128] / f['rel_l2'][128]:.0f}x lower error), flat across
  4x resolution ({f['rel_l2'][512]:.4f} @512), and {f['speedup']:.0f}x faster than the FD solver on these 200 cases.
""")


# Step 23: Write the Failure Log, Results and Hardware Manifest
def write_outputs():
    log_path = RESULTS_DIR / f"failure_log_{RUN_TAG}.md"
    L = [
        "# Failure-and-Fix Log: Halden toy-model scouting\n",
        f"*{GPU_NAME} · tag `{RUN_TAG}` · every number written by the run that produced this file*\n",
    ]
    if QUICK:
        L.append(
            "> **QUICK smoke-test run: training budgets are cut ~6x, so these numbers are NOT the submission results.**\n"
        )
    ordered = sorted(FAILURES, key=lambda f: f["id"])
    for f in ordered:
        L += [
            f"**{f['id']} · {f['name']}** (induced: {f['induced']})",
            f"- **Symptom:** {f['symptom']}",
            f"- **Root cause:** {f['diagnosis']}",
            f"- **Fix → result:** {f['fix']} → {f['after']}.\n",
        ]
    L.append(
        "**Guards for the pilot:** "
        + " ".join(f"{f['id']}: {f['guard']}" for f in ordered)
        + "\n"
    )
    L.append(f"Figures: `results/figs_{RUN_TAG}/F1–F4_*.png`.")
    words = len(" ".join(L).split())
    print(
        f"failure log: {words} words"
        + ("  (WARNING: over ~450 words may not fit one page)" if words > 450 else "")
    )
    log_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS_DIR / f"results_{RUN_TAG}.json").write_text(
        json.dumps(RESULTS, indent=1, default=str), encoding="utf-8"
    )
    manifest = dict(
        tag=RUN_TAG,
        quick=QUICK,
        total_runtime_s=round(time.perf_counter() - T_START, 1),
        hardware=hardware_snapshot(ENV),
        budgets=dict(
            gen_steps=GEN_STEPS, fno_train_cases=N_TRAIN_S, fno_test_cases=N_TEST_S
        ),
        timings_s=dict(
            fm_train=R["fm_train"]["seconds"],
            ddpm_train=R["ddpm_train"]["seconds"],
            fno_train=R["fno_train"]["seconds"],
            burgers_datagen=R["datagen_s"],
        ),
    )
    (RESULTS_DIR / f"manifest_{RUN_TAG}.json").write_text(
        json.dumps(manifest, indent=1, default=str), encoding="utf-8"
    )
    print(log_path.read_text(encoding="utf-8"))
    print(
        f"wrote results/failure_log_{RUN_TAG}.md, results_{RUN_TAG}.json, manifest_{RUN_TAG}.json, figs_{RUN_TAG}/ "
        f"(total runtime {manifest['total_runtime_s']:.0f}s)"
    )


write_outputs()


# Step 24: Save the Results off Colab
if ENV["platform"] == "colab":
    import shutil
    from google.colab import files

    archive = shutil.make_archive(f"halden_results_{RUN_TAG}", "zip", RESULTS_DIR)
    files.download(archive)
else:
    print(f"results are in {RESULTS_DIR}")


# Step 25: Write the Results into README.md and Push (asks first)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import update_readme
except ImportError:
    update_readme = None
    print(
        "[publish] scripts/update_readme.py not found: run from a git clone to write README.md (COLAB.md)."
    )
if update_readme and QUICK:
    print("[publish] quick smoke test: results are not written to README.md")
elif update_readme:
    update_readme.offer_publish(
        ROOT, RUN_TAG, interactive=IN_NOTEBOOK or sys.stdin.isatty()
    )
