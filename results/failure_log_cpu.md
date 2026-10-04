# Failure-and-Fix Log: Halden toy-model scouting

*CPU only · tag `cpu` · every number written by the run that produced this file*

**F1 · Exploding learning rate (flow matching)** (induced: Adam lr 0.05, no warmup, no clipping)
- **Symptom:** grad-norm peaked at 1,113; samples no better than a Gaussian (precision 0.51 vs 0.50).
- **Root cause:** LR range test: peak grad-norm ~1.0 at gentle LRs, 453 at lr 0.032 (the cliff); 0.05 is past it.
- **Fix → result:** lr 0.003 (cliff / 10) + warmup, cosine decay, clip 1.0 → precision 0.93 (real data 0.95).

**F2 · Mismatched noise schedule (DDPM)** (induced: trained on T=1000, sampled with a rebuilt T=100 schedule)
- **Symptom:** recall 0.94 -> 0.66, precision 0.91 -> 0.71; no error raised.
- **Root cause:** same t, different noise (sigma 0.47 vs trained 0.96 at t=0.5). Denoiser probe: eps-MSE 0.08 on trained-schedule inputs vs 1.01; network fine, schedule wrong.
- **Fix → result:** respace 100 steps of the trained schedule (or DDIM) → precision 0.91 / recall 0.92, 9x faster than 1000 steps.

**F3 · Train/eval resolution mismatch (FNO)** (induced: coordinate feature = arange(N)/128)
- **Symptom:** rel-L2 0.0035 at the training grid (128) but 0.0644 at 512 (18x worse).
- **Root cause:** data identical across grids (same 1024-pt solves). Patching only the coordinate on the same weights: 0.0035 at 512, so the encoding is the cause (it spans [0,4) at N=512).
- **Fix → result:** physical coordinate arange(N)/N, retrained → rel-L2 0.0035 at 128/256/512 (heat-only baseline 0.71).

**F4 · Metric blind spot (unplanned)** (induced: first run: 6k steps, judged on SWD alone)
- **Symptom:** SWD ~0.03, near the 0.014 floor, while the scatter plot showed blobs with no spiral arms.
- **Root cause:** checkpoints: precision 0.787 -> 0.850 -> 0.894 -> 0.915 -> 0.925 rises steadily; SWD 0.056 -> 0.057 -> 0.033 -> 0.029 -> 0.031 wanders. SWD averages 1D projections, so it cannot see the arms.
- **Fix → result:** train 20k steps; gate on precision/recall + scatter plot → precision FM 0.93, DDPM 0.91.

**Guards for the pilot:** F1: 2-min LR range test before every new run; alert on grad-norm spikes. F2: Save the schedule with the checkpoint; sampler refuses a mismatch. F3: Report operator error at 2+ resolutions; test features are grid-independent. F4: Acceptance metric must separate a known-bad model from a good one.

Figures: `results/figs_cpu/F1–F4_*.png`.
