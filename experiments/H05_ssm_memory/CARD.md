<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H05 · A state-space memory layer makes plans steadier and reactions earlier

**Naren, Siddesh** · NAVSIM, neuroNCAP, Bench2Drive, AlpaSim · ⏸ parked (inconclusive) · 21 Oct – 4 Nov 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Priority:** 5

## Question

Does a planner that carries a running state across ticks drive better than one that sees only the last 0.5 s? Extended comfort is the base's biggest NAVSIM gap (−0.077); neuroNCAP's side crashes are runs that never brake although the car was in view for 2 s or more; Bench2Drive stalls after a first stop.

## Hypothesis

Drive-JEPA sees two frames 0.5 s apart and the current ego status; nothing survives from one tick to the next. A state-space layer (Mamba-style: a learned recurrence with a fixed-size state, linear in sequence length) between the frozen encoder and the planner can carry the scene forward: how the crossing car has been moving, a brake already begun, an actor that has left the view. Expected: earlier, steadier reactions in closed loop and better NAVSIM extended comfort, with no open-loop loss (no size is given). Refuted at step 0 if a probe on the last k ticks stops improving at k = 2 (little for memory to add: the layer is not built); at step 2 unless the arm with state carried across ticks beats the same layer reset every tick (else the gain is capacity, not memory); at step 3 unless it beats that control in closed loop, nothing regresses, and reactions come no later.

## Predictions

⚠️ Not pre-registered: the step-0 rule was fixed in this record's hypothesis (committed 3 Nov, 4cf7c95) and in the 27 Oct plan, not in a PREDICTIONS.md

## What we learned

Against the model's own view (one moment: now and 0.5 s ago), one extra moment improved a ridge probe of human speed at +2 s by 0.021 m/s [0.014, 0.028] (4%, events slice, 176 val logs); 0.013 [0.008, 0.019] of it is the camera frame from 1 s ago, possibly just the extra 128 numbers. Beyond 1 s only the car state's past helped (0.007 m/s); camera history added nothing (CI includes 0). Braking was never better than one moment; the rise with longer camera windows is likely overfitting. Limits: 128-number pooled summaries, ordinary driving, no crossing-car events; the probe could not see what a trained memory could. The registered step-0 refutation ("stops improving at k = 2") was not met by ridge speed, which still improved from k = 2 to k = 4 (0.0070 m/s [0.0023, 0.0120]); control (c) puts that gain on the car state's past.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H05.r1**<br>feature cache: released Drive-JEPA's outputs at every moment of navtrain | released | —; frozen: all of Drive-JEPA: encoder, generator and scorer | navsim `navtrain (vendor log split: 978 train / 214 val logs)` 126,694 moments (eval) — every moment with both front images; 91,977 of them are graded tokens (train 76,657 / val 15,320) | commit `731129f` | — |
| **H05.r2**<br>step-0 probe (k = 1, 2, 4, 8), control (c) camera past vs car-state past, and the k = 1 refit | H05.r1 | ridge and logistic probes, a one-hidden-layer MLP (offline, CPU); frozen: all of Drive-JEPA: encoder, generator and scorer | navsim `vendor train logs, graded, chain_pos >= 8, 3 s of log ahead` 35,230 moments (train)<br>navsim `vendor val logs, graded, chain_pos >= 8, 3 s of log ahead` 7,942 moments (held-out) — 176 of the 214 val logs have such a moment; events slice (\|dv\| > 1 m/s within 2 s) 3,302 | commit `ecfaaa9` | — |

- **H05.r1:** 126,694 moments (0 failures, 10.1 h, 4 decode workers on cores 0-3) into {data}/memory_cache/release: encoder map (512, 256), scorer input (32, 256), proposals (32, 8, 3), logits (32, 6), ego (11), float32, 71 GB. Released checkpoint sha256 740335c8...2d30d05. Ran at d48b35a; rebased onto b2d2c59 as 731129f with src/gallops/memory and gallops.compat unchanged (empty diff).
- **H05.r2:** Probe ran at af3d58e (54 min), control (c) and the k = 1 refit at 8e3bf5a (25 min, 2 min); rebased onto b2d2c59 as 2c3a94b / ecfaaa9 with src/gallops/memory unchanged (empty diff). The refit reproduced all 80 stored values and CIs exactly before computing differences. Pooled encoder map (mean + 4x8) -> PCA 128 fitted on train moments, plus ego 11, per moment; windows of k moments along chains. Paired, log-clustered bootstrap, 2,000 draws, seed 0. k = 1 is the model's own view (now and 0.5 s ago).

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H05.r2 ★ | offline human speed at +2 s, RMSE, events slice: camera history k = 4 vs k = 2 with the car state now only (control V3) 0.5001 → **0.4992** (n=3,302 moments) | V3 at k = 2 | ●●○○ moderate · open-loop | {data}/memory_cache/probe/c/control_c.json; paired difference -0.0009 [-0.0045, +0.0025] |
| H05.r2 | offline human speed at +2 s, RMSE, events slice: one extra moment (camera + car) vs the model's own view, k = 2 vs k = 1 0.5117 → 0.4909 (n=3,302 moments) | k = 1 | ●●○○ moderate · open-loop | {data}/memory_cache/probe/probe.json; paired difference -0.0208 [-0.0279, -0.0135] |
| H05.r2 | offline human speed at +2 s, RMSE, events slice: camera frame from 1 s ago on top of 1 s of car history (V1 - V2 at k = 2) 0.5043 → 0.4909 (n=3,302 moments) | V2 at k = 2 | ●●○○ moderate · open-loop | {data}/memory_cache/probe/c/control_c.json; paired difference -0.0134 [-0.0186, -0.0082]; may be the extra 128 numbers |
| H05.r2 | offline human speed at +2 s, RMSE, events slice: car history k = 4 vs k = 2 with the camera now only (control V2) 0.5043 → 0.4971 (n=3,302 moments) | V2 at k = 2 | ●●○○ moderate · open-loop | {data}/memory_cache/probe/c/control_c.json; paired difference -0.0072 [-0.0103, -0.0043] |
| H05.r2 | offline braking onset within 2 s, log-loss, events slice: camera + car k = 2 vs k = 1 0.1988 → 0.2085 (n=3,302 moments) | k = 1 | ●●○○ moderate · open-loop | {data}/memory_cache/probe/probe.json; paired difference +0.0098 [+0.0049, +0.0151]; likely overfitting |
