# S4 — the real training run

**VERDICT: PARTIAL — V2 and V3 fail.** All 10 passes ran, no tripwire fired. The generator improved a lot on unseen clips; the scorer did
not. Candidate = pass 9 (lowest unseen total error, 1.33 against 6.74 for the shipped weights).

Run 2026-10-07 01:20 → 13:32 on {site-host}. Everything below is on `val_full` (24,657 unseen examples, 11,842 graded) unless it says otherwise.

## Pre-registered, at the best pass (9)

| id | statement | result | verdict |
|---|---|---|---|
| V1 | highway: fastest of 32 ≥ 0.9 × human speed on ≥ 70 % | 99.6 % of 6,534 (shipped weights: 14.3 %) | holds |
| V2 | scorer: pick = true best on ≥ 50 % of graded | 37.9 % of 11,842 (shipped: 34.2 %) | **fails** |
| V3 | pass examples: pick within 1.5 m of the human on ≥ 60 % | 14.8 % of 1,570 (shipped: 14.3 %) | **fails** |
| V4 | ordinary: 4 s distance not worse than pass 0 by more than 5 % | 18.2 % better (6.73 → 5.51 m) | holds |
| V5 | no tripwire before pass 10 | none; 10 passes done | holds |

V2 and V3 fail on every pass, not only on the best one (V2 between 33.6 % and 44.1 %, V3 between 13.0 % and 15.9 %), and neither trends up.

## What the run shows

- **Generator: large gain, most of it in pass 1.** Unseen total error 6.74 → 1.82 after one pass, then a slow drift to 1.33–1.36. Highway
  error fell from 21.6 to about 2.6; the chosen highway plan ends 16 m from the human instead of 41 m.
- **Highway menu overshoots.** The 14 m/s ceiling is gone, but the fastest of the 32 proposals is 1.4–2.8 × the human's speed (1.70 × at
  pass 9, median 37.8 m/s against 21.1). V1 holds partly because of that. The chosen plan is slower than the human (0.78 × at pass 9).
- **Scorer: training error falls, unseen error does not.** On graded examples the scorer part drops from 0.65 to 0.41 on the training
  probe but stalls at 0.49–0.51 on unseen clips from pass 4 on. Its pick equals the true best on 38 % (highway 11 %, pass 43 %, ordinary 50 %).
- **Close-pass is a scorer problem, not a menu problem.** A proposal within 1.5 m of the human is on the menu on 93.5 % of pass examples
  (shipped: 83.8 %); the scorer picks one on 14.8 %. Its pick is a median 5.2 m from the human, the closest proposal 0.46 m.
- **No forgetting.** Ordinary is 13–22 % better than the shipped weights on every pass.
- **Unseen error is noisy pass to pass** (1.35, 1.46, 1.36, 1.39, 1.60, 1.33, 1.36 for passes 4–10) while training error falls
  steadily. The swings are almost all highway. Passes 4, 6, 9 and 10 are within 0.03 of each other; "best pass = 9" is a thin margin.
- **Pull-out on unseen clips rests on one clip** (32 examples): 7.7 m → 7.7 m at pass 9, no evidence either way.

## Settings (as run)

| setting | value |
|---|---|
| Data | all 126,981 training examples (`04_training_set/manifest.parquet`); ungraded examples train the generator only |
| Sampling | 148,796 draws per pass: highway or graded examples twice (21,815), never more than twice |
| Share seen, every pass | ordinary 81.9 %, highway 8.6 %, pull-out 6.1 %, pass 3.4 %; graded 29.3 % of draws (17.2 % of examples) |
| Start | shipped checkpoint, sha256 `740335c8…d30d05` |
| Trained / frozen | generator (2.38 M) + scorer (4.70 M) trained; encoder (304.1 M) frozen |
| Catalogue | `vocab_10240.npy`, sha256 `4b0bc3c9…39e785` |
| Inputs | acceleration zeroed; bonnet mask 37 rows |
| Schedule | 10 passes, 4,650 steps each; Adam, lr 1e-4, batch 32, 16-bit mixed precision, no clipping, GridMask jitter (all upstream) |
| Evaluation | after every pass; small set (608) for passes 1–6 during the run, `val_full` from pass 7; passes 0–6 re-evaluated on `val_full` |
| Environment | torch 2.8.0+cu128 overlay + stubbed mmcv, as in the smoke test |

One-line manifest with exact values: `settings_manifest.json`.

**Parity re-run before training** (8 samples, overlay on GPU vs the shipped env on CPU): plans differ by at most 0.13 mm, scorer outputs by
1.9e-5, same pick on 8 of 8 (`parity/parity_s4_ov_gpu.npz`).

## Part A — full unseen set

`04_training_set/val_full/manifest_val_full.parquet`: 24,657 examples, 801 clips, 499 chunks, 11,842 graded (48.0 %), 117 with the D3 exemption.

| source | type | clips | examples | graded | dropped by D2 |
|---|---|---|---|---|---|
| map | highway | 110 | 3,340 | 3,315 | 173 |
| map | pass | 57 | 1,570 | 1,563 | 254 |
| map | ordinary | 234 | 6,981 | 6,964 | 595 |
| raw | highway | 100 | 3,194 | 0 | — |
| raw | pull-out | 1 | 32 | 0 | — |
| raw | ordinary | 299 | 9,540 | 0 | — |

- Asserted: same schema as train, 0 chunks shared with train, 0 exam / test / train clips, unique example ids.
- **Pull-outs in the 400 seeded raw clips: 1.** 68 clips had a move-off; one meets the pull-out rule.
- 100 raw clips hold ≥ 16 m/s for 4.5 s and are typed highway (train's raw ordinary pool excluded such clips).
- D2 dropped 1,022 moments (7.9 % of map moments; train 6.2 %); three ordinary map clips lost all their moments.
- Took 6.9 h against the ~5 h target (grading at 10 processes on a loaded machine, plus a 45-minute stall before the manifest step).
- Detail: `partA.md`.

## Per-pass tables

### R1 — error (loss) per pass

| pass | training, as trained | training probe | unseen total | unseen generator | unseen scorer (graded) |
|---|---|---|---|---|---|
| 0 |  | 5.97 | 6.74 | 6.43 | 0.631 |
| 1 | 2.32 | 1.55 | 1.82 | 1.56 | 0.535 |
| 2 | 1.94 | 1.35 | 1.64 | 1.39 | 0.517 |
| 3 | 1.83 | 1.46 | 1.83 | 1.57 | 0.556 |
| 4 | 1.74 | 1.2 | 1.35 | 1.11 | 0.495 |
| 5 | 1.69 | 1.24 | 1.46 | 1.23 | 0.496 |
| 6 | 1.63 | 1.18 | 1.36 | 1.11 | 0.512 |
| 7 | 1.61 | 1.21 | 1.39 | 1.14 | 0.514 |
| 8 | 1.55 | 1.2 | 1.6 | 1.37 | 0.492 |
| 9 | 1.52 | 1.1 | 1.33 | 1.09 | 0.506 |
| 10 | 1.49 | 1.11 | 1.36 | 1.12 | 0.502 |

Unseen total error by type (graded / not graded):

| pass | highway graded | highway ungraded | ordinary graded | ordinary ungraded | pass graded | pass ungraded | pull-out ungraded |
|---|---|---|---|---|---|---|---|
| 0 | 26.6 | 16.5 | 1.9 | 0.91 | 1.86 | 2.31 | 0.948 |
| 1 | 5.11 | 2.65 | 1.56 | 0.625 | 1.65 | 1.3 | 0.619 |
| 2 | 4.49 | 2.29 | 1.48 | 0.562 | 1.59 | 1.23 | 0.608 |
| 3 | 5.27 | 2.76 | 1.54 | 0.586 | 1.61 | 1.5 | 0.555 |
| 4 | 3.29 | 1.66 | 1.4 | 0.507 | 1.53 | 1.25 | 0.519 |
| 5 | 3.81 | 1.89 | 1.42 | 0.528 | 1.55 | 1.65 | 0.53 |
| 6 | 3.45 | 1.66 | 1.37 | 0.502 | 1.51 | 1.58 | 0.574 |
| 7 | 3.56 | 1.68 | 1.4 | 0.512 | 1.55 | 1.18 | 0.515 |
| 8 | 4.6 | 2.38 | 1.37 | 0.496 | 1.5 | 1.43 | 0.511 |
| 9 | 3.33 | 1.63 | 1.36 | 0.489 | 1.53 | 1.43 | 0.526 |
| 10 | 3.58 | 1.65 | 1.34 | 0.479 | 1.5 | 1.24 | 0.561 |

### R2 — generator

Chosen plan's distance from the human's path at 4 s, metres (mean):

| pass | highway graded | highway not graded | ordinary graded | ordinary not graded | pass graded | pass not graded | pull-out not graded |
|---|---|---|---|---|---|---|---|
| 0 | 42.5 | 39.9 | 6.65 | 6.79 | 7.01 | 16 | 7.71 |
| 1 | 18.5 | 17 | 5.37 | 6 | 6.58 | 9.4 | 8.18 |
| 2 | 20.7 | 19.4 | 5.46 | 5.9 | 6.81 | 10.9 | 8.49 |
| 3 | 15.8 | 15.9 | 5.52 | 6.04 | 7.23 | 8.69 | 8.98 |
| 4 | 19.6 | 19.3 | 5.3 | 5.54 | 6.4 | 9.87 | 8.34 |
| 5 | 19.1 | 17.8 | 5.26 | 5.58 | 6.22 | 10 | 7.24 |
| 6 | 16.8 | 15.3 | 5.38 | 5.73 | 6.45 | 9.63 | 8.21 |
| 7 | 17.1 | 16.9 | 5.03 | 5.46 | 6.34 | 8.47 | 7.3 |
| 8 | 20.5 | 19.4 | 5.39 | 5.78 | 6.27 | 9.56 | 8.18 |
| 9 | 16.8 | 15.7 | 5.24 | 5.71 | 6.36 | 9.19 | 7.68 |
| 10 | 15.1 | 13.2 | 5.07 | 5.55 | 6.2 | 9.84 | 7.51 |

Highway, fastest of the 32 proposals vs the human (speed over the last half second of the 4 s):

| pass | examples | fastest of 32 / human speed (median) | fastest ≥ 0.9 × human | fastest of 32, m/s (median) | human, m/s (median) | chosen / human speed (median) |
|---|---|---|---|---|---|---|
| 0 | 6534 | 0.695 | 14.3 % | 14.3 | 21.1 | 0.504 |
| 1 | 6534 | 2.2 | 99.8 % | 48.8 | 21.1 | 0.893 |
| 2 | 6534 | 2.84 | 99.9 % | 64.6 | 21.1 | 0.755 |
| 3 | 6534 | 1.79 | 99.9 % | 38.8 | 21.1 | 0.847 |
| 4 | 6534 | 1.94 | 99.4 % | 41.4 | 21.1 | 0.719 |
| 5 | 6534 | 1.75 | 97.7 % | 39.5 | 21.1 | 0.75 |
| 6 | 6534 | 1.6 | 99.3 % | 34.3 | 21.1 | 0.799 |
| 7 | 6534 | 1.85 | 99.5 % | 43.1 | 21.1 | 0.823 |
| 8 | 6534 | 1.42 | 98.4 % | 28.6 | 21.1 | 0.716 |
| 9 | 6534 | 1.7 | 99.6 % | 37.8 | 21.1 | 0.775 |
| 10 | 6534 | 1.66 | 99.8 % | 35.7 | 21.1 | 0.869 |

### R3 — scorer (graded examples)

| pass | pick = true best, all | rank agreement, all | pick = best, highway | pick = best, pass | pick = best, ordinary |
|---|---|---|---|---|---|
| 0 | 34.2 % | 0.112 | 5.1 % | 42.7 % | 46.1 % |
| 1 | 44.1 % | 0.298 | 14.7 % | 47.5 % | 57.4 % |
| 2 | 37.6 % | 0.243 | 11.4 % | 43.1 % | 48.8 % |
| 3 | 37.6 % | 0.249 | 12.7 % | 39.3 % | 49.2 % |
| 4 | 37.8 % | 0.383 | 9.6 % | 42.4 % | 50.2 % |
| 5 | 34.7 % | 0.25 | 8.0 % | 37.6 % | 46.8 % |
| 6 | 42.8 % | 0.254 | 15.4 % | 45.0 % | 55.3 % |
| 7 | 33.6 % | 0.242 | 9.6 % | 37.1 % | 44.2 % |
| 8 | 35.8 % | 0.279 | 6.4 % | 41.1 % | 48.5 % |
| 9 | 37.9 % | 0.232 | 11.1 % | 42.7 % | 49.5 % |
| 10 | 40.3 % | 0.243 | 16.0 % | 41.5 % | 51.6 % |

'pick = true best' counts a pick whose true grade equals the best grade among the 32 (ties count). Rank agreement = Spearman correlation between the scorer's 32 scores and the true grades, mean over examples.

Pass examples: the scorer's pick vs the closest of the 32 proposals:

| pass | examples | pick: distance from human at 4 s, m (median) | closest of 32, m (median) | pick within 1.5 m | closest of 32 within 1.5 m |
|---|---|---|---|---|---|
| 0 | 1570 | 5.47 | 0.512 | 14.3 % | 83.8 % |
| 1 | 1570 | 5.49 | 0.59 | 15.9 % | 89.5 % |
| 2 | 1570 | 5.54 | 0.546 | 15.0 % | 92.4 % |
| 3 | 1570 | 5.93 | 0.517 | 13.1 % | 93.1 % |
| 4 | 1570 | 5.09 | 0.469 | 15.0 % | 92.3 % |
| 5 | 1570 | 5.09 | 0.436 | 14.0 % | 93.7 % |
| 6 | 1570 | 5.05 | 0.481 | 13.0 % | 95.1 % |
| 7 | 1570 | 4.99 | 0.459 | 14.4 % | 93.8 % |
| 8 | 1570 | 4.97 | 0.449 | 15.0 % | 93.1 % |
| 9 | 1570 | 5.15 | 0.464 | 14.8 % | 93.5 % |
| 10 | 1570 | 4.96 | 0.447 | 15.0 % | 94.5 % |

### R4 — ordinary (forgetting tripwire)

| pass | ordinary 4 s distance, m | vs pass 0 |
|---|---|---|
| 0 | 6.73 | +0.0 % |
| 1 | 5.74 | -14.8 % |
| 2 | 5.71 | -15.1 % |
| 3 | 5.82 | -13.5 % |
| 4 | 5.44 | -19.3 % |
| 5 | 5.45 | -19.1 % |
| 6 | 5.58 | -17.1 % |
| 7 | 5.28 | -21.6 % |
| 8 | 5.62 | -16.6 % |
| 9 | 5.51 | -18.2 % |
| 10 | 5.35 | -20.6 % |

### R5 — share seen in training

| pass | draws | graded_share_seen | ordinary | highway | pull-out | pass |
|---|---|---|---|---|---|---|
| 1 | 148796 | 0.293 | 0.819 | 0.0862 | 0.0608 | 0.0336 |
| 2 | 148796 | 0.293 | 0.819 | 0.0862 | 0.0608 | 0.0336 |
| 3 | 148796 | 0.293 | 0.819 | 0.0862 | 0.0608 | 0.0336 |
| 4 | 148796 | 0.293 | 0.819 | 0.0862 | 0.0608 | 0.0336 |
| 5 | 148796 | 0.293 | 0.819 | 0.0862 | 0.0608 | 0.0336 |
| 6 | 148796 | 0.293 | 0.819 | 0.0862 | 0.0608 | 0.0336 |
| 7 | 148796 | 0.293 | 0.819 | 0.0862 | 0.0608 | 0.0336 |
| 8 | 148796 | 0.293 | 0.819 | 0.0862 | 0.0608 | 0.0336 |
| 9 | 148796 | 0.293 | 0.819 | 0.0862 | 0.0608 | 0.0336 |
| 10 | 148796 | 0.293 | 0.819 | 0.0862 | 0.0608 | 0.0336 |


### Time and GPU

| pass | steps | sec_per_step | train min | gpu_reserved_gb | elapsed_h | evaluated on (during run) |
|---|---|---|---|---|---|---|
| 1 | 4650 | 1.29 | 99.7 | 4.41 | 1.73 | val |
| 2 | 4650 | 0.874 | 67.7 | 4.41 | 2.87 | val |
| 3 | 4650 | 0.696 | 53.9 | 9.21 | 3.85 | val |
| 4 | 4650 | 0.697 | 54 | 9.21 | 4.79 | val |
| 5 | 4650 | 0.627 | 48.6 | 9.21 | 5.61 | val |
| 6 | 4650 | 0.554 | 42.9 | 9.21 | 6.33 | val |
| 7 | 4650 | 0.861 | 66.7 | 9.21 | 7.71 | val_full |
| 8 | 4650 | 0.966 | 74.9 | 9.21 | 9.24 | val_full |
| 9 | 4650 | 0.977 | 75.7 | 9.21 | 10.8 | val_full |
| 10 | 4650 | 0.824 | 63.9 | 9.21 | 12.1 | val_full |

Seconds per step include grading the proposals of graded examples on the CPU. Passes 7–9 shared the machine with the re-evaluation process.

## Curves

- `curves_total.png` — training vs unseen error per pass: total, generator part, scorer part.
- `curves_by_type.png` — total error per type, training probe vs unseen.
- `curves_distance_by_type.png` — chosen plan vs human at 4 s per type.

The "training probe" is a fixed seeded sample of 815 training examples (150 per type × graded group where available) evaluated exactly
like the unseen set, so the two lines are comparable. "Training, as trained" is the running loss with jitter and sample weights.

## Tripwires

None fired. T1 came closest: unseen error rose at passes 7 and 8 (1.36 → 1.39 → 1.60) and fell at pass 9 (1.33). T2: ordinary never
worse than pass 0. T3: no non-finite loss. T4: GPU peak 9.2 GiB reserved (8 GiB of it pre-reserved) under the 12 GiB cap; the 10 h wall
cap was removed by Sid at 04:39 and the run took 12.1 h.

## Deliverables

| item | where |
|---|---|
| Candidate weights (pass 9) | `{runs}/drivejepa_retrain/06_first_training_run/candidate/drive_jepa_s4_pass09.ckpt` |
| sha256 / size | `8a71c3c0d84a9758de46c7b46eaf18556a62ac4837feea7271040e7a0caacdd2` / 1,273,002,795 bytes |
| Layout | upstream state dict (`agent._pad_model.*`): 294 encoder tensors bit-for-bit from the shipped file, 201 from S4 (185 differ) |
| Every pass's weights | `{runs}/drivejepa_retrain/06_first_training_run/run/ckpt/pass_00.pt … pass_10.pt` (trained tensors + optimiser) |
| Adapter 0.1.6 patch | `adapter_0.1.6.diff` (156 lines, **not applied**): 0.1.4b + candidate path with size and sha256 check + bonnet mask + zeroed acceleration |
| Sanity run | `sanity/result.csv` |
| Readouts | `run/readouts/` (per example), `run/evals.csv`, `run/curve.csv`, `run/tables.md`, `run/checks.json` |

**Offline sanity run (3 examples, CPU, shipped torch 2.1):** highway 30.0 m/s, pass 8.9 m/s, raw ordinary 7.4 m/s. Each was taken from
the raw clip through the 0.1.6-patched planner's own preprocessing and through the adapter's model wrapper with the candidate file, and
compared with the training-side loader and agent. Both images, `lidar2img`, `img_shape` and the ego-status row the model reads are
bit-identical on 3 of 3; the 32 proposals and scores differ by 0.0; the chosen plan is the same on 3 of 3. The sanity run does not
exercise the momentum re-score (it needs a previous plan).

## Flags

1. **0.1.6 design choice to confirm.** The model input carries zeroed acceleration; the momentum re-score still gets the car's real
   acceleration, read from the observation, because it simulates from the real state. The patch is written that way.
2. **Evaluation set changed mid-run.** Tripwires were checked on the small set for passes 1–6 and on `val_full` from pass 7. On
   `val_full`, checked after the fact, no tripwire would have fired earlier either.
3. **Three launches before the run proper, and three resumes inside it.** Launch 1 died at step 0 because another tenant had filled the
   GPU; launch 2 I stopped after 8 minutes to add per-type evaluation. The run was resumed from saved weights and optimiser at the
   pass 2 boundary (twice) and the pass 3 boundary, for a grading-throughput change, an evaluation fix and the cap removal. About
   6 minutes of pass 3 were redone. Logs of the aborted launches: `aborted_launch1/`, `aborted_launch2/`.
4. **Evaluation bug, fixed at 04:20.** A group whose last batch had fewer than 4 examples lost one batch and double-counted another.
   Training batches were never affected. All evaluations in this report were computed after the fix, and every evaluation now asserts
   that each example is scored exactly once.
5. **Loader change: slim copy.** The NAS delivers about 25 examples/s from the original 2.4 MB files, which would be 1.7 h per pass. The
   patched loader writes a lossless copy of exactly what the model reads (0.3 MB) on first read and verifies it bit-identical before it
   lands. 45 GB at `{runs}/drivejepa_retrain/06_first_training_run/slim/`. The 448×768 `camera_feature`, which the model never reads, is dropped.
6. **Grading is the speed limit, not the GPU.** One graded example per grading task (upstream: two) and 12 processes brought a step
   from 0.87 s to 0.55–0.70 s. This changes how work is spread over processes, not what is computed.
7. **Mixed precision.** The forward pass runs in 16-bit, as upstream; the loss is computed in 32-bit.
8. **Wall-clock counter.** The logged 12.09 h excludes about 10 minutes between stops and restarts; real time was 12.2 h.
9. **V1 measures the fastest proposal over the last half second of the 4 s plan**, as in the smoke test.
10. Nothing was written under `adapters/`, `runs/` or `drafts/`. Vendored code is untouched; all model and loader changes are in
    `05_small_overfit_check/train_patch/` (diffs regenerated: `drive_jepa_agent.diff`, `dataset.diff`).
