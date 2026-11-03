<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# GC05 · Drive-JEPA retrained on NVIDIA Physical AI AV

**Siddesh** · AlpaSim · ✅ positive · 1 Oct – 12 Oct 2026 · gap-closing

[← all experiments](../../README.md#gap-closing-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Evidence for:** D05 ([decision log](../../README.md#decision-log))

## Question

Released Drive-JEPA scores 0.421 on AlpaSim's 30 scenes, below constant velocity (0.4876), and its 8,192-path vocabulary tops out near 15 m/s. Does retraining it on in-domain NVIDIA Physical AI AV driving, with a vocabulary that reaches highway speed, close that gap?

## Predictions

[PREDICTIONS_phase1.md](12_second_training_run/PREDICTIONS_phase1.md), written before the run.

## What we learned

In-domain data closes most of the AlpaSim gap: 0.5433 → 0.7697 on 30 scenes, one drive each (moderate, adapted). The gain is the generator's; the retrained scorer failed its own pick-the-best check (37.9% against a 50% bar), so the shipped checkpoint keeps the first run's scorer and retrains only the generator. Two of its three at-fault collisions are new, and two scenes never move under any driver.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **GC05.run1**<br>first training run: generator and scorer together, 10 passes; pass 9 kept | released (740335c800e2) | generator, scorer (7.08 M); frozen: encoder | physicalai_av `train` 126,981 samples (train) — from 4,015 clips; AlpaSim's 913 public scenes (which hold the 30 test scenes) excluded<br>physicalai_av `val_full` 24,657 examples (selection) — 801 clips; chose which pass to keep | [scripts in 06_first_training_run/](06_first_training_run/)<br>Drive-JEPA @ `e21f474` + patches | `{weights}/drivejepa_physicalai/first-run/step_000000009/`<br>sha256 `8a71c3c0d84a9758de46c7b46eaf18556a62ac4837feea7271040e7a0caacdd2` |
| **GC05.run2**<br>second training run: generator only, scorer frozen; 1 pilot pass + 2 passes (final_P0) | GC05.run1 | generator (2.38 M); frozen: encoder, scorer | physicalai_av `train` (train) — run 1's training set minus 145 held-out training clips<br>physicalai_av `train (145 held-out clips)` 4,400 examples (selection) — chose the arm and the pass by a rule written beforehand (PREDICTIONS_phase1.md)<br>physicalai_av `val` (eval) — the after-the-fact evaluation on clips neither run trained on (unseen_clip_evaluation.md) | [scripts in 12_second_training_run/](12_second_training_run/)<br>commit `3e8901a`<br>Drive-JEPA @ `e21f474` + patches | `{weights}/drivejepa_physicalai/second-run-generator/step_000000002/`<br>sha256 `29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916`<br>shipped as AlpaSim driver 0.1.7<br>⚠️ sidecar: train_git_sha was set after the fact; `val` is missing from trained_on |

- **GC05.run1:** Driven in AlpaSim as driver 0.1.6; never shipped.
- **GC05.run2:** Spacing weight lam = 0 (a tie-break, pilot_choice.json). Merged file = released encoder + run-1 scorer + run-2 generator, bit for bit (gen_ckpt.json).

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| GC05.run1 | AlpaSim scene_score 0.5433 → 0.7032 (n=30 scenes)<br>30 scenes, one drive each; at-fault collisions 4 → 2; highway and pull-outs not fixed (07_closed_loop_test/REPORT.md: MIXED) | released, driver 0.1.4b (momentum-aware selection) | ●○○○ argued · adapted | experiments/GC05_physicalai_retrain/07_closed_loop_test/runs_017/tables/totals.json |
| GC05.run1 | offline scorer picks the best of 32 (val) 34.2% → 37.9%<br>the run's own bar was ≥ 50%: failed, so the scorer was frozen for run 2 | released scorer | ●●○○ moderate | experiments/GC05_physicalai_retrain/06_first_training_run/REPORT.md |
| GC05.run2 ★ | AlpaSim scene_score 0.5433 → **0.7697** (n=30 scenes)<br>30 scenes, one drive each; SE 0.069; +0.226 paired; at-fault 4 → 3, two of them new; constant velocity 0.4876 (different simulator setting) | released, driver 0.1.4b (momentum-aware selection) | ●●○○ moderate · adapted | experiments/GC05_physicalai_retrain/07_closed_loop_test/runs_017/tables/totals.json |
| GC05.run2 | AlpaSim scene_score 0.7032 → 0.7697 (n=30 scenes)<br>30 scenes, one drive each; SE 0.068; +0.066 paired: not a resolved difference | run 1, driver 0.1.6 | ○○○○ unverified · adapted | experiments/GC05_physicalai_retrain/07_closed_loop_test/runs_017/tables/totals.json |
| GC05.run2 | offline scorer picks the best of 32 (val, batch 1) 38.3% → 54.6%<br>95% interval 52.1–56.9; same scorer weights, better candidates | run 1's checkpoint | ●●○○ moderate | experiments/GC05_physicalai_retrain/12_second_training_run/unseen_clip_evaluation.md |

## Checks that decided something

- **Small overfit check (05)** (7 Oct): 30 epochs on 1% of the set: BROKEN by its own rule (41% against ≤ 25%), under-trained by the author's reading → go-ahead for the full run (Sid's call)
- **Why the scorer did not improve (08–11)** (9 Oct): the menu caps the pick; a replacement head on frozen features fails on unseen clips (regret 0.110): features are the ceiling → freeze the scorer; run 2 retrains the generator only
- **Why 0.1.6 swerves and stays stopped (14)** (9 Oct): swerves: scorer, generator and controller together; one stuck scene is the scorer, the other both → a fix was tested offline only; no closed-loop check was run
- **Best possible pick (13), pre-registered** (12 Oct): with the best candidate always driven, the car still does not pull out on d2715023 or b44a4a44 → no training started for pull-outs

## Evidence outside this repo

- AlpaSim runs: paddock PR #68 recorded them in results/alpasim/*/legacy-*_30scenes. The retrained checkpoint's two, legacy-drivejepa_017_20hz and legacy-drivejepa_017_20hz_30scenes, are copied in experiments/GC05_physicalai_retrain/paddock/results/, since paddock holds only the released models' (paddock #132 removed them); the other drivers' rows stay in paddock.
- The AlpaSim driver that served the retrained checkpoint, paddock's tools/models/drivejepa/alpasim_driver/ at 0.1.7, is copied in experiments/GC05_physicalai_retrain/paddock/ (paddock #121 removed it).
- Weights, training set and large tables: off-repo at the training site, each with bytes and sha256 in experiments/GC05_physicalai_retrain/POINTERS.md.
