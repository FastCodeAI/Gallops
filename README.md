<h1 align="center">
  <img src="docs/assets/gallops.svg" height="44" alt=""> Gallops
</h1>

<p align="center"><strong>Gallops trains driving policies. <a href="https://github.com/FastCodeAI/Paddock">Paddock</a> scores them. Neither repo imports the other.</strong></p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href=".github/workflows/check.yml"><img src="https://img.shields.io/badge/gate-make_check_in_CI-555555.svg" alt="Gate: make check in CI"></a>
<!-- experiments:badges -->
  <a href="#gap-closing-experiments"><img src="https://img.shields.io/badge/gap_closing-8_experiments-2E8B4E.svg" alt="gap closing: 8 experiments"></a>
  <a href="#hypothesis-experiments"><img src="https://img.shields.io/badge/hypotheses-10_open_or_tested-1F6FEB.svg" alt="hypotheses: 10 open or tested"></a>
  <a href="experiments/README.md#regression-gate-log"><img src="https://img.shields.io/badge/gated_checkpoints-4-B4560A.svg" alt="gated checkpoints: 4"></a>
  <a href="#where-we-are"><img src="https://img.shields.io/badge/gallops_version-none_yet-555555.svg" alt="gallops version: none yet"></a>
<!-- /experiments:badges -->
</p>

We are building a JEPA-based driving model, and this repo is its evidence: every
experiment, who ran it, and what it showed. Click an ID for that experiment's card,
with its data, code, checkpoint sha256, results and gate verdicts.

![Gap closing leads to an agreed base. Hypotheses of one change each are then judged by the regression gate, until the gate ships a version.](docs/assets/experiments.svg)

## Where we are

<!-- experiments:status -->
- **Base model:** the 4-camera stack (`H01.run2`), agreed as the base on 19 Oct and not yet through the gate; gate comparisons are still against released Drive-JEPA.
- **Shipped version:** none yet. A version is earned by a gate SHIP ([how](docs/experiments.md#versions)).
- **Last gate run:** 6 Nov · [H15.r3-a-s0](experiments/H15_prediction_memory/CARD.md) · ⏸ **HOLD**.
- **In progress:** 7 hypotheses, 3 enablers.
<!-- /experiments:status -->

<a id="training-so-far"></a>

## Gap-closing experiments

Each of us took some benchmarks and tried to close released Drive-JEPA's gap on them.

<!-- experiments:gap-closing -->
| ID | Experiment | Owner | Headline result | Strength | Outcome |
|---|---|---|---|---|---|
| [**GC01**](experiments/GC01_rerank_kitscenes/CARD.md) | KITScenes re-ranker | Naren | KITScenes ADE@3s (m) 2.521 → **0.973** (n=223 samples) | ●●●○ strong · adapted, open-loop | ◐ mixed |
| [**GC02**](experiments/GC02_selection_without_training/CARD.md) | Selection without training | Naren, Siddesh | AlpaSim scene_score 0.4209 → **0.5433** (n=30 scenes) | ●●○○ moderate | ◐ mixed |
| [**GC03**](experiments/GC03_swerve_finetune/CARD.md) | Swerve finetune, 11 runs | Naren | neuroNCAP ncap_score 3.568 → **3.671** (n=20 scenario pairs) | ○○○○ unverified · blind | ⏸ parked (inconclusive) |
| [**GC04**](experiments/GC04_bench2drive/CARD.md) | Bench2Drive retrain (Split-v4, Final) | Navvrat | Bench2Drive mean driving score 57.60 → **73.6** (n=50 routes) | ●●○○ moderate · adapted, selected-on-report, no-record | ✅ positive |
| [**GC05**](experiments/GC05_physicalai_retrain/CARD.md) | Physical AI AV retrain | Siddesh | AlpaSim scene_score 0.5433 → **0.7697** (n=30 scenes) | ●●○○ moderate · adapted | ✅ positive |
| [**GC06**](experiments/GC06_scorer_cache/CARD.md) | Scorer on cached features | Naren | offline held-out collision ranking AUC, worst of 5 folds 0.59 → **0.660** (n=5 scenes) | ●●○○ moderate · blind, open-loop | ⏸ parked (inconclusive) |
| [**GC07**](experiments/GC07_subscore_scorer/CARD.md) | Sub-score scorer, seeing | Naren | neuroNCAP ncap_score, frontal 2.36 → **5.00** (n=3 runs) | ●●○○ moderate · smoke, adapted, no-record | ⏸ parked (mixed) |
| [**GC08**](experiments/GC08_expert_v6_imitation/CARD.md) | Expert v6 imitation | Naren | neuroNCAP ncap_score, mean of 3 shapes 3.01 → **2.31** (n=9 runs) | ●●○○ moderate · smoke, adapted, no-record | ❌ negative |
<!-- /experiments:gap-closing -->

## Hypothesis experiments

One structural change each, on the agreed base, decided by paddock's regression gate.

<!-- experiments:hypotheses -->
| ID | Hypothesis | Owner | Status | Headline result | Strength | Gate |
|---|---|---|---|---|---|---|
| [**H01**](experiments/H01_four_camera_base/CARD.md) | 4-camera base (WA-JEPA + DJ planner) | Navvrat | ⏳ running | NAVSIM navtest EPDMS, human filter 87.78 → **85.27** (n=12,146 scenes) | ●●○○ moderate · adapted, open-loop | — |
| [**H04**](experiments/H04_subscore_scorer_navsim/CARD.md) | Sub-score scorer, NAVSIM gap | Navvrat, Siddesh | ⏳ running | NAVSIM navtest EPDMS, human filter, locked pick 87.78 → **87.30** (n=12,146 scenes) | ●●○○ moderate · adapted, open-loop | — |
| [**H05**](experiments/H05_ssm_memory/CARD.md) | State-space memory | Naren, Siddesh | ⏸ parked (inconclusive) | offline human speed at +2 s, RMSE, events slice: camera history k = 4 vs k = 2 with the car state now only (control V3) 0.5001 → **0.4992** (n=3,302 moments) | ●●○○ moderate · open-loop | — |
| [**H06**](experiments/H06_global_read/CARD.md) | Global read and side braking | Navvrat | ⏳ running | NAVSIM score, global read off 0.870 → **0.852** (n=500 scenes) | ●●○○ moderate · no-record, open-loop | — |
| [**H07**](experiments/H07_encoder_lora/CARD.md) | Encoder LoRA, domain gap | Siddesh | ⏳ running | KITScenes pick ADE (3 s), m 2.372 → **1.879** (n=223 moments) | ●●○○ moderate · adapted, open-loop | ⛔ **FAIL** 28 Oct |
| [**H08**](experiments/H08_sideways_velocity_fov/CARD.md) | Sideways velocity and camera FOV | Naren | ⏳ running | KITScenes ADE (m), cropped 3.222 → **2.521** (n=223 keyframes) | ●●○○ moderate · open-loop | — |
| [**H14**](experiments/H14_aux_heads/CARD.md) | Aux heads on the refiner | Shreedhar | ⏳ running | nuScenes picked-plan L2 at 3 s, m 3.760 → **2.934** (n=2,996 keyframes) | ●●○○ moderate · adapted, open-loop | — |
| [**H15**](experiments/H15_prediction_memory/CARD.md) | Prediction memory | Naren, Siddesh | ⏳ running | offline P1: mean strict PDM grade of the pick, (a) carried L=4 0.86715 → **0.86827** (n=15,320 samples) | ●●○○ moderate · open-loop | ⏸ **HOLD** 6 Nov |
| [**H17**](experiments/H17_generator_scorer_memory/CARD.md) | Generator and scorer memory | Siddesh | ⏸ parked (negative) | offline mean grade of the picked path: memory carried vs reset every moment 0.7355 → **0.7313** (n=1,131 moments) | ●●○○ moderate · open-loop | — |
| [**H19**](experiments/H19_motion_history/CARD.md) | Car motion history | Siddesh | ⏸ parked (negative) | AlpaSim mean windowed score, 30 scenes, history shown (line written beforehand 0.8194, not met) 0.7527 → **0.7771** (n=30 scenes) | ●●○○ moderate | — |
<!-- /experiments:hypotheses -->

**Strength:** ●●●● proof · ●●●○ strong · ●●○○ moderate · ●○○○ argued · ○○○○ unverified.
**Nothing here is yet a closed-loop safety gain on a split nothing was chosen on,**
and `blind` results predate the 12 Oct camera fix.
[How to read these results](experiments/README.md#how-to-read-the-results).

Enablers, the gate log and the decision log are in
[experiments/README.md](experiments/README.md). These tables are generated from each
experiment's `experiment.toml`, and `make check` fails if they drift.

## Quickstart

```bash
make setup && source .venv/bin/activate     # .venv + install (no torch)
cp gallops.toml.example gallops.toml        # then point it at YOUR weights, runs and data
gallops config                              # what this machine resolved
gallops show <checkpoint dir>               # what a checkpoint claims about itself
```

Point `[roots.weights]` at the same directory as paddock's. Add the training stack with
`uv pip install --python .venv/bin/python -e ".[train]"`.

## Find your way

| Looking for | Go to |
|---|---|
| Every experiment, gate verdict and decision | [experiments/README.md](experiments/README.md) |
| How to run and record an experiment | [docs/experiments.md](docs/experiments.md) |
| Why this exists, what to build, how the repo is laid out | [docs/STRATEGY.md](docs/STRATEGY.md) |
| What gallops-v0 is | [docs/architecture.md](docs/architecture.md) |
| Writing or reading a checkpoint | [docs/checkpoints.md](docs/checkpoints.md) |
| Frames, splits, what counts as a result | [docs/conventions.md](docs/conventions.md) |
| What data exists | [docs/data.md](docs/data.md) |
| Reviewing a PR | [docs/reviewing.md](docs/reviewing.md) |

Apache-2.0, matching paddock.
