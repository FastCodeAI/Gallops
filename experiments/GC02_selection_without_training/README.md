# GC02: selection changes with no training

Before any training, the team tried to close Drive-JEPA's gap by changing only
**which of its 32 candidate trajectories gets driven**. There were three tries:

- the paper's own momentum-aware selection;
- the model's other signals used as the picker;
- extra swerves handed to its scorer.

None of this code is in gallops. It all runs in paddock.

This page summarises its sources and adds no analysis. Every number below is
copied from them:

- paddock `docs/models/drivejepa.md` and `docs/ROADMAP.md`, on `main`;
- paddock branch `record/drivejepa-camera-blind`: `docs/models/drivejepa_momentum.md`,
  `docs/models/drivejepa-camera-blind.md`, and the run files they name;
- paddock branch `analysis/drivejepa-ncap-candidates`: `docs/models/drivejepa-neuroncap.md`,
  at commits `d737456` (23 Sep) and `66dddac` (26 Sep);
- [models_compare.json](../drivejepa_retrain/07_closed_loop_test/runs_017/tables/models_compare.json)
  in this repo, which holds the AlpaSim numbers.

**Every neuroNCAP number here is camera-blind.** Until paddock #62, cameras were
placed relative to nuScenes' LIDAR_TOP, so no image reached the model. A black
frame gave the same plan. These results describe a model that planned from ego
status alone.

## 1. Momentum-aware selection (Siddesh; neuroNCAP run at Naren's site)

This is a step from the Drive-JEPA paper, upstream's `calibrate_score`. Each
candidate is simulated forward next to the plan driven at the previous replan.
Candidates whose two-frame comfort is within NAVSIM's thresholds score higher:
the score is `(14·S + 2·S_c) / 16`, with `S_c` either 0 or 1. Paddock had left
the step off everywhere.

| Where | Without | With | Read as |
|---|---:|---:|---|
| AlpaSim, 30 scenes, one drive each (driver 0.1.4 → 0.1.4b) | 0.4209 | **0.5433** | at-fault collisions 7 → 4; no paired SE recorded |
| neuroNCAP, full set, 20 pairs × 10 seeds | 3.568 | 3.618 | +0.050, pair SE 0.050, t 0.99: "no measurable effect" |

- **On neuroNCAP:**
  - It changed the pick on 324 of 13,977 ticks (2.4%).
  - The whole gain is one pair, side-0108, where seeds 1 and 8 went from a
    collision to a clean pass.
  - A replay of 1,242 recorded ticks through the adapter scored bit-identically
    to the vendor's own code.
- **On AlpaSim:**
  - 0.1.4b became the driver that GC05's retrained checkpoint was measured against.
  - The AlpaSim driver was not camera-blind, so its number does not depend on #62.
- **In paddock's own node:** the step arrived in #59 (merged 13 Oct, off by
  default). It was declared on in #95 (26 Oct) because it is part of the
  published method. Paddock's ROADMAP says the neuroNCAP comparison has to be
  made again on the node that sees the camera.

## 2. The model's own signals as the picker (Naren)

These checks ran offline over the 32-candidate dumps of 7 pairs × 3 seeds. First,
what the candidates hold before a crash:

- **Frontal:** no swerve anywhere. The widest candidate moves 0.8–1.2 m sideways.
- **Stationary:** a safe stop on 97–98% of ticks.
- **Side:** a safe option on 79–98% of ticks.

Each signal was then judged on the 322 ticks where the driven plan hits although
one of the 32 was clear:

| Signal | Rescues |
|---|---|
| the scorer's TTC logit | 17 of 322, only by braking everywhere: KITScenes ADE 0.97 → 3.96 |
| `pred_col_agent`, the training-only collision head | 0 of 322 |
| a linear probe on the 256 features every scorer head reads | 0 of 322 |
| the KITScenes re-ranker ([GC01](../GC01_rerank_kitscenes/)) | 0 of 322; it never picks a stop |

Every signal is dominated by plan length. The probe's risk correlates 0.97 with
it, and its safest candidate is the shortest one on all 322 ticks.

## 3. Swerves injected into the candidate set (Naren)

Six swerves were added per tick across three frontal episodes, at ±1.5, 2.5 and
3.5 m. They entered at the last refinement, so the scorer read real features for
them. Driving was untouched: the episodes reproduced their recorded scores, with
0.00e+00 drift on the untouched slots.

The scorer preferred a swerve on **0 of 434** ticks. Where a swerve was clear,
the best clear one scored 0.151 against 0.808 for the plan that hits. The
analysis page states one caveat: these swerves are far outside anything the
scorer was trained on, so a low score may mean unfamiliarity rather than a
judgement about danger.

## What it led to

The analysis page closes on fine-tuning **both halves** of the model: proposing
swerves and choosing them. That needs actor-insertion data on scenes outside the
benchmark. This became the swerve finetune,
[GC03](../GC03_swerve_finetune/), from 26 Sep.
