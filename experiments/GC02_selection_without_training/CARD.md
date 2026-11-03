<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# GC02 · Selection changes with no training: momentum, the model's own signals, injected swerves

**Naren, Siddesh** · AlpaSim, neuroNCAP · ◐ mixed · 23 Sep – 13 Oct 2026 · gap-closing

[← all experiments](../../README.md#gap-closing-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Evidence for:** D02 ([decision log](../../README.md#decision-log))

## Question

Drive-JEPA proposes 32 trajectories and a NAVSIM-trained scorer drives one. Before training anything: can a better choice among what it already proposes -- the paper's own momentum-aware selection, the model's other heads and logits, or extra swerves handed to its scorer -- close its gap on AlpaSim and neuroNCAP?

## What we learned

Momentum-aware selection lifted AlpaSim from 0.4209 to 0.5433 on 30 scenes, one drive each, and became driver 0.1.4b; on the camera-blind neuroNCAP set it changed 2.4% of picks for +0.050 (t 0.99, no measurable effect). No signal the blind model computes picked a clear plan on any of 322 rescue ticks, and its scorer refused injected swerves on all 434 frontal ticks, so the neuroNCAP gap was judged to need training.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **GC02.momentum-alpasim**<br>momentum-aware selection in the native AlpaSim driver: 0.1.4 → 0.1.4b | released | — | alpasim `30 scenes: the benchmark's first 10 (public_2601) plus 20 more` 30 scenes (eval) — one rollout per scene; scene ids in models_compare.json (pilot, ext) | — | — |
| **GC02.momentum-ncap**<br>momentum-aware selection on the full neuroNCAP set, separate adapter copy drivejepa_momentum | released | — | neuroncap `full official set: 20 scenario pairs × 10 seeds` 200 episodes (eval) | — | — |
| **GC02.signals**<br>the model's own signals as the picker: TTC logit, training-only collision head, a linear probe, the KITScenes re-ranker | released | — | — | — | — |
| **GC02.inject**<br>six synthetic swerves injected into the candidate set at the last refinement (±1.5, 2.5, 3.5 m) | released | — | — | — | — |

- **GC02.momentum-alpasim:** No training. The driver hands the model its previous plan; upstream's calibrate_score simulates each proposal and the previous plan forward and adds a flat bonus for two-frame comfort, (14·S + 2·S_c)/16. Driver work by Siddesh; the date of the 0.1.4b drives is not recorded in the sources. The AlpaSim driver is not camera-blind (no lidar in its chain; paddock docs/models/drivejepa.md).
- **GC02.momentum-ncap:** Run 20261002T065426-a87115 at site fastcode-shakaal, paired seed for seed with drivejepa 20260905T220225-f11ac0. Verified first: bit-identical scores to the vendor's calibrate_score on 1,242 replayed ticks. Both runs camera-blind (paddock #62). Paddock's own node got the stage in #59 (merged 13 Oct, off by default), which quotes this measurement, and declared it on in #95 (26 Oct).
- **GC02.signals:** Offline, over 32-candidate dumps of 7 pairs × 3 seeds from the camera-blind model, each run first reproducing the September sweep's plans and scores exactly. 'Clear' is an open-loop verdict: one candidate followed 4 s with no replanning. The rescue set is the 322 ticks where the driven plan hits although one of the 32 was clear. Every signal is dominated by plan length (the probe's risk correlates 0.97 with it).
- **GC02.inject:** Three frontal episodes. The plan comes from a clean first pass, so the episodes reproduce their recorded scores (2.171, 1.865, 1.747) and untouched slots score bit-identically (0.00e+00 drift over 509 ticks). The injected swerves enter before Bev_refiner, so the scorer reads real sampled features for them.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| GC02.momentum-alpasim ★ | AlpaSim scene_score 0.4209 → **0.5433** (n=30 scenes)<br>30 scenes, one drive each; at-fault collisions 7 → 4, any collision 12 → 10; no paired SE recorded. Constant velocity scores 0.4876 on the same 30 (one simulator setting differs) | released, driver 0.1.4 | ●●○○ moderate | experiments/GC05_physicalai_retrain/07_closed_loop_test/runs_017/tables/models_compare.json; paddock docs/models/drivejepa.md |
| GC02.momentum-ncap | neuroNCAP ncap_score 3.568 → 3.618 (n=20 scenario pairs)<br>full set, 20 pairs × 10 seeds, paired seed for seed; SE 0.050; t 0.99; read by its author as no measurable effect: the whole +0.050 is side-0108 seeds 1 and 8. Pick changed on 324 of 13,977 ticks (2.4%); collision rate 45.5% → 44.5% | drivejepa, run 20260905T220225-f11ac0 | ○○○○ unverified · blind | paddock branch record/drivejepa-camera-blind: docs/models/drivejepa_momentum.md, results/neuroncap/drivejepa_momentum/20261002T065426-a87115 |
| GC02.signals | neuroNCAP rescue ticks on which the collision head's safest pick is clear 0 (n=322 ticks)<br>7 pairs × 3 seeds, offline over candidate dumps; also 0 of 322 for the linear probe on proposal_feature and for the KITScenes re-ranker. The head does rank hitting candidates as riskier: 85–94% (stationary), 77–91% (frontal), blind on side | — | ●●○○ moderate · blind, open-loop | paddock branch analysis/drivejepa-ncap-candidates, docs/models/drivejepa-neuroncap.md (d737456, 66dddac) |
| GC02.signals | neuroNCAP rescue ticks with the TTC logit as the picker 17 (n=322 ticks)<br>7 pairs × 3 seeds, offline over candidate dumps; only by braking everywhere: it costs KITScenes ADE 0.97 → 3.96 | — | ●●○○ moderate · blind, open-loop | paddock branch analysis/drivejepa-ncap-candidates, docs/models/drivejepa-neuroncap.md (d737456) |
| GC02.inject | neuroNCAP frontal ticks on which an injected swerve is preferred over the driven plan 0 (n=434 ticks)<br>3 frontal pairs, one episode each; on the 248 of 314 hitting ticks where a swerve was clear, the best clear swerve scored 0.151 against 0.808 for the plan that hits. The swerves are far outside the scorer's training, so a low score may be unfamiliarity | — | ●○○○ argued · blind, smoke | paddock branch analysis/drivejepa-ncap-candidates, docs/models/drivejepa-neuroncap.md (66dddac) |

## Checks that decided something

- **Momentum adapter against the vendor's own code**: 1,242 recorded neuroNCAP ticks replayed through the adapter and through calibrate_score: scores bit-identical (max |diff| 0), picks matched; on 20 real nuScenes keyframes only the pick changed (1 tick) → the n=200 run measures the vendor's stage, not a re-implementation
- **What the 32 candidates hold before a crash (camera-blind)** (23 Sep): frontal: no swerve anywhere (widest 0.8–1.2 m sideways); stationary: a safe stop on 97–98% of ticks, ranked about 9th; side: a safe option on 79–98% of ticks, ranked 4th–8th → frontal cannot be fixed by choosing better; stationary and side could in principle

## Evidence outside this repo

- No gallops code: momentum is upstream's calibrate_score, run by paddock's AlpaSim driver (tools/models/drivejepa/alpasim_driver/, tracked at 0.1.7) and by the drivejepa node (paddock #59, on by default since #95).
- neuroNCAP momentum run 20261002T065426-a87115 and its separate adapter (drivejepa_momentum): paddock branch record/drivejepa-camera-blind, kept, not for merging.
- Signal and injection analyses: paddock branch analysis/drivejepa-ncap-candidates (no PR); the dump and injection adapters stay outside paddock; dumps under {runs}.
- AlpaSim 0.1.4b drives: {runs}/alpasim_bulk/baseline_014b/ (experiments/GC05_physicalai_retrain/POINTERS.md).
