# Phase 2: generator and encoder LoRA trained together on KITScenes, with a generator-only control

Date 2026-10-19/20, shakaal.

**Terms** (same as the phase-1 rerun report):
- **Generator:** the part of Drive-JEPA that proposes the 32 candidate trajectories, called the **menu**.
- **Encoder LoRA:** the rank-16 low-rank adapter on the ViT-L camera encoder, from pass 3 of the encoder LoRA work. It has 192 tensors.
- **Hide-and-guess game:** the self-supervised task from that work. Patches of the camera image are hidden. A small **guesser** network predicts the hidden patches' features, and the targets come from an **answer key**: a slowly moving copy of the encoder (momentum 0.99925). The loss is unchanged (`players.game_loss`).
- **1b (headline):** the 3 s endpoint error of the candidate closest to the human path. "Closest" means the lowest average distance over 0.5–3 s.
- **Variety:** the mean distance between pairs of candidates.
- **Drift:** each candidate's mean distance from the same candidate in the pass-3 starting menu.
- **Groups:** all are among the 1,763 picking moments.
  - **SLOW:** below 15 m/s (1,681 moments).
  - **FAST:** 15 m/s and above (82 moments).
  - **SLOW TURNS:** SLOW moments whose heading changes by more than 0.3 rad by 3 s (342 moments).
- **Tails:** all three are on SLOW TURNS.
  - **T1:** the number of turns whose 1b got worse by more than 0.5 m.
  - **T2:** the largest single increase in 1b, with the moment it happened at.
  - **T3:** the mean 1b over the worst 10 % of turns (34 moments).
- **p9.5:** the best checkpoint of the phase-1 rerun (no stay term, pass 9.5). Both runs start from it.
- **Turning check:** a check passes when SLOW TURNS 1b is no worse than the run's own step 0 (0.5354 m).
- **Check:** a full evaluation on the 1,763 moments, with batch 1 and no GridMask. A pass has 435 steps. Checks run at step 0 and after steps 218 and 435 of each pass.

## Summary

- **Neither run meaningfully beats its start** (p9.5, SLOW 1b 0.346 m).
- **Control:** best at pass 0.5 with SLOW 1b **0.340 m**, which is 0.006 m better and smaller than its 0.01 m check-to-check swing. After that it slowly got worse and stopped at pass 3.5 after 6 checks without improving.
- **Phase 2 does not beat step 0.** Its best qualifying check is pass 1.0 at **0.355 m**. It stopped at pass 3.0 after two turning-check failures in a row. Pass 3.0 read 0.347 but failed the turning check.
- **Matched step for step,** phase 2 is equal to or worse than the control on SLOW (−0.011 to +0.032 m). It is slightly better on SLOW TURNS early on and worse later. FAST swings by up to ±0.9 m, which is noise on 82 moments.
- **The game barely moves:** its loss goes 0.4853 → 0.4845, and the encoder drifts by a cosine distance of only 0.03 → 0.06.
- **Phase 2's one gain** is on the 103 baseline moments: 0.369 m, against 0.388 for the control and 0.403 at the start. That is weak evidence: 103 moments and one checkpoint.
- **Tails, against p9.5:** the control adds 15 turns that are more than 0.5 m worse, and phase 2 adds 21.
- **Recommendation:** keep p9.5. Training the encoder together with the generator at these settings added nothing measurable.

## Gates

| Gate | Control | Phase 2 |
|---|---|---|
| **G1:** step 0 reproduces p9.5 | **PASS, exact.** All 1,763 menus are bit-identical to the p9.5 dump, and the 103-moment metrics are equal. | **PASS, exact** (same checks). |
| **G2:** gradient smoke (5 steps) | **PASS.** All 41 generator tensors get gradient (2,381,123 values). The encoder, projector, scorer and fixed grids get none. Peak 3.64 GiB. | **PASS.** 8,672,579 trainable values. The 41 generator tensors, all 192 LoRA tensors and 151 of 160 guesser tensors get gradient. The 9 without are the unused mask tokens 1–9, which is expected because only token 0 is used. The generator loss on its own reaches all 192 LoRA tensors. The answer key, projector, scorer and fixed grids get none. Peak 6.98 GiB. |
| **G3:** same batches in both runs | — | **PASS.** The first 20 batch hashes match, and so do all 1,305 steps the two runs have in common. Each hash covers the moment ids and the GridMask random state before and after the step. |

- **Fresh optimiser state:** both runs started from a fresh Adam state. No optimiser state from the phase-1 rerun was carried over.
- **GridMask:** both runs used the same draws, from a separately seeded stream. It was applied in 1,066 of 15,632 encoder calls in the control and 917 of 14,372 in phase 2. It was never applied in evaluation.

## Main table (1,763 picking moments; 1b as mean / median in m; T columns on SLOW TURNS)

| Model | 1b ALL mean / median | 1b SLOW | 1b FAST | 1b SLOW TURNS | SLOW within 2 m | SLOW variety | menu drift | SLOW better / worse > 10 cm vs pass-3 | (scorer pick ADE) | 1b on the 103 | T1 vs pass-3 / p9.5 | T2 (m) vs pass-3 / p9.5 | T3 (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original model | 1.156 / 0.465 | 0.615 / 0.442 | 12.254 / 12.209 | 0.885 / 0.599 | 18.30 | 2.833 | 0.185 | 467 / 560 | 2.666 | 1.061 | 32 / 97 | 2.76 / 5.81 | 3.047 |
| pass-3 start | 1.093 / 0.452 | 0.572 / 0.438 | 11.779 / 11.560 | 0.804 / 0.574 | 18.57 | 2.815 | 0.000 | 0 / 0 | 2.252 | 1.036 | 0 / 85 | 0.00 / 3.41 | 2.681 |
| p9.5 (phase-1 rerun best) | 0.391 / 0.266 | 0.346 / 0.249 | 1.305 / 1.193 | 0.535 / 0.406 | 26.13 | 1.564 | 1.593 | 1000 / 244 | 1.496 | 0.403 | 10 / 0 | 1.00 / 0.00 | 1.737 |
| control (generator only), best (pass 0.5) | 0.368 / 0.256 | 0.340 / 0.242 | 0.932 / 0.783 | 0.533 / 0.383 | 26.22 | 1.543 | 1.917 | 1009 / 230 | 1.553 | 0.388 | 15 / 15 | 1.03 / 1.28 | 1.884 |
| phase 2 (generator + LoRA), best (pass 1.0) | 0.405 / 0.263 | 0.355 / 0.253 | 1.436 / 1.318 | 0.531 / 0.406 | 26.37 | 1.532 | 1.674 | 978 / 229 | 1.374 | 0.369 | 15 / 21 | 0.90 / 1.32 | 1.663 |

## Matched: phase 2 minus control at the same step (negative = phase 2 better)

| pass | SLOW 1b | SLOW-TURN 1b | FAST 1b | SLOW variety | T1 vs p9.5 | T3 |
|---|---|---|---|---|---|---|
| 0.5 | +0.032 | -0.020 | +0.266 | -0.024 | +5 | -0.134 |
| 1.0 | +0.001 | -0.013 | -0.046 | +0.017 | +1 | +0.010 |
| 1.5 | +0.021 | +0.019 | -0.214 | +0.062 | +16 | +0.044 |
| 2.0 | +0.022 | +0.004 | +0.206 | -0.057 | +2 | -0.057 |
| 2.5 | -0.001 | +0.011 | -0.873 | +0.127 | +9 | +0.042 |
| 3.0 | -0.011 | +0.016 | -0.137 | -0.042 | +3 | +0.124 |

## Every check, per run

### control (generator only): no SLOW 1b improvement for 6 checks (pass 3.5); best pass 0.5

| pass | SLOW 1b | SLOW-TURN 1b | turn ok | FAST 1b | ALL 1b | SLOW var | T1 vs p9.5 | T2 vs p9.5 | T3 | 
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.346 | 0.535 | yes | 1.30 | 0.391 | 1.564 | 0 | 0.00 | 1.737 |
| 0.5 | 0.340 | 0.533 | yes | 0.93 | 0.368 | 1.543 | 15 | 1.28 | 1.884 |
| 1.0 | 0.355 | 0.544 | NO | 1.48 | 0.407 | 1.515 | 20 | 1.12 | 1.653 |
| 1.5 | 0.347 | 0.532 | yes | 1.35 | 0.394 | 1.484 | 17 | 1.04 | 1.734 |
| 2.0 | 0.351 | 0.528 | yes | 0.83 | 0.373 | 1.609 | 18 | 1.31 | 1.712 |
| 2.5 | 0.356 | 0.537 | NO | 1.87 | 0.426 | 1.416 | 13 | 1.18 | 1.749 |
| 3.0 | 0.357 | 0.528 | yes | 1.07 | 0.391 | 1.459 | 17 | 0.87 | 1.667 |
| 3.5 | 0.358 | 0.558 | NO | 1.36 | 0.405 | 1.302 | 22 | 1.07 | 1.826 |

### phase 2 (generator + LoRA): turning check failed at 2 checks in a row (pass 3.0); best pass 1.0

| pass | SLOW 1b | SLOW-TURN 1b | turn ok | FAST 1b | ALL 1b | SLOW var | T1 vs p9.5 | T2 vs p9.5 | T3 | encoder drift (cos) | game loss (picking pairs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.346 | 0.535 | yes | 1.30 | 0.391 | 1.564 | 0 | 0.00 | 1.737 | 0.00000 | 0.48534 |
| 0.5 | 0.373 | 0.513 | yes | 1.20 | 0.411 | 1.519 | 20 | 1.05 | 1.749 | 0.03067 | 0.48720 |
| 1.0 | 0.355 | 0.531 | yes | 1.44 | 0.405 | 1.532 | 21 | 1.32 | 1.663 | 0.03936 | 0.48690 |
| 1.5 | 0.367 | 0.551 | NO | 1.14 | 0.403 | 1.546 | 33 | 1.40 | 1.777 | 0.04392 | 0.48665 |
| 2.0 | 0.373 | 0.533 | yes | 1.03 | 0.404 | 1.553 | 20 | 1.42 | 1.655 | 0.05471 | 0.48533 |
| 2.5 | 0.355 | 0.548 | NO | 1.00 | 0.385 | 1.543 | 22 | 1.20 | 1.792 | 0.04931 | 0.48477 |
| 3.0 | 0.347 | 0.544 | NO | 0.94 | 0.374 | 1.417 | 20 | 1.58 | 1.791 | 0.06160 | 0.48445 |


## Plots

- `out/plot_slow_1b_by_progress.png`: SLOW 1b and SLOW-TURN 1b at each check, for both runs. The dashed line is p9.5.
- `out/plot_phase2_minus_control.png`: phase 2 minus the control at each step. SLOW and SLOW TURNS are in one panel, FAST is separate.
- `out/plot_slow_1b_vs_variety.png`: SLOW 1b against SLOW variety, one point per check. A star marks each run's best.

What they show:
- The control's best check is the first one. Every later check sits at or above the start.
- Phase 2 zig-zags between 0.347 and 0.373 m and never settles below the start.
- Neither run narrows the menu much further: SLOW variety stays at 1.42–1.61 m, against 1.56 at the start. The one exception is the control's last check, at 1.30.

## Menu pictures (p9.5 | control best | phase 2 best)

Grey lines are the 32 candidates. Red is the closest candidate. The dashed line is the human path.

`out/menus_left_turn_and_largest_slow_turn_increases.png`:

| Moment | p9.5 | Control | Phase 2 |
|---|---|---|---|
| Left turn `87537280` t = 135 | 0.77 | **0.20** | 0.47 |
| Largest T2 `f79ac82f` t = 95 | **0.35** | 0.83 | 1.67 |
| Largest T2 `3e45d731` t = 85 | 0.88 | **0.60** | 1.84 |
| Largest T2 `505d91cf` t = 80 | **0.40** | 0.47 | 1.22 |

- The left turn improves in both runs, and the menu keeps its left fan.
- On the three moments where phase 2's turns got worst, its menu still fans the right way, but it falls short of the human path's curvature.

`out/menus_fast.png`:

| Moment | p9.5 | Control | Phase 2 |
|---|---|---|---|
| `e964e075` t = 115 | **0.23** | 1.28 | 2.45 |
| `79fcee45` t = 315 | **1.71** | 2.09 | 2.19 |

- On both moments the menus stop short of the human path. These two are worse than p9.5 in both runs, even though the FAST mean improved in the control.

`out/menus_most_improved_slow.png`:

| Moment | p9.5 | Control | Phase 2 |
|---|---|---|---|
| `4a1ada44` t = 40 | 1.77 | 1.86 | **0.15** |
| `8f20f94d` t = 40 | 2.34 | 1.66 | **0.77** |

- These are phase 2's largest gains.
- `4a1ada44` gains because one candidate goes straight through the gentle left. `8f20f94d` gains because the right-turn fan tightens toward the human path.

The pictured moments and their exact values are in `out/pictures.json`.

## Phase-1 rerun, tails backfilled for every check

T1 and T2 are counted against pass 3 and against p9.5. The ★ marks each run's chosen best. At step 0, T2 is 0 and its moment label is arbitrary. The same data is in `out/backfill_s2_tails.json`.

| pass | w0: SLOW 1b · T1 (vs pass-3 / p9.5) · T2 vs pass-3 (m) · T3 | w0.3: SLOW 1b · T1 (vs pass-3 / p9.5) · T2 vs pass-3 (m) · T3 | w1: SLOW 1b · T1 (vs pass-3 / p9.5) · T2 vs pass-3 (m) · T3 |
|---|---|---|---|
| 0.0 | 0.572 · 0 / 85 · 0.00 (f79ac82f t=110) · 2.681 | 0.572 · 0 / 85 · 0.00 (f79ac82f t=110) · 2.681 | 0.572 · 0 / 85 · 0.00 (f79ac82f t=110) · 2.681 |
| 0.5 | 0.439 · 17 / 58 · 1.07 (2498d199 t=215) · 2.129 | 0.480 · 8 / 61 · 1.43 (f79ac82f t=80) · 2.487 | 0.512 · 3 / 74 · 1.21 (f79ac82f t=80) · 2.669 |
| 1.0 | 0.412 · 22 / 47 · 1.28 (8f20f94d t=60) · 2.195 | 0.453 · 4 / 55 · 1.07 (f79ac82f t=90) · 2.307 | 0.518 · 3 / 70 · 1.54 (f79ac82f t=80) · 2.657 |
| 1.5 | 0.411 · 22 / 46 · 1.33 (8f20f94d t=60) · 2.153 | 0.449 · 9 / 56 · 1.42 (f79ac82f t=80) · 2.436 | 0.510 · 2 / 73 · 1.05 (0d99e104 t=180) · 2.538 |
| 2.0 | 0.401 · 21 / 36 · 1.29 (ab9ebedc t=245) · 2.029 | 0.457 · 7 / 48 · 0.88 (f79ac82f t=90) · 2.290 | 0.505 · 4 / 71 · 0.89 (2498d199 t=200) · 2.638 |
| 2.5 | 0.381 · 11 / 28 · 0.85 (ab9ebedc t=245) · 1.944 | 0.448 · 12 / 46 · 0.97 (8f20f94d t=50) · 2.274 | 0.509 · 4 / 72 · 1.20 (f79ac82f t=80) · 2.586 |
| 3.0 | 0.399 · 14 / 30 · 1.65 (ab9ebedc t=245) · 1.887 | 0.436 · 7 / 47 · 0.69 (8f20f94d t=60) · 2.126 | 0.509 · 5 / 69 · 1.30 (f79ac82f t=80) · 2.650 |
| 3.5 | 0.382 · 13 / 34 · 1.33 (ab9ebedc t=245) · 1.940 | 0.445 · 9 / 48 · 0.93 (0d99e104 t=65) · 2.184 | 0.501 · 2 / 67 · 0.60 (c5679dbf t=135) · 2.550 |
| 4.0 | 0.384 · 25 / 37 · 1.05 (87537280 t=135) · 2.029 | 0.439 · 9 / 49 · 0.69 (174d46e1 t=25) · 2.127 | 0.493 · 5 / 67 · 0.62 (804115ba t=185) · 2.477 |
| 4.5 | 0.368 · 11 / 30 · 1.23 (ab9ebedc t=245) · 1.902 | 0.436 · 10 / 33 · 0.92 (0d99e104 t=65) · 2.044 | 0.494 · 1 / 64 · 0.66 (804115ba t=185) · 2.471 |
| 5.0 | 0.371 · 13 / 28 · 1.11 (0d99e104 t=65) · 1.884 | 0.437 · 9 / 48 · 1.48 (f79ac82f t=80) · 2.228 | 0.498 · 6 / 64 · 0.86 (4a1ada44 t=50) · 2.548 |
| 5.5 | 0.376 · 22 / 31 · 1.42 (f79ac82f t=90) · 2.028 | 0.435 · 9 / 36 · 1.02 (8f20f94d t=50) · 2.042 | 0.486 · 2 / 63 · 0.88 (4a1ada44 t=50) · 2.436 |
| 6.0 | 0.362 · 15 / 31 · 1.34 (8f20f94d t=60) · 1.958 | 0.422 · 7 / 37 · 0.79 (0d99e104 t=65) · 1.998 | 0.487 · 3 / 64 · 0.72 (c562bd58 t=75) · 2.388 |
| 6.5 | 0.367 · 20 / 22 · 1.00 (8f20f94d t=60) · 1.885 | 0.430 · 9 / 39 · 0.85 (c562bd58 t=75) · 1.965 | 0.496 · 6 / 68 · 0.73 (4a1ada44 t=50) · 2.447 |
| 7.0 | 0.364 · 20 / 27 · 1.29 (8f20f94d t=60) · 1.943 | 0.428 · 11 / 27 · 1.13 (8f20f94d t=60) · 1.991 | 0.486 · 3 / 69 · 0.87 (4a1ada44 t=50) · 2.481 |
| 7.5 | 0.373 · 16 / 33 · 1.58 (e30d98f0 t=200) · 2.025 | 0.423 · 5 / 35 · 0.78 (f79ac82f t=80) · 2.035 | 0.491 · 7 / 64 · 0.87 (4a1ada44 t=50) · 2.469 |
| 8.0 | 0.369 · 13 / 24 · 1.19 (ab9ebedc t=245) · 1.943 | 0.416 · 8 / 30 · 0.72 (0d99e104 t=105) · 2.003 | 0.486 ★ · 5 / 67 · 0.77 (4a1ada44 t=50) · 2.479 |
| 8.5 | 0.366 · 17 / 19 · 0.99 (f79ac82f t=80) · 1.890 | 0.413 ★ · 7 / 33 · 0.75 (e7e0844b t=75) · 1.970 | 0.496 · 5 / 68 · 0.85 (4a1ada44 t=50) · 2.510 |
| 9.0 | 0.360 · 9 / 12 · 1.14 (ab9ebedc t=245) · 1.849 | 0.427 · 9 / 32 · 1.04 (0d99e104 t=65) · 1.967 | 0.488 · 4 / 60 · 0.91 (4a1ada44 t=50) · 2.483 |
| 9.5 | 0.346 ★ · 10 / 0 · 1.00 (8f20f94d t=60) · 1.737 | 0.433 · 5 / 34 · 0.63 (804115ba t=25) · 2.070 | 0.493 · 10 / 62 · 0.88 (4a1ada44 t=50) · 2.474 |
| 10.0 | 0.360 · 10 / 25 · 1.18 (f79ac82f t=90) · 1.748 | 0.423 · 7 / 33 · 0.76 (0d99e104 t=65) · 1.948 | 0.490 · 2 / 65 · 0.75 (4a1ada44 t=50) · 2.460 |

- **T3 falls steadily with training at w = 0:** 2.68 → 1.74 m at pass 9.5. The stay term slows that fall.
- **The moments behind T2 repeat:** `f79ac82f`, `8f20f94d` and `4a1ada44` again and again. These are the same turns that sit at the top of this phase's pictures.

## Deviations and choices not fixed by the brief

1. **Fresh optimiser state** in both runs, as the brief asked. This is stated here because both runs therefore restart Adam's moment estimates.
2. **Two optimisers in phase 2,** both stepped after one shared backward pass:
   - Adam for the generator, lr 3e-5, the same as the control.
   - AdamW for the LoRA and the guesser, at lr 1e-4. The LoRA has weight decay 0. The guesser has weight decay 0.04 on weights and 0 on biases and norms, as in the encoder LoRA work.
   - Both learning rates are constant. The encoder LoRA work used a warm-up ramp; it was not used here.
3. **The encoder stays in eval mode while training.** It has no dropout, so the only effect is that GridMask is driven explicitly, in the same way as in the control.
4. **The 9 unused guesser mask tokens get no gradient.** This is by design (only token 0 is used), and G2 allows exactly those 9.
5. **Drift at step 0 is set to 0.** The reference is the pass-3 encoder's projector output, which phase 2 starts from exactly.
6. **The game pairs come from a separate stream.** Every 5th frame of the 482 train scenes, 32 pairs per batch, shuffled per pass with its own seeds, so the planner batches stay identical to the control's.
7. **Game loss is measured** on the fixed set of 2,000 picking-scene pairs from the encoder LoRA work (fixed masks, seed 20261016).
8. **The planner shuffles restart from seed 0,** so both runs see batches in the same order as pass 1 of the phase-1 rerun.
9. **Phase 2 ran twice:**
   - The first attempt stalled in swap at step 100 of pass 1, because the shared host's swap was full. Its log is kept as `logs/phase2_attempt1_swap_stall.log`.
   - The rerun used spawned, long-lived data loaders and a smaller selection object. It repeated G1–G3, and all passed.
   - The control ran once, with forked loaders. Both kinds of loader yield the same batches, which G3 confirms.
   - The rerun printed "done" at 03:53, then hung on disk I/O while shutting down. All outputs were already written, so I stopped it.
10. **The control trains the LoRA-merged encoder exactly as the phase-1 rerun did,** so step 0 matches p9.5 bit for bit. In phase 2 the LoRA is unmerged for training and merged into a copy for each evaluation.
11. **Selection caveat:** as before, the best checkpoint is chosen on the same 1,763 moments it is scored on. Each run's checks swing by 0.01–0.03 m, so 0.340 against 0.346 is not a real gain.

## Files

- **Code:**
  - `src/phase2.py`: both runs and the gates.
  - `src/p2report.py`: the tables, plots and pictures.
  - `src/wiring.py`: the pre-launch smoke test.
  - Run with `./run.sh`.
- **Logs:**
  - `logs/phase2_attempt1_swap_stall.log`: the full control run and phase-2 attempt 1.
  - `logs/phase2.log`: the phase-2 rerun.
  - `logs/report.log`: the report build.
- **Results:**
  - `out/run_{control,phase2}.json`: every check.
  - `out/hashes_*.json`: the per-step hashes.
  - `out/state.json`: gates, stop reasons and the best checkpoints.
  - `out/tables.md`: the generated tables.
- **Checkpoints and dumps:** `{runs}/drivejepa_kitscenes/generator_joint_kit/s3/{control,phase2}/{ckpt,dumps}/`. Control best: `control/ckpt/p00.5.pt`. Phase 2 best: `phase2/ckpt/p01.0.pt`.
- **Held-out data:** none of the 223 held-out moments (val + overlap_train_val) or test was read.
