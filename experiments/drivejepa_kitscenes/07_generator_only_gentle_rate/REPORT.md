# Phase 1 rerun: generator only, lower learning rate, with and without a stay-near-original term

Date 2026-10-17, shakaal.

**Terms.**
- **Generator:** the part of Drive-JEPA that proposes 32 candidate trajectories, the **menu**.
- **Scorer:** grades the menu and picks one. It stays frozen, like the encoder.
- **1b (headline):** the endpoint error at 3 s of the candidate closest to the human path. Closest means lowest average distance over 0.5–3 s.
- **Variety:** the mean distance between pairs of candidates.
- **Drift:** the mean distance of each candidate from the same candidate in the starting menu.
- **Stay-near-original term:** it pulls the 31 candidates that are not the winner back toward their positions in the starting menu. The weight on it is **w**.
- **Starting model:** the release weights with the pass-3 encoder LoRA merged.
- **Moment groups:** all are among the 1,763 picking moments.
  - **SLOW:** human speed below 15 m/s (1,681 moments).
  - **FAST:** 15 m/s and above (82 moments).
  - **SLOW TURNS:** SLOW moments where the heading changes by more than 0.3 rad at 3 s (342 moments, so no flag for fewer than 50).
- **Check:** a full evaluation on the 1,763 moments. One runs at step 0, after each half pass and at the end of each pass.

## Summary

- **Best run: w = 0 (no stay term), checkpoint at pass 9.5.** It has the lowest SLOW 1b of any check in any run. The turning check passed at all 20 of its checks.
- **The lower learning rate alone fixed the phase-1 problem as the brief measures it:**
  - SLOW 1b: 0.572 → **0.346 m** (phase 1: 0.469).
  - SLOW TURNS 1b: 0.804 → **0.535 m** (phase 1: 0.628).
  - FAST 1b: 11.78 → **1.31 m** (phase 1: 1.88).
  - On the 103 baseline moments: 1.036 → 0.403 m.
- **But w = 0 still narrows the slow menu, more than phase 1 did:** variety 2.82 → 1.56 m (phase 1: 1.84). The narrowing is gradual. 1b keeps improving while it happens, and the scorer's pick improves too: pick ADE 2.25 → 1.50 m.
- **The stay term keeps the spread but costs accuracy:**
  - w = 0.3: variety stays at 2.76 m; SLOW 1b 0.413; SLOW TURNS 0.596; FAST 3.77 (unstable, 1.7–5.9 across checks).
  - w = 1.0: SLOW 1b only 0.486.
  - It holds the 31 non-winners near the starting menu, so at high speed the menu cannot spread out to the faster paths.
- **Turns were not lost in any run on average.** The turning check passed at all 60 checks.
  - The phase-1 left turn (`87537280` t = 135) is still worse than at the start. 1b at the start was 0.14 m; at the best checkpoints it is 0.77 m (w = 0), 0.58 m (w = 0.3) and 0.53 m (w = 1). Phase 1 had 2.63 m.
  - The menu now keeps a left fan there.
- **Recommended start for phase 2: run w = 0, pass 9.5** (`{runs}/drivejepa_kitscenes/generator_joint_kit/s2/w0/ckpt/p09.5.pt`). If you want the menu spread protected, use w = 0.3, pass 8.5.
  - **Caveat:** these checkpoints were chosen on the same 1,763 moments they are scored on. 0.346 is the minimum of 20 checks whose neighbours read 0.36–0.37, so expect about 0.36 on fresh moments.

## Gates

| Gate | Result |
|---|---|
| **G1:** starting model reproduces | **PASS, exact.** On the 103: equal to `05_generator_survey/out/g_menu.json`. On the 1,763: equal to the 06_generator_only_fast_rate starting-model row. The 1,763 menus are bit-identical to the s1 starting dump. |
| **G2a:** recomputed winner gives upstream's human term | **PASS, exact** (`torch.equal`). Per refinement pass (4 of 4) and for the combined value (2.2823 with GridMask, 2.2148 without). |
| **G2b:** stay term is 0 at step 0 | **PASS, exactly 0.0** on all 4 passes, in training mode (GridMask applied) and in eval mode. |
| **G2c:** gradients | **PASS.** At step 0 (total loss at w = 1), and on the stay term alone after 5 smoke steps (value 0.243): all 41 trainable generator tensors get gradient. The encoder, projector, scorer, fixed grids and the frozen copy get none, and the frozen copy is unchanged. Peak 3.91 GiB, 0.135 s/step. Generator restored bit-exactly. |

- **GridMask:** applied in 303 of 435 training batches per pass, with the encoder in eval mode. It was never applied in evaluation.
- **Same data in all three runs:** each restarts from seed 0, so all three saw the same batches and the same masks (counts identical). Only w differs.

## Main table (1,763 picking moments; 1b as mean / median in m)

| Model | 1b ALL | 1b SLOW | 1b FAST | 1b SLOW TURNS | SLOW within 2 m | SLOW variety | menu drift | SLOW better / worse > 10 cm vs start | (scorer pick ADE) | 1b on the 103 |
|---|---|---|---|---|---|---|---|---|---|---|
| original model | 1.156 / 0.465 | 0.615 / 0.442 | 12.25 / 12.21 | 0.885 / 0.599 | 18.30 | 2.833 | 0.185 | 467 / 560 | 2.666 | 1.061 |
| starting model | 1.093 / 0.452 | 0.572 / 0.438 | 11.78 / 11.56 | 0.804 / 0.574 | 18.57 | 2.815 | 0 | — | 2.252 | 1.036 |
| phase 1 (lr 1e-4), pass 1 | 0.534 / 0.366 | 0.469 / 0.344 | 1.88 / 1.64 | 0.628 / 0.436 | 23.95 | 1.842 | 1.993 | 794 / 432 | 1.509 | 0.581 |
| **w = 0, pass 9.5** | **0.391 / 0.266** | **0.346 / 0.249** | **1.31 / 1.19** | **0.535 / 0.406** | 26.13 | 1.564 | 1.593 | 1000 / 244 | 1.496 | 0.403 |
| w = 0.3, pass 8.5 | 0.569 / 0.306 | 0.413 / 0.295 | 3.77 / 3.63 | 0.596 / 0.447 | 19.31 | 2.759 | 0.356 | 893 / 287 | 2.073 | 0.525 |
| w = 1.0, pass 8.0 | 0.569 / 0.364 | 0.486 / 0.346 | 2.28 / 1.98 | 0.724 / 0.509 | 18.70 | 2.809 | 0.132 | 634 / 190 | 2.140 | 0.524 |

**Slow turns, better / worse by more than 10 cm:**
- w = 0: 190 / 86
- w = 0.3: 177 / 80
- w = 1.0: 122 / 48

**Slow-turn variety:** 2.62 at the start; 1.66 (w = 0), 2.56 (w = 0.3), 2.62 (w = 1.0).

All 82 FAST moments improve in every run.

## Every check, per run (compact)

- None of the stopping rules fired: all three runs reached 10 passes.
- The full per-check tables are in `out/tables.md`; the per-check records are in `out/run_w*.json`.

| Run | SLOW 1b at pass 0.5 / 1 / 2 / 3 / 5 / 7 / 10 | SLOW-TURN 1b at the same checks | SLOW variety at pass 1 → 10 | Best |
|---|---|---|---|---|
| w = 0 | 0.439 / 0.412 / 0.401 / 0.399 / 0.371 / 0.364 / 0.360 | 0.687 / 0.650 / 0.613 / 0.591 / 0.576 / 0.576 / 0.561 | 2.48 → 1.64 | 9.5 (0.346) |
| w = 0.3 | 0.480 / 0.453 / 0.457 / 0.436 / 0.437 / 0.428 / 0.423 | 0.713 / 0.666 / 0.657 / 0.645 / 0.649 / 0.610 / 0.599 | 2.79 → 2.78 | 8.5 (0.413) |
| w = 1.0 | 0.512 / 0.518 / 0.505 / 0.509 / 0.498 / 0.486 / 0.490 | 0.756 / 0.763 / 0.750 / 0.757 / 0.715 / 0.717 / 0.713 | 2.82 → 2.81 | 8.0 (0.486) |

## Plots and pictures (`out/`)

- `plot_slow_1b_vs_variety.png`: SLOW 1b against SLOW variety, one line per run and one point per check. The starting model and phase-1 pass 1 are marked.
  - w = 0 moves down and to the left: more accurate and narrower.
  - w = 0.3 and w = 1.0 move straight down, keeping their variety.
- `plot_slow_turn_1b_by_progress.png`: SLOW-TURN 1b against passes, with the starting model dashed. Every point is below the line.
- Menu pictures: top-down, forward is up, human path dashed, closest candidate red. Columns: the starting model, then each run's best checkpoint.
  - `menus_left_turn_and_worst_slow_turns.png`: the phase-1 left turn, plus the 3 SLOW TURNS worst at phase-1 pass 1.
    - In the left turn, every run keeps a left fan. w = 0 has a tighter fan aimed slightly off.
    - Two of the three worst slow turns (`804115ba` t = 185, `8f20f94d` t = 50) stay unsolved in every run. The human turns harder than any candidate.
  - `menus_fast.png`: the 2 FAST moments with the largest starting error.
    - `e964e075` t = 115: 17.3 m → 0.23 m (w = 0); 1.11 m (w = 0.3).
    - `79fcee45` t = 315: 13.6 m → 1.71 m (w = 0); 9.5 m (w = 0.3). Here the stay term keeps the menu too slow.
  - `menus_most_improved_slow.png`: the 2 most improved SLOW moments under w = 0. 1b goes 4.3 → 0.58 m and 4.5 → 1.08 m. w = 0.3 gets about half of that and w = 1.0 almost none.

## Setup as run

- **Model.** As in 06_generator_only_fast_rate, imported unchanged:
  - Release checkpoint (sha256 asserted) with the pass-3 LoRA merged.
  - Encoder, projector and scorer frozen; 2,381,123 trainable generator values (41 tensors) with the fixed grids frozen.
  - GridMask on for training batches only; evaluation at batch 1 without GridMask.
- **Loss.** Upstream `trajectory_loss_anchors`, called unchanged with empty vocabulary lists, plus w × the stay term.
  - **Frozen copy:** a copy of `hist_encoding`, `init_feature` and the shared refiner, frozen at the starting weights. It runs without gradient on the encoder output caught from the same forward pass, so both generators see the same GridMask draw.
  - **Per refinement pass:** the winner is the argmin over the 32 of the mean over 8 waypoints of |dx| + |dy| + |dheading| to the human path. The other 31 use the same distance to the same-index starting candidate, averaged over the 31 and over the batch.
  - **Pass weights:** 0.001 / 0.01 / 0.1 / 1, combined the way upstream does it.
- **Optimiser and data.**
  - Adam, lr 3e-5, constant, no weight decay, no clipping; bf16 autocast; seed 0.
  - Batch 32 with drop_last: 435 steps per pass.
  - 13,937 training moments, `gt` command.
- **Time.** About 3.5 min per pass, including two checks; the three runs took 1 h 48 min in total.
- **Files.** `{runs}/drivejepa_kitscenes/generator_joint_kit/s2/`, 884 MB:
  - `w{0,0.3,1}/ckpt/pNN.N.pt`: the generator state at every check.
  - `w*/dumps/`: the menus at every check.
  - `dumps/start.pkl`.
- **Held-out data.** The 223 held-out moments (val + overlap_train_val) and test were not read.

## Deviations from the brief

1. **The step-0 check is the G1 evaluation, reused for all three runs** rather than re-run per run. Every run starts from the same weights, which G1 showed give bit-identical menus.
2. **The original-model row comes from 06_generator_only_fast_rate's saved dump of the original model**, not a new build. That dump came from the same evaluation path.
3. **The mid-pass check is at step 218 of 435**, because 435 is odd.
4. **"No improvement" counts against the best SLOW 1b so far, including the starting model.** It never came close to firing.
5. **The frozen copy runs in eval mode while the trained generator is in training mode.** I first checked that the generator has no dropout or batch-norm layers, so the two modes compute the same thing; G2b's exact 0 in training mode confirms it.
6. **Picture choices the brief left open:**
   - FAST: the two FAST moments with the largest starting error, from different scenes.
   - "Most improved SLOW": measured with the best run, w = 0.
   - The 3 worst slow turns exclude the left-turn scene, which is already shown.
7. **G2's smoke steps (5, shuffle seed 12345) are separate from training.** The generator was restored and the seeds reset to 0 afterwards.

## Decisions for you (my recommendation first)

1. **Which checkpoint should phase 2 start from?**
   - *Recommend:* w = 0, pass 9.5. It wins on every 1b column and on scorer pick error, and it passes the turning check.
   - The cost is a 44 % narrower slow menu. The metrics here can't tell whether that hurts, because they reward having one candidate near the human path.
   - Alternative: w = 0.3, pass 8.5, if you count menu spread as worth protecting for the scorer or for closed-loop driving. It gives up about 0.07 m of SLOW 1b and is unstable at speed.
2. **Is a narrow menu acceptable?**
   - *Recommend:* judge it in closed loop or with the scorer in the loop (phase 2), not by variety alone.
   - Variety fell at every check in the w = 0 run, while 1b and the scorer pick kept improving.
3. **Should the stay term be continued?**
   - *Recommend:* not in its current form. It protects the slow menu but also blocks the fast fan it should let move.
   - If spread matters, a version that applies only below 15 m/s, or a smaller w (about 0.1), would be the next thing to try.
