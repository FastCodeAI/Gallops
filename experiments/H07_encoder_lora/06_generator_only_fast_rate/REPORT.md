# Phase 1: training only the generator on KIT, with the encoder frozen

Date 2026-10-17, shakaal.

**Terms.**
- **Generator:** the part of Drive-JEPA that proposes 32 candidate trajectories, the **menu**.
- **Scorer:** grades the 32 and picks one. It is frozen and not part of the result.
- **Encoder:** the image network, here the release encoder with the pass-3 LoRA merged in.
- **Moment:** one frame of one scene.
- **MAIN:** all 1,763 every-5th-frame moments of the 50 picking scenes (train scenes held out for decisions).
- **CONTINUITY:** the 103 baseline-rule moments inside MAIN.
- **Metric numbering:** as in the task and `../05_generator_survey/REPORT.md` §G. All use waypoints 1–6 (0.5–3 s).
  - 1a: the closest candidate's average distance error (ADE).
  - **1b: that same candidate's endpoint error at 3 s (the headline).**

## Summary

- **The run stopped after pass 1 because the variety guard fired.** Variety (metric 3) fell from 2.827 to 2.024 m, 28 % below the starting model; the limit was 10 %. The other guard did not fire: candidates within 2 m rose from 17.70 to 23.08.
- **The menu got much closer to the human path:**
  - MAIN 1b: 1.093 → **0.534 m** mean, 0.452 → 0.366 m median.
  - 1a: 0.617 → 0.317 m.
  - CONTINUITY 1b: 1.036 → 0.581 m.
  - Worst 20 % of moments: 1b 3.762 → 1.219 m.
- **Most of the mean gain comes from 82 moments at 15 m/s or faster** (4.7 % of MAIN). There the starting menu cannot reach the human's speed: 1b 11.78 → 1.88 m, all 82 better.
- **Below 15 m/s the gain is real but modest.**
  - 1b goes 0.572 → 0.469 m (median change −0.08 m): 794 moments improve by more than 10 cm and 432 get worse by more than 10 cm.
  - **This is where the variety went:** 2.82 → 1.84 m. At 15 m/s and above, variety rose (3.07 → 5.74).
- **Some slow turns lost their turning candidates.** The second-worst picture (`87537280` t = 135, a slow left turn) shows the menu collapsing to straight paths, with 1b going 0.14 → 2.63 m.
- **Best pass = 1**, the only pass. Both gates passed, after one rerun caused by a bug in my own check (deviation 2 below).
- Cost: 59 s of training plus 74 s of evaluation per pass; peak 3.7 GiB.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| Data: features and human path assembled from the frame cache plus per-scene calibration equal the full wrapper path on the 103 moments | **PASS**: 103/103 features and 103/103 human paths bit-identical. Calibration is the same at the first and last moment in 50/50 picking scenes, and constant across moments in 30/30 multi-moment scenes. The unused side-camera tensor is not assembled. | `out/data_build.json`, `src/kitdata.py` |
| G1: starting model reproduces `05_generator_survey/out/g_menu.json` on the 103 | **PASS, exact**: original and starting model, every number equal, using s0's own metric function. This stage's metric function gives identical values. | `out/passes.json` G1 |
| G2: 20-step smoke run | **PASS**. Details below. | `out/passes.json` G2 |

G2 details:
- Gradients:
  - All 41 trainable generator tensors had non-zero gradients at every step.
  - No gradient reached the encoder, the projector (the 1024→256 layer between encoder and generator), the scorer (all heads) or the 4 fixed grids.
  - There are no LoRA tensors to check, because the LoRA is merged.
- Loss on a fixed batch, measured without GridMask: 2.215 → 1.557.
- GridMask:
  - It was applied in 13 of 20 training calls.
  - At every one of those calls the encoder, projector and backbone were in eval mode, and GridMask's own training flag was on.
- Cost: peak 3.74 GiB allocated, 0.118 s/step.
- The generator was restored bit-exactly afterwards.

**During training:**
- GridMask was applied in 313 of 435 batches (72 %), always with the encoder in eval mode.
- Every evaluation asserts that GridMask never acts. The evaluation path is s2's `forward_dump`, which calls `agent.eval()`.

## Per pass (MAIN, mean / median; CONTINUITY means)

| Model | 1a closest ADE | 1b its FDE@3s | lowest FDE@3s | 2 count <1 m / <2 m (mean) | 3 variety | 4 worst-20 % 1a / 1b (mean) | 5 drift | 6 pick ADE | continuity 1a / 1b (mean) |
|---|---|---|---|---|---|---|---|---|---|
| original | 0.637 / 0.286 | **1.156 / 0.465** | 1.012 / 0.322 | 9.57 / 17.45 | 2.843 / 2.973 | 2.175 / 4.081 | 0.185 / 0.144 | 2.666 / 1.382 | 0.573 / 1.061 |
| start (pass 3) | 0.617 / 0.286 | **1.093 / 0.452** | 0.942 / 0.306 | 9.77 / 17.70 | 2.827 / 2.961 | 2.052 / 3.762 | 0.000 / 0.000 | 2.252 / 1.186 | 0.563 / 1.036 |
| pass 1 ★ | 0.317 / 0.242 | **0.534 / 0.366** | 0.361 / 0.250 | 13.52 / 23.08 | 2.024 / 1.727 | 0.677 / 1.219 | 1.993 / 1.178 | 1.509 / 0.997 | 0.323 / 0.581 |

- Training loss: 0.730 over the pass, 0.645 over the last 50 steps.
- The worst-20 % list is the 353 MAIN moments ranked by the original model's 1a. It is fixed once in `out/worst20_main.json`.
- Plot: `out/metrics_by_pass.png` (metrics 1b, 2 within 2 m, 3 and 5 against pass, with the starting model dashed). It has a single point per metric.

**Where the change in 1b sits** (pass 1 minus start, `out/best_vs_start.json`):

| Moments | n | 1b start → pass 1 (mean) | Median change | Better / worse by > 10 cm | Variety start → pass 1 |
|---|---|---|---|---|---|
| Below 15 m/s | 1,681 | 0.572 → 0.469 | −0.079 | 794 / 432 | 2.82 → 1.84 |
| 15 m/s and above | 82 | 11.78 → 1.88 | −9.61 | 82 / 0 | 3.07 → 5.74 |
| Turning (heading change > 0.3 rad at 4 s) | 450 | 1.596 → 0.677 | −0.141 | 247 / 108 | 2.66 → 1.97 |
| Straight-ish | 1,313 | 0.921 → 0.485 | −0.084 | 629 / 324 | 2.88 → 2.04 |
| Worst 20 % | 353 | 3.762 → 1.219 | −0.457 | 248 / 64 | — |

**Pictures** (top-down, forward is up, left is left; human path dashed, closest candidate red; one moment per scene):
- `out/menus_most_improved.png`: three high-speed or long-path moments where the starting menu fell short in distance.
- `out/menus_most_worsened.png`: two of three are turns where pass 1 kept fewer turning candidates. `87537280` t = 135 lost the whole left fan.
- `out/menus_worst20_examples.png`.

## Setup as run

- **Weights.** Release checkpoint (sha256 `740335c8…` asserted) with the pass-3 LoRA merged. This uses s0 `g0_gpu.build`, which is s3 `close.py` loading plus s2 `f_final.merge`.
- **Frozen vs trainable.** Encoder, projector and scorer are frozen. The generator is trainable (see deviation 1).
- **Loss.** Upstream `agent.trajectory_loss_anchors`, called unchanged with empty vocabulary lists. That is the human-path term only: winner takes all, pass weights 0.001 / 0.01 / 0.1 / 1.
- **Optimiser and run.** Adam, lr 1e-4, constant, no weight decay, no clipping. bf16 autocast for training, fp32 for evaluation. Seed 0. Batch 32 with the last partial batch dropped: 435 steps per pass. Reshuffled every pass.
- **Data.**
  - 13,937 training moments, using the `gt` driving command.
  - Images come live from the s2 frame cache through the frozen encoder.
  - The calibration features (`lidar2img`, `img_shape`) are computed once per scene by the upstream feature builder, using the adapter's own ground-height estimate.
  - Ego status and human path come from `build_sample` (`camera_frames=()`, so no images are read).
- **Files.**
  - Checkpoint: `{runs}/drivejepa_kitscenes/generator_joint_kit/s1/ckpt/pass_01.pt` (generator, optimiser, random-number states).
  - Dumps: `{runs}/drivejepa_kitscenes/generator_joint_kit/s1/dumps/{original,start,pass_01}.pkl`.
  - Total size 58 MB.
- **Held-out data.** The 223 held-out moments (val + overlap_train_val) and test were not read.

## Deviations from the task

1. **Trainable count is 2,381,123 (41 tensors), not 2,389,827.**
   - The asserted generator total is 2,389,827.
   - Of that, 8,704 values are four fixed BEV reference grids (`reference_points`, `ref_3d`, `hybird_ref_2d`, `ref_2d`). Upstream creates them with `requires_grad=False` (`bevformer/encoder.py:60-64`), so upstream never trains them.
   - Training them would move the geometry the attention samples from. I kept them frozen, and G2 checks they get no gradient.
2. **G2 ran twice.**
   - The first run failed only on my own counter: it counted the one evaluation-mode call made after the 20 steps, giving 21 GridMask calls against 20 expected.
   - Every gradient, loss and mode check passed in that run.
   - I moved the snapshot before that call and reran everything from scratch. G1 passed identically both times.
   - The first log is kept as `logs/phase1_attempt1_g2_counter_bug.log` and `out/passes_attempt1.json`. No criterion was loosened.
3. **Evaluation batch is fixed at 1.** That is the s0/s2 path, and the only batch size at which G1 can match s0 exactly. At other batch sizes the encoder output differs by up to 2e-3 (s0 §E).
4. **Metric 5 (drift) uses waypoints 1–6**, like the other metrics. The earlier drift tool used all 8.
5. **The G2 smoke run used its own shuffle seed (12345)**, and all seeds were reset to 0 before training. So the training stream is exactly "seed 0".
6. **Pictures take one moment per scene.** Unfiltered, all three worst-20 % examples came from the same high-speed scene.
7. **The stopping rule never got to act.** The guard stopped the run first, so best pass = 1 by default.

## Decisions for you (my recommendation first)

1. **Is the variety drop a defect or an expected tightening?**
   - The menu moved toward the human path. Within 2 m: +5.4 candidates; lowest 3 s error 0.94 → 0.36 m. That alone narrows the spread.
   - But slow turns lost options: 108 turning moments got worse by more than 10 cm.
   - *Recommend:* treat it as partly a defect. The loss gives gradient only to each moment's single closest candidate. Nothing keeps the other 31 spread out (upstream's spread term has weight 0).
2. **Next run.** *Recommend:* rerun with a lower learning rate (3e-5), evaluating every half-pass, with the same guards. That shows whether there is a point that keeps most of the below-15 m/s gain without the collapse. Alternatives:
   - (b) Continue from pass 1 with the variety guard turned into a flag only.
   - (c) Turn on upstream's spread term (`inter_weight`, 0 in the release config).
3. **Headline framing.** *Recommend:* report below-15 m/s and 15 m/s-and-above separately in every table. The 82 high-speed moments carry most of the mean improvement, because the starting menu's speed ceiling is around 15 m/s.

## Files

- `run.sh`: s0's interpreter, importing s1/s2/s0 sources unchanged.
- `src/kitdata.py`: per-scene calibration, moment table, data gate.
- `src/phase1.py`: preflight, G1, G2, training, per-pass evaluation and guards.
- `src/report.py`: table, plot, pictures, breakdowns.
- `src/wiring.py`: pre-launch smoke test.
- `out/`:
  - `passes.json` (gates and per-pass results), `data_build.json`
  - `worst20_main.json`, `table.md`, `best_vs_start.json`
  - `metrics_by_pass.png`, `menus_*.png`
  - `passes_attempt1.json`
- `logs/`: data build, both phase-1 attempts.
