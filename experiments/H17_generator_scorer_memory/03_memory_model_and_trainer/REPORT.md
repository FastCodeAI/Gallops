# Drive-JEPA memory, approach 3: build step 3, the memory model and the in-order trainer

shakaal, 2026-11-09 (22:03–22:18 IST). GPU used only for the tests below (step-0 replays, routing, 2 × 50 smoke steps). No pilot run, no checkpoint choice, no closed loop, no docker, no pushes. Writes only in this memory folder (`src/`, `out/`, `logs/`) and `{experiment_store}/runs/build3_tests/` (the step-0 dump). Both held-out slices were not opened.

Abbreviations as in `APPROACH3_RECON.md` (`G`, `RD`, `NAS`, `PAI`, `UP`, `GM`), `A3` = `{experiment_store}/`, `AG` = `RD/s10_retrain/train_patch/drive_jepa_agent_s10.py`. Plain names: encoder, generator, scorer, rating head, memory. Every number is read from a file named in the same row.

**Read first (verified):** `APPROACH3_RECON.md` C1–C5; `BUILD1_SLICES_CACHE.md`; `BUILD2_CACHE.md` (cache layout). Reused by import, never edited: `G/s1_build/src/pai_model.py`, `pai_data.py`, `G/s8_018b_build/src/pai_model_b.py`, `G/s9_018b_train/src/pai_model_c.py`, `train_018b.py`, `G/s2_train/src/pai_train.py`. **Starting model:** `NAS/018b/release/drive_jepa_018b_p06.ckpt`, sha256 `3d566eb2…8e66` verified inside every test before loading (`out/s6_*.json: agent.sha256`); 495 / 495 tensors re-compared after the strict load (`s6_model.load_agent`, = `s4_cache_sample.py:30-38`).

## Summary

| | result | files |
|---|---|---|
| new parameters | generator read 263,168, scorer read (summary) 263,424, rating-head new columns 262,144, update block 876,801; **total 1,665,537** (target ≈ 1.5 M, +11.0 %) | `out/s6_params.json` |
| T1 step-0 identity | carry **PASS**, wiped **PASS**: plans, logits and picks bit-identical to 0.1.8b's cached-token forward on all 2,591 moments of the 84 picking clips | `out/s6_t1.json` |
| 0.1.8b at step 0 (the check code) | pick grade (map, 1,131 moments) **0.7685**; closest-of-32 ADE (3 s) **0.238 m**; pick-follows-previous **29.2 %** (1,923 pairs); hard-brake picks above 10 m/s **0** / 745; pull-away go rate **84.0 %** (106) | `out/s6_t1.json: step0_018b` |
| T2 routing | **PASS** on a batch with 24 map rows (the chain's own fresh batch held no map moment, so grade / collision / drivable had no graph there: `out/s6_t2t3.json: t2_routing`) | `out/s6_t2_map.json: t2_routing` |
| T3 wiped | **PASS**: M zero at every read (0.0), update block gradient after 5 steps 27 of 27 tensors None, the rest all-zero, carry M after the first moment 33.16, 32.63, 31.66, 31.54, 30.55 | `out/s6_t2t3.json: t3_wiped` |
| T4 smoke (50 steps per mode) | losses falling: carry loss yes, generator yes, grade yes, human no, collision yes, drivable yes; wiped loss yes, generator yes, grade yes, human no, collision yes, drivable yes; NaN / inf: **no**; step 50 carry: M norm 64.04, gate mean 0.812; GPU peak 3.4 GB | `out/s6_t4.json` |
| T5 timing (steps 10–50) | **1.65 s / step** = forward 1.08 + grading wait 0.40 + backward 0.10; pass 4,288 steps with event over-sampling (3,839 without); **3 passes ≈ 7.9 h** (7.3 h without) | `out/s6_t4.json: t5` |

## The model

Starting point: 0.1.8b pass 0.6 = frozen encoder (cached, build 2) + generator + scorer (recon C1). Three new blocks, all 256-d (`src/s6_model.py`):

```
                     cached tokens (512 x 256, build 2)            ego row (11)
                              |                                        |
                              |                  hist_encoding(ego) + init_feature  -> 256 queries        (drive_jepa_model.py:61-62)
                              |                                        |
                              |            +---------------------------+-------------------------------+
                              |            |  GENERATOR READ: queries cross-attend into M (16 x 256), |
          M (16 x 256) ------>|----------->|  one attention layer, output projection ZERO-init,        |
          zeros at the start  |            |  result ADDED to the queries                              |
          of a window/clip    |            +---------------------------+-------------------------------+
                              |                                        |
                              +-------> refiner x 4 (traj_decoder + Bev_refiner, deformable cross-attention into the tokens)   (:64-67)
                                                                       |
                                                 proposals (32 x 8 x 3)  and  bev_feature (256 x 256)
                                                                       |
                                     proposal feature = max over the 8 poses (32 x 256)              (scorer.py:41)
                                                                       |
          M ----> SCORER READ: learned query attention-pools M -> summary (256)                         |
                                                                       |                               |
                                   RATING HEAD  Linear(256 -> 1024) widened to 512 inputs: W x + W_new summary + b   (W, b = 0.1.8b; W_new = 0)
                                   LayerNorm / ReLU / Linear / LayerNorm / ReLU / Linear(1024 -> 6)   (transformer_decoder.py:70-78)
                                                                       |
                                   pick = argmax sigmoid(last logit)                                   (drive_jepa_model.py:85, 90)
                                   collision head, drivable-area head: unchanged, no M                 (scorer.py:47-51)
                                                                       |
          UPDATE (after the pick):  M's 16 slots (+ a learned slot embedding on the query) attend to [32 image tokens = 4 x 8 block means of the token grid (probe.py:66-72,
                                    without the global mean); motion token (dx, dy, dheading since the previous moment, current frame);
                                    pick token (the scorer's argmax plan, 8 x 3, DETACHED)], then FFN -> candidate (16 x 256);
                                    gate per slot g = sigmoid(Linear([old; candidate])); M_new = (1 - g) old + g candidate
                                                                       |
          carry: M_new -> next moment          wiped: M = 0 before EVERY moment (same weights; mode flag in the checkpoint)
```

**Where the blocks sit, in the code.** `MemoryModel.forward_moment` (`src/s6_model.py`) is `UP/drive_jepa_model.py:50-51, 59-73, 85, 90-91` and `UP/score_module/scorer.py:36-42` re-done around the cached tokens: the tokens are handed to the refiner in the exact form `ImgEncoder.forward` returns (`(feat_flatten (1, 512, B, 256), spatial_shape (16, 32), level_start_index, {img_metas: lidar2img[:, 1:2], img_shape})`, `UP/bevformer/simple_image_encoder.py:64-66`; `encoder.py:123, 141-142` reads `lidar2img` and `img_shape`). `hist_encoding`, `init_feature`, the shared `Traj_refiner` and the scorer heads are the 0.1.8b modules themselves (0.1.7's batch fix installed, `RD/s10_retrain/train_patch/batch_fix.py`), called as the model calls them. The generator read is applied to the 256 queries after they are formed and before the first refiner pass; the scorer summary enters the rating head only.

**Rating head widening, exactly.** `scorer.pred_score.mlp[0]` is `Linear(256, 1024)`. The widened layer is implemented as `mlp[0] (x) + mem_in(summary)` with `mem_in = Linear(256, 1024, bias=False)` zero-initialised: the same linear map as one `Linear(512, 1024)` whose first 256 columns are 0.1.8b's and whose last 256 are zero. Two matmuls rather than one so that at step 0 the head is **bit-identical** to 0.1.8b (one 512-wide matmul with zero columns is a different cuBLAS accumulation order and can differ in the last bit); a checkpoint can export the single 512-wide weight by concatenation at any time. The rating head's parameter group in the routing table includes `mem_in` (flag 2).

**Update inputs, exactly.** Image tokens: `block_means(tokens)` = the (4 × 8) block means of the 16 × 32 grid of 256-d tokens, i.e. `GM/probe.py:66-72 pool_map` without its global mean (the cache's `pooled[:, 1:]` holds the same 32 vectors; computing them from the tokens costs nothing and keeps one input source). Motion: the previous moment's logged ego motion, read from the previous moment's human future: its waypoint 0 (t + 0.5 s = the current anchor) is the current pose in the previous frame (x₁, y₁, h₁); the motion in the current frame is (R(−h₁)(x₁, y₁), h₁) (`s6_model.motion_from_prev_future`); embedded as (dx, dy, cos h, sin h) → Linear(4 → 256). Consistency of that source: the previous future moved into the current frame against the current future, max mismatch 0.1313 m, 99th percentile 0.0341 m, median 0.00068 m over 2,494 consecutive pairs (`out/s6_t1.json: motion_source_check`). At the first moment of a window or clip the motion is zero (no previous moment, as at the car's first call). Pick: `proposals[argmax]` of the moment, detached, flattened (24) → Linear(24 → 256). The 34 keys carry a learned type embedding (image / motion / pick) and the 32 image tokens a learned position embedding. **Slot identity (not in the design as written, flag 4):** the attention query of slot i is `M_old[i] + slot_emb[i]`, a learned 16 × 256 table. Without it the first smoke run showed all 16 slots with the same gate value at every step: M starts at zeros, so every slot formed the same query, received the same candidate and the same gate, and the 16 slots stayed identical forever (one slot in effect). Everything else of the update is as specified.

**Parameter count** (`out/s6_params.json`, `MemoryModel.new_param_count`):

| block | parameters | what |
|---|---|---|
| generator read | 263,168 | q, k, v, o projections 256 → 256 (o zero-init) |
| scorer read: summary | 263,424 | learned query + q, k, v, o |
| rating head: new input columns | 262,144 | 256 × 1024, zero-init |
| update block | 876,801 | image / motion / pick embeddings + position / type embeddings + **slot embedding 16 × 256 (flag 4)**, attention, 2 LayerNorms, FFN 256 → 1024 → 256, gate 512 → 1 |
| **total new** | **1,665,537** | target ≈ 1.5 M (+11.0 %); memory state 16 × 256 = 4,096 values |

**Modes.** `MemoryModel.mode ∈ {carry, wiped}`; `step_memory` hands `M_new` on (carry) or zeros (wiped: zero before every moment); the trainer and the check zero M again at every window / clip start and whenever a clip's anchors are not consecutive. `save_checkpoint` writes the mode, the start model's sha and the parameter counts into the `.pt` and a `.json` beside it; `load_checkpoint` reads the mode from the file, never from an argument.

## The trainer (`src/s6_train.py`, `src/s6_data.py`)

**Data.** One read per clip of `A3/cache/<slice>/<clip>.pt` (tokens fp32, ego row, human future, flags; `BUILD2_CACHE.md`). The acceleration columns of the ego row (slots 5:7) are zeroed as the model's loader does (`RD/s3_5_smoke/train_patch/dataset_s35.py` `_finish`: "`features["ego_status"][..., 5:7] = 0.0`"; the cache kept the converter's values). Not in the cache: `lidar2img` (2, 4, 4) and `img_shape` (2, 3) are read from the slim npz of the clip's first moment (`PAI/s4_train/slim/<clip>/<sid>.npz`, keys written by `dataset_s35.py:190-196`); the metric cache and the anchor index of map moments from `PAI/s3_assembly/s1/cache/<clip>/<sid>/{metric_cache.pkl, anchors_scores_index_ext.npy}` (`dataset_s35.py _finish`; `AG:176-179 _index_path`). **`lidar2img` and `img_shape` are fixed per clip, and in fact identical across clips:** on 140 sampled clips (120 training, 20 picking; all 4,447 moments) both are constant within every clip and there is **one** distinct `lidar2img` across the 140 clips, `img_shape` (448, 768, 3) for both frames (`out/s6_lidar2img_constancy.json`, `logs/s6_lidar2img_constancy.log`). A sidecar is therefore possible and may even be a single constant (OPEN 3); the trainer reads one npz per clip for now.

**Windows.** 16 consecutive moments (8 s), anchors 0.5 s apart; 4 warm-up moments without loss, 12 scored. A window never crosses a missing anchor (`s6_data.runs_of`: maximal runs of consecutive anchors). Per clip and pass, each run of n ≥ 16 moments gives ⌊n / 16⌋ windows, each with its own random start inside the run (seeded by seed and pass); clips with no run of 16 are skipped: **57 of 4,015 training clips skipped, 3,958 used** (115 clips have a missing anchor inside, recon C3; runs are 32 long on most clips). Event over-sampling (flag, default on): a window holding any moment of `out/memory_moments_train.parquet` (6,511 moments) is drawn twice. Pass length: **4,288 steps of 2 windows (8,576 windows, 898 event windows duplicated) with over-sampling, 3,839 steps (7,678 windows) without** (`out/s6_t4.json: pass_plan`; 0.1.8b: 4,650 steps of 32 moments). Batch = 2 windows (32 moments), seeded shuffle; the next step's two clips are read in a thread while the step runs.

**Forward.** Moment by moment in time order: M → generator (with the read) → scorer (with the summary) → pick → update M; `M` is zeros, detached, at the window start; backprop through all 16 moments (the warm-up moments are in the graph through M; they carry no loss). bf16 autocast as 0.1.8b (`pai_model.py:210`); the check forward is fp32, batch 1.

**Losses, 0.1.8b attempt 2 unchanged, by import** (`NAS/018b/runs/018b_attempt2/run.json: settings`: `w_human 0.5, w_pull_away 2.0`):

| lesson | code | weight / rule |
|---|---|---|
| generator | `AG:384-424 generator_loss_s10` (winner-takes-all L1 + 0.5 × vocabulary term, stages with `prev_weight` 0.1; `inter_weight` 0) | 1 |
| grade (map moments) | `pai_model_c.Model018c.scorer_lessons` (`:44-87`), targets = the generator's own 32 plans graded online by `pai_model_b.Grader` (`:201-258`) | `final_score_weight` 1 |
| collision, drivable area (map moments) | same, S35 `score_loss` parts | `pred_ce_weight` 1 + `pred_l1_weight` 0.1; `pred_area_weight` 2 |
| human closeness (all moments) | `pai_model.scorer_loss` targets (`:252-259`) inside `scorer_lessons` | 0.5 |
| pull-away ×2 | `pai_model_c.pull_away` (`:29-34`): v < 1 m/s and human ≥ 1 m at 4 s | weighted means |
| hard-brake rule | `pai_model_b.hard_brake` (`:168-173`): grade target 0 for step1 < −6 m/s² at v > 10 m/s | — |

`scorer_lessons` is called on the 24 scored moments of a step stacked as rows (row = (t − 4) · 2 + window), so every lesson is the attempt-2 weighted mean **over scored moments**. Map moments are submitted to the grading pool moment by moment while the window runs and collected after the forward (`s6_train.window_step`); the hide-and-guess lesson of 0.1.8b has no place here (frozen encoder). Grader pool: 10 spawned processes on the launcher's two pinned cores (`src/run_model.sh`: `taskset -c 0-1 nice -n 19`).

**Replay routing (the choice).** As 0.1.8b (`pai_model.py:194-201`, `G/s8_018b_build/REPORT.md` G2), the scorer lessons read the rating, collision and drivable heads on a replay bev: the refiner from the same ego/query init and the same four proposal lists with the refiner weights detached. **How M enters the replay:** the replay's generator read runs with its weights detached *and* M detached (`MemoryModel.replay`, under `no_grad`), so through the replay bev the scorer lessons reach nothing trainable. They reach M, and so the update block, **only through the scorer's own summary**, which stays attached (`forward_moment`: `rating(bev_r, proposals.detach(), summary)`). Result: grade and human lessons → rating head + summary + update block; never the generator or its look-up (0.1.8b's scorer lessons never reached the generator either); collision and drivable → their heads only, as the inputs to those heads carry no gradient. The alternative, M attached through the detached look-up in the replay, would let the scorer lessons also shape what the *generator* reads from memory; OPEN 1.

**Optimisers.** Generator Adam 3e-5 (41 tensors, 2,381,123 values), scorer Adam 1e-4 (30 tensors) as `pai_model.py:233-235`; memory blocks Adam 1e-4. Seed recorded in every record and checkpoint (`seed 0` in all tests).

**Checks and rules (built, not run as a training run).** `run_check`: the 84 picking clips (cache, picking slice) **replayed in order from each clip's first moment** in the run's mode, fp32, batch 1 (2,591 moments; a non-consecutive anchor starts a new chain, none in these clips); measured on the 2,000 `in_picking` moments: pick grade on the 1,131 map moments (`Grader.grade_menus` on the moment's own menu, as `train_018b.run_check:192-216`), hard-brake picks above 10 m/s and pull-away go rate (`train_018b.safety:155-189`, `go_split:137-152`), closest-of-32 ADE (3 s) (`pai_train.per_moment:117-129` m1a), and the **pick-follows-previous rate**: the pick at t against the previous pick moved into the current frame with the logged motion; their shared 0.5–3.5 s window is the previous pick's waypoints 1–7 against the current pick's waypoints 0–6; ADE over those 7 poses (xy) < 0.5 m counts as "follows"; counted on in_picking moments whose previous moment is consecutive in the clip (1,923 pairs). Cadence every 930 steps and at the end of each pass; stop after 6 checks without a better choosable pick, at most 3 passes; choice = best pick grade among checks passing both safety gates: hard-brake picks above 10 m/s **= 0** (Sid's build-3 rule; 0.1.8b used "≤ step 0", flag 3) and pull-away go rate ≥ 0.66 (`train_018b.py:71`). Closest-of-32 ADE is reported against 0.1.8b's with the +0.02 m tolerance.

## Tests

### T1 step-0 identity (`out/s6_t1.json`, `logs/s6_t1.log`)

Reference: 0.1.8b's cached-token forward = `pai_model.forward_dump` (batch 1, fp32) with the encoder swapped for a module returning the cached tokens (`s5_cache_check.py`'s method), on every moment of the 84 picking clips in clip order (2,591 moments). Then the memory model, untrained (seed 0), in carry and in wiped mode, replayed in order from each clip's first moment.

| arm | moments | plans bit-identical | logits bit-identical | picks bit-identical | max plan diff (m) | max logit diff | M read, max |abs| | result |
|---|---|---|---|---|---|---|---|---|
| carry | 2,591 | yes | yes | yes | 0.0 | 0.0 | 3.81 | **PASS** |
| wiped | 2,591 | yes | yes | yes | 0.0 | 0.0 | 0.00 | **PASS** |

The check code at step 0 (= 0.1.8b's baseline on these measures; carry and wiped give the same numbers, `step0_identical_measures: yes`):

| measure | 0.1.8b at step 0 | source |
|---|---|---|
| pick grade, map moments (1,131) | **0.7685** | `step0_018b.pick_grade_mean_map` |
| closest-of-32 ADE (3 s), all 2,000 | **0.238 m** | `.closest_of_32_ade3_all` |
| scorer-pick ADE (3 s), all / < 15 m/s | 1.108 / 1.056 m | `.pick_ade3_all`, `.pick_ade3_slow15` |
| pick-follows-previous rate | **29.2 %** (1,923 pairs; ADE between the two picks mean 1.104 m, median 0.840 m) | `.follows_previous` |
| hard-brake picks above 10 m/s | **0** of 745 moments above 10 m/s | `.hard_picks_above10` |
| pull-away go rate | **84.0 %** (106 standstill moments where the human pulled away) | `.go_rate_pull_away` |

Hard-brake gate: 0.1.8b's step-0 count is 0; under Sid's build-3 rule (must be 0) the gate can be passed.

### T2 routing (`out/s6_t2_map.json: t2_routing`; `logs/s6_t2map.log`)

Carry mode, 5 warm-start steps at the real learning rates (so the zero-initialised layers are non-zero), then on a fresh batch (24 map rows of 24) one backward per lesson alone. "yes" = some gradient entry non-zero in the group.

| lesson | generator | generator look-up | rating head | scorer summary | collision head | drivable head | update block |
|---|---|---|---|---|---|---|---|
| generator | **yes** | **yes** | no | no | no | no | **yes** |
| grade | no | no | **yes** | **yes** | no | no | **yes** |
| human | no | no | **yes** | **yes** | no | no | **yes** |
| collision | no | no | no | no | **yes** | no | no |
| drivable | no | no | no | no | no | **yes** | no |

Expected (Sid): generator → generator, look-up, update block; grade and human → rating head, summary, update block, not the generator or the look-up; collision → collision head only; drivable → drivable head only. **PASS** 

### T3 wiped really is wiped (`out/s6_t2t3.json: t3_wiped`)

| check | value | result |
|---|---|---|
| wiped: M at every read is all zeros (5 steps + 1 = 96 window-moments) | max |M| read = 0.0 | pass |
| wiped: update block gradient after 5 steps (the summed lessons) | 27 of 27 tensors None, the rest all-zero | pass |
| carry: M non-zero after the first moment (norm, 5 steps) | 33.16, 32.63, 31.66, 31.54, 30.55 | pass |

### T4 smoke run (`out/s6_t4.json`, `logs/s6_t4.log`)

50 steps per mode, real learning rates, seed 0, event over-sampling on, the first 50 batches of pass 1 (same batches in both modes). Not saved as candidates.

carry:

| step | loss | generator | grade | human | collision | drivable | gate mean | M norm | grading wait s | step s |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 2.0999 | 0.4241 | 0.0000 | 1.6758 | 0.0000 | 0.0000 | 0.740 | 63.99 | 0.00 | 1.06 |
| 20 | 2.5185 | 0.7848 | 0.0000 | 1.7337 | 0.0000 | 0.0000 | 0.834 | 63.98 | 0.00 | 0.77 |
| 30 | 2.2078 | 0.4344 | 0.1830 | 1.5843 | 0.0061 | 0.0000 | 0.681 | 64.02 | 3.47 | 5.50 |
| 40 | 2.0943 | 0.4661 | 0.0000 | 1.6282 | 0.0000 | 0.0000 | 0.596 | 63.62 | 0.00 | 0.50 |
| 50 | 1.8821 | 0.2945 | 0.0000 | 1.5875 | 0.0000 | 0.0000 | 0.812 | 64.04 | 0.00 | 1.32 |

wiped:

| step | loss | generator | grade | human | collision | drivable | gate mean | M norm | grading wait s | step s |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 2.0956 | 0.3878 | 0.0000 | 1.7078 | 0.0000 | 0.0000 | 0.549 | 33.79 | 0.00 | 0.78 |
| 20 | 2.5248 | 0.7822 | 0.0000 | 1.7426 | 0.0000 | 0.0000 | 0.572 | 34.50 | 0.00 | 0.53 |
| 30 | 2.1096 | 0.4253 | 0.1577 | 1.5182 | 0.0083 | 0.0000 | 0.529 | 33.87 | 0.15 | 1.32 |
| 40 | 2.1696 | 0.5013 | 0.0000 | 1.6683 | 0.0000 | 0.0000 | 0.522 | 34.56 | 0.00 | 0.57 |
| 50 | 1.8762 | 0.2987 | 0.0000 | 1.5775 | 0.0000 | 0.0000 | 0.539 | 34.86 | 0.00 | 0.66 |

Mean of steps 1–10 vs 41–50: carry loss 2.424 → 2.307, generator 0.671 → 0.567, grade 0.116 → 0.068, human 1.629 → 1.671, collision 0.008 → 0.001, drivable 0.000 → 0.000; wiped loss 2.390 → 2.308, generator 0.645 → 0.576, grade 0.117 → 0.073, human 1.619 → 1.659, collision 0.007 → 0.001, drivable 0.000 → 0.000. Finite everywhere: carry yes, wiped yes. Gate mean per slot at step 50 (carry): 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81. Map rows per step 3.6 of 24 (whole-clip windows: 0, 12 or 24). GPU peak 3.4 GB.

Data wait (NAS read of the two clips, prefetched 4 batches ahead, excluded from the step time): mean 0.0 s, max 0.0 s per step in carry mode; GPU shared with other users' jobs during the run (`gpu_before`: 38897 MiB, 97887 MiB, 99 %).

### T5 timing (`out/s6_t4.json: t5`)

| | seconds per step (mean, steps 10–50) |
|---|---|
| forward (16 moments × 2 windows, incl. the 12 replays, grading submitted on the way) | 1.08 |
| grading wait after the forward (pool of 10 on cores 0–1) | 0.40 |
| backward + optimiser steps | 0.10 |
| **step** | **1.65** (wiped 1.65) |

Projection (check = 463 s at step 0 = the faster of the two in-order replays of the 84 clips + the measures incl. grading the 1,131 menus, `s6_t1.json: carry/wiped.replay_wall_s, step0_018b.time_s.measures`; checks every 930 steps and at the pass end):

| | steps / pass | checks / pass | training h / pass | wall h / pass | wall h / 3 passes |
|---|---|---|---|---|---|
| event over-sampling on | 4,288 | 5 | 1.96 | 2.60 | **7.9** |
| off | 3,839 | 5 | 1.76 | 2.40 | 7.3 |

Grading: the wait is 24.5 % of the step → grading does not dominate the step. A scaling estimate (not run; the pool is CPU-bound and its work splits over cores up to the 10 workers) puts the wait at ≈ 0 with 4 or 8 cores, i.e. the grading would fit under the forward and the step would be ≈ 1.2 s; the gain over 2 cores is the 0.40 s wait, about a quarter of the step.

## Flags

1. **Environment shim (`src/s6_env.py`).** On 2026-11-09 a root-disk relief move (another session's script, `{nas_store}/_move_log_20261109.txt`) replaced `Paddock/paddock-sid/Alpasim/analysis_3.0` with a symlink to the NAS. `s35_common.py` (read-only, inside that tree) locates the vendored navsim through its own resolved path, which then pointed at a NAS folder that does not exist, and every 0.1.8b import failed (`logs/s6_all_attempt1_save_dump_crash.log` is the second launch; the first died on this). The shim re-points the module constant at the local tree before `setup_env()` runs, in this process and in each grading worker; nothing under the earlier lanes was edited. While the move ran the NAS delivered 0.6–4 MB/s (18 MB test read in 4.2 s; the 1.27 GB start model took 572 s to read once), so the first T1 took 82 min and its check timing is not representative; after the move finished (16:18) the NAS read at 69 MB/s and all tests were rerun (`logs/s6_chain.log`).
2. **Rating-head widening is two matmuls, not one widened `Linear`** (same linear map; bit-identity at step 0, see "The model"). The routing table's "rating head" group holds the 10 `pred_score` tensors plus the new 256 × 1024 columns (11 tensors).
3. **Hard-brake gate rule.** Sid's build-3 rule is "must be 0"; 0.1.8b's trainer used "≤ step 0" (`train_018b.py:1281`). 0.1.8b-p06 has **0** hard-brake picks above 10 m/s at step 0 on the picking set (0.1.7 had 43, `run.json: reference_step0`), so both rules coincide at step 0 here; the trainer implements Sid's rule with the 0.1.8b rule selectable (`HARD_GATE_RULE`). OPEN 2.
4. **Slot embedding added to the update block (+4,096 parameters), a deviation from the design as written.** First smoke run (archived `out/attempt_before_slot_emb/s6_t4.json`): gate mean per slot at step 50 = 0.16 on all 16 slots in carry mode (0.58 in wiped), i.e. the slots never differentiated (zeros at the start + identical queries). With the slot embedding the 16 gates differ, if only slightly after 50 steps (0.811–0.813 at step 50 in carry mode, `s6_t4.json: carry.gate_mean_per_slot_step50`). OPEN 4.
5. **New parameters 1,665,537 against the ≈ 1.5 M target (+11.0 %).** The update block is 876,801 of them (FFN 256 → 1024 → 256 = 0.53 M). Within reach of the target by halving the FFN width (−0.26 M) if Sid wants it exact.
6. **Smoke-run trajectories are not bit-reproducible across launches.** Step 1 is identical in every run (loss 2.1986), but the killed run (`logs/s6_all_attempt2_killed_by_session_restart.log`) and the rerun before the slot embedding differ at step 50 (carry loss 1.8959 vs 1.8496): the deformable-attention backward uses atomic adds, as in 0.1.8b. T4/T5 numbers are from one run each.
7. **Windows and pass length.** 57 of 4,015 training clips have no run of 16 consecutive moments and are skipped; the pass has 4,288 steps with event over-sampling (898 of 7,678 windows duplicated) and 3,839 without, against 0.1.8b's 4,650 steps of 32 single moments; moments seen per pass 137,216 (with duplicates) vs 0.1.8b's 148,796 draws.
8. **Motion input source.** The car motion since the previous moment is taken from the previous moment's logged human future (waypoint 0). Consistency against the current moment's future: median mismatch 0.7 mm, 99th percentile 0.034 m, max 0.131 m on 2,494 consecutive pairs (converter interpolation). In the car this input is the simulated ego motion between calls; at a window's or chain's first moment it is zero.
9. **0.1.8b's pick is not steady from moment to moment**: the pick-follows-previous rate at step 0 is 29.2 % (1,923 pairs), the ADE between consecutive picks (moved into the same frame) has median 0.84 m and mean 1.10 m. Reported, not gated; it is the baseline the memory arms are compared against.
10. **Session restart killed the first smoke run** at wiped step 22 (14:45 IST; harness background jobs die with the session). Reruns were launched detached (`setsid nohup`). The human-closeness lesson does not fall within 50 steps in either mode (1.629 → 1.671 carry); the other four fall. 50 steps is a smoke test, not evidence either way.
11. **Grading share.** With the pool on the two pinned cores the grading wait is 0.40 s of a 1.65 s step (24.5 %); map rows average 3.6 of 24 per step (whole-clip windows give 0, 12 or 24). The 4- and 8-core figures in T5 are a scaling estimate, not a measurement.
12. **Idle grading-worker processes from an earlier session** (10 `drive-jepa` python processes started 2026-11-07, 0 % CPU, `ps`) are still alive on shakaal; not touched.

## OPEN for Sid

1. **Replay routing: M detached in the replay's generator read (built) vs attached.** Recommend: keep it detached — the scorer lessons then teach the update block only through the scorer's own summary, and the generator's use of memory is taught by the generator lesson alone (mirrors 0.1.8b, where scorer lessons never reached the generator; T2 table holds). Alternative: attach M in the replay's look-up (weights still detached), so grade / human lessons also shape what the generator reads from M; one line in `MemoryModel.replay`.
2. **Hard-brake gate.** Recommend: Sid's "must be 0" as built (0.1.8b-p06 is at 0 at step 0, so no check is lost by it). Alternative: 0.1.8b's "≤ step 0" (`HARD_GATE_RULE = "step0"`), identical here unless the count rises.
3. **Camera sidecar.** `lidar2img` and `img_shape` are constant within every sampled clip and identical across the 140 sampled clips. Recommend: one verification pass over all 4,015 + 84 clips' first and last moments (≈ 8,200 small NAS reads, no GPU) and then a single constant file under `A3/cache/` read once per run. Alternative: keep one npz read per clip (as now; negligible cost).
4. **Slot embedding (flag 4).** Recommend: accept the learned 16 × 256 slot embedding on the update query (+4,096 parameters) as part of the design. Alternative: a learned non-zero initial M instead of zeros (breaks the symmetry too, but then "zero at the start" is no longer literally true and wiped mode would reset to that learned state).
5. **Event over-sampling default.** Recommend: on (pass 4,288 steps, +12 % over 3,839; the memory moments are 5 % of the data). Alternative: off, for a pass closer to 0.1.8b's length.
6. **Parameter budget.** Recommend: keep 1,665,537 (+11 %). Alternative: FFN 256 → 512 → 256 in the update block → ≈ 1.40 M.

## Files

- `src/s6_model.py` (model), `src/s6_data.py` (cache reader, windows, prefetch), `src/s6_train.py` (step, lessons, replay routing, check, run), `src/s6_tests.py` (T1–T5), `src/s6_env.py` (environment shim, flag 1); launcher `src/run_model.sh` (2 pinned cores, nice 19)
- `out/s6_params.json`, `out/s6_t1.json`, `out/s6_t2_map.json` (T2), `out/s6_t2t3.json` (T3; T2 on a batch without map rows), `out/s6_t4.json` (T4, T5), `out/s6_lidar2img_constancy.json`; `out/attempt_before_slot_emb/` (the same five files from the run before the slot embedding, flag 4)
- `logs/s6_all.log` (params, T1, T2, T3, T4, T5, final run), `logs/s6_t2map.log`, `logs/s6_chain.log` (every launch and return code), `logs/s6_all_attempt1_save_dump_crash.log`, `logs/s6_all_attempt2_killed_by_session_restart.log`, `logs/s6_rest_before_slot_emb.log`, `logs/s6_lidar2img_constancy.log`
- `A3/runs/build3_tests/step0_018b_picking_dump.npz`, `step0_018b_grades.npy` (0.1.8b's step-0 plans, logits, picks and menu grades on the 2,591 picking-clip moments)
