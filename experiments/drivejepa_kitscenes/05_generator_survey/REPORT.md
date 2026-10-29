# Generator on KIT: read-only check before training

Date 2026-10-17, shakaal. Nothing was trained and no weights were saved. No tracked file, Gallops branch file, experiments/drivejepa_kitscenes file or vendored file was edited. The 223 held-out moments (val + overlap_train_val) and test were **not read**. About 6 min of GPU time in total.

**Folder.** The brief's `experiments/drivejepa_kitscenes/` no longer exists. It was renamed to `experiments/drivejepa_kitscenes/` (holding `experiments/drivejepa_kitscenes/`), and an empty `experiments/drivejepa_kitscenes/` was created at 00:38 the same night. This work therefore sits in `experiments/drivejepa_kitscenes/05_generator_survey/`. Large outputs go to `{runs}/drivejepa_kitscenes/generator_joint_kit/s0/`, which is empty because nothing large was produced.

**Terms.**
- **Generator:** the part of the planner that proposes the 32 candidate trajectories.
- **Scorer:** grades the 32 candidates and picks one.
- **Menu:** the 32 candidates.
- **Moment:** one keyframe of one scene.
- **Picking set:** the 50 train scenes held out for checkpoint choice. With the baseline keyframe rule it gives 103 moments.
- **ADE:** average distance error over 0–3 s. **FDE@3s:** distance error at 3 s.

Shorthand for paths:
- `DEVKIT` = `{drivejepa_repo}/navsim_v2`. This is upstream `e21f474`, the code that ran every KIT evaluation.
- `AG` = `DEVKIT/navsim/agents/drive_jepa_perception_based/`.
- `ADP` = `DEVKIT/scripts/inference/kitscenes_adapter.py`; `RUN` = `…/run_kitscenes_inference.py`.
- `ENC` = `experiments/drivejepa_kitscenes/`.

The vendored Paddock copy of the four model/agent files is byte-identical to `DEVKIT` (`diff -q`, no output).

## Summary

- **Confirmed:**
  - The upstream generator loss is winner-takes-all over the 32 candidates, on all 4 refinement passes, with weights 0.001 / 0.01 / 0.1 / 1 (§A).
  - Both of our KIT models use the unmodified upstream release for the generator and scorer. The October Physical AI AV retrain is not loaded anywhere (§B).
  - The generator has 2.39 M parameters. The projector sits outside it (§C).
  - All 13,937 training moments (every 5th frame) have a full, correctly timed 4 s human path in the generator's frame (§D).
  - The frozen encoder is deterministic. Caching the projector output for all moments takes 3.4 GiB in fp16 (§E).
  - Both losses run in one step at batch 32, with a peak of 7.5 GiB (§F).
  - The baseline menu is reproduced bit-for-bit, and the metrics are in §G.
- **Blocks phase 1 (decisions, not code):**
  1. Half of the upstream generator loss pulls toward vocabulary plans that NAVSIM's rule-based grader rated ≥ 0.95. Building those grades needs a NAVSIM metric cache (map + agents), and KIT has none. So "upstream's loss" can only be the human-path half on KIT.
  2. Upstream also lets the scorer's losses flow back into the generator. With the scorer frozen and out of the loss, that path is gone.
  3. Caching encoder outputs loses upstream's only image augmentation (GridMask).
- **Blocks phase 2: nothing technical.** What remains is choosing the weight between the two losses and the learning rates (open questions).

## A. Upstream generator loss

**Where.** `pad_loss` (`AG/drive_jepa_agent.py:271-346`) calls `trajectory_loss_anchors` (`:221-269`, called at `:282`). `trajectory_loss_topk` (`:178-219`) exists but is not called.

**Form, per refinement pass p** (`proposals_i` is B×32×8×3, holding x, y, heading):
- **Human term (`:227-232`).** For each candidate, take the mean over the 8 waypoints of the L1 norm |dx| + |dy| + |dheading|. Keep the **minimum over the 32** and average over the batch. This is **winner-takes-all**: only the candidate closest to the human path gets a gradient. Heading is in the distance.
- **Vocabulary term (`:234-261`).**
  - For each sample, draw up to 4 random indices from a precomputed list. The list holds the plans in the 8,192-plan vocabulary (`./data/8192.npy`, subsampled `[:, 4::5]` to 0.5 s, `:49-50`) that scored ≥ 0.95 under the NAVSIM grader (`scripts/misc/convert_to_index.py:17-23`, capped at 256).
  - Each drawn plan pulls its own closest candidate (again winner-takes-all).
  - It is added as `0.5 · Σ_samples mean_k(...) / B` (`:261`).
  - The list is read from `anchors_scores_index_v2/<token>.npy` beside the metric cache (`score_module/compute_navsim_score.py:54-55`).
- **Variety term:** `diversity_loss` (`:167-176`) is −min pairwise distance. It is **off**: `inter_weight = 0` (`drive_jepa_config.py:50`).
- **Pass weights.** `trajectory_loss = prev_weight · trajectory_loss + loss_p` (`:264`) with `prev_weight = 0.1` (`config:56`). So passes 1–4 get **0.001, 0.01, 0.1, 1.0**.
- **Two quirks:**
  1. The outer loop is `zip(proposal_list, scores_index)` (`:226`), so it runs min(4, batch) passes. At batch < 4 the later passes are silently dropped.
  2. The inner loop reuses the name `idx_arr`. This is harmless.

**Other terms in the total loss** (`:313-323`), all with weights from `config:49-59`:
- Scorer BCE on the final grade (weight 1; sub-grades weight 0).
- Agent-collision head: BCE weight 1 and L1 weight 0.1.
- Drivable-area head: weight 2.
- Agent-box and BEV-map losses are absent: the heads return `None`.
- **These scorer losses reach the generator.** The scorer reads `proposals` and `bev_feature` without detaching them (`drive_jepa_model.py:71`; `score_module/scorer.py:41,48`). Only the grading call to the external grader detaches (`agent:110,277`).

**Target.** `targets["trajectory"]` is the logged future, 8 poses at 0.5 s in the ego frame (`drive_jepa_features.py:143-150`). On KIT the equivalent is `ADP.build_sample(...).future_poses` (§D).

**Upstream training settings:**

| Setting | Value | Evidence |
|---|---|---|
| Optimiser | Adam, no weight decay | `agent:356-364`; checkpoint optimiser groups `weight_decay 0` |
| Learning rate | 1e-4 for everything outside `_backbone`; 1e-5 for `_backbone` (encoder **and projector**) | `agent:360-361`; `config/common/agent/drive_jepa_perception_based_agent.yaml:17`; saved groups 1e-5 (294 tensors) / 1e-4 (75) |
| Schedule | Constant, no scheduler | Checkpoint `lr_schedulers: []`, `scheduler_progress 0` |
| Batch | 32 **per GPU**, `--nproc_per_node=gpu` | `scripts/training/train_drive_jepa_perception_based.sh:6-8`. Global batch and GPU count are **UNCONFIRMED** |
| Length | max 20 epochs; release saved at epoch index 12 (13 epochs, 17,277 steps, 1,329 steps/epoch), best by `val/score` 0.8767 | Train script `:10`; checkpoint `epoch`, `loops`, `callbacks` |
| Precision / clipping / seed | fp16 mixed (GradScaler state present), no clipping, seed 0 | `config/training/default_training.yaml`; checkpoint `MixedPrecision` |
| Augmentation | GridMask on the input images only, probability 0.7, training mode only. No crops or flips | `bevformer/simple_image_encoder.py:43-45,66-67`; `grid_mask.py:89` |
| Data | navtrain (NAVSIM v2) | Train script `:4` |

**UNCONFIRMED.** The release's own run directory is named `train_ipad_vjepa_v2_topk_anchors_1` (checkpoint `callbacks`), which suggests a top-k + vocabulary variant. Whether the release was trained with exactly the shipped `trajectory_loss_anchors` cannot be settled from the files.

## B. Weights actually loaded in our KIT evaluation

| Model | File(s) | Generator and scorer from | Encoder |
|---|---|---|---|
| Original | `{weights}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt`, sha256 `740335c8…0d05` (asserted in `ENC/02_encoder_build/src/common.py:53-54`, `ENC/03_encoder_training/src/f_final.py:67`) | The release, loaded strict by `AG/drive_jepa_agent.py:59-66` via `RUN.build_agent` (`RUN:108-116`) | The release's (the encoder is first built from `vitl_merge_3dataset_e50.pt`, `simple_image_encoder.py:25,36`, then overwritten by the strict load) |
| Pass 3 | The same file, plus `{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/checkpoints/pass_03/state.pt` (`lora` key: 192 encoder LoRA tensors only) | **The release, unchanged** | Release + LoRA (`ENC/04_encoder_checks/src/close.py:48-66,103`; merged per `ENC/03_encoder_training/src/f_final.py:38-51`) |

- **What the release was trained on:** NAVSIM navtrain (nuPlan / OpenScene, US and Singapore cities). Evidence: train script `:4` and the `RUN:8` docstring.
- **What the scorer grades:** 6 logits, `pred_score` head. The last one, after a sigmoid, is the grade used to pick (`drive_jepa_model.py:85-91`). It was trained by BCE against NAVSIM's rule-based PDM grades from the navtrain metric cache (`agent:160-163, 103-131`).
- **Not the October retrain.** `grep` over `ENC/**/*.py` shows no load of any October Physical AI AV retrain checkpoint (`pass_09.pt`, generator checkpoints). The only `torch.load`s are the agent file, e50, the LoRA states and the selection features.

## C. Generator parts (`./run.sh src/g0_gpu.py parts` → `out/ce_parts.json`)

| Part | Parameters |
|---|---|
| `hist_encoding`: ego status (11) → 256 | 3,072 |
| `init_feature`: learned queries, 256 rows × 256 = 32 candidates × 8 waypoints | 65,536 |
| Refiner, `Bev_refiner.positional_encoding` | 5,120 |
| Refiner, `Bev_refiner.bev_decoder`: one BEVFormer layer (self-attention over the 256 queries + deformable cross-attention to image tokens + FFN) | 996,160 (attention 460,352, FFN 525,568, norms 1,536) |
| Refiner, `traj_decoder`: output head, MLP 256→1024→1024→3 with LayerNorm (`transformer_decoder.py:65-78`) | 1,319,939 |
| **Generator total** | **2,389,827** |

- **The 4 refinement passes share one module.** `_trajectory_head` holds 4 entries but 1 distinct object (`drive_jepa_model.py:35-36`). The checkpoint stores it 4×: 168 keys = 4 × 42.
- **Scorer:** 4,704,992 in total. `pred_score` (1,323,014) is the head used at inference. `pred_col_agent` (2,054,864) and `pred_area` (1,327,114) are training-only heads. Whole model: 311.2 M; encoder 303.9 M.
- **The projector (1024→256, 262,400) is outside the generator.** It is `_backbone.projector` (`simple_image_encoder.py:360,370`).
  - **Upstream trains it**, in the `_backbone` group at 0.1 × lr (`agent:360`; it is among the 294 tensors in that group in the checkpoint).
  - The October retrain froze it together with the encoder: the `"backbone" not in n` filter, `GB 05_small_overfit_check/train_patch/drive_jepa_agent.diff:262-265`.

## D. Horizon, timing and frame (`src/d_horizon.py`, `src/d_sign.py` → `out/d_horizon.json`, `out/d_sign.json`, `out/d_moments.json`)

- **Moment rule:**
  - t is a multiple of 5.
  - t ≥ 15. `build_sample` indexes history t−15 (`ADP:301-304`), and a smaller t would wrap around the array.
  - t + 40 ≤ N − 1: 8 waypoints at 5 frames each (`ADP:349-350`). This is also the horizon the `gt` driving command reads (`ADP:283-284`, with `num_future_frames = 8` at `RUN:210`).
- **Timing.**
  - Every train and picking scene has exactly one pose per frame: 0 / 532 scenes mismatch, and pose timestamps are within 50 µs of frame timestamps.
  - Frame steps fall within 0.1000 ± 3e-14 s.
  - All future waypoints sit within 20 ms of k × 0.5 s: 0 moments dropped.
- **Counts:**
  - **Training moments: 13,937 from 482 scenes** (no scene has zero).
  - The picking scenes give 1,763 under the same rule, but checkpoint picking keeps the baseline rule's **103**.
  - Both images every moment needs (t−5 and t) are in the s2 frame cache; scene frame counts match the cache index for all 532 scenes.
- **Frame and axes.**
  - The human path is built with the wrapper's own `ADP.load_poses` and `to_relative_poses` (`ADP:100-123`): x forward, y left, heading counter-clockwise, relative to the pose at t.
  - Moving moments: 12,410 / 12,410 have x > 0 at 0.5 s. The median gap between travel direction and recorded heading is 1.4°.
  - Turning moments (|heading change| > 0.3 rad at 4 s): **2,290 / 2,296** have the sideways offset and the heading change with the same sign (1,150 left and 1,146 right turns). The 6 others are consistent with S-bends.
  - The evaluation scores the generator output against exactly this array (`RUN:119-132`), and §G shows the wrapper's human path is bit-identical to the cached one on all 103 moments.
- **UNCONFIRMED: reference point.** Which body point KIT's `poses.txt` refers to, against NAVSIM's rear axle, was not checked. A fixed offset would only shift paths in turns, by centimetres.

## E. Phase 1 feasibility: frozen pass-3 encoder (`out/ce_parts.json`)

- **Deterministic: yes.**
  - With pass 3 merged in eval mode, the encoder and projector outputs on 16 picking moments are **bit-identical** when repeated at batch 1.
  - The only random module in the whole planner is GridMask. Its training flag is off in eval mode. No dropout or drop-path has p > 0.
  - **Caveat:** batch 8 against batch 1 differs by up to 2.0e-3 at the projector output (values up to 10.2) and 5.1e-3 at the encoder output. Build any cache with a fixed batch size.
- **Cached size per moment:**
  - Projector output: 512 tokens × 256 = 256 KiB in fp16, 512 KiB in fp32.
  - Encoder output, needed if the projector were trained: 512 × 1024 = 1 MiB fp16, 2 MiB fp32.
  - The generator also needs `lidar2img` (4×4×4) and `ego_status` (4×11). These are negligible.
  - Do **not** reuse the selection-feature layout. It also stores an unused 4×3×448×768 side-camera tensor (16.5 MB per moment), which is why that file is 2.0 GB for 103 moments.
- **Totals for 13,937 + 103 = 14,040 moments:**

  | What is cached | fp16 | fp32 |
  |---|---|---|
  | Projector output | **3.4 GiB** | 6.9 GiB |
  | Encoder output | 13.7 GiB | 27.4 GiB |

  `/` has 117 GB free.
- **Two things a cache needs:**
  1. **`lidar2img` depends on a per-scene ground height estimated from one lidar sweep** (`ADP:208-230,232-240`). That means one NAS read per train scene (482).
  2. **A cache bakes in "no GridMask".** Upstream applies GridMask to 70 % of training images.
  - Running the frozen encoder live from the existing frame cache is the alternative. The s1/s2 data path already keeps up with the GPU (`ENC/03_encoder_training/REPORT.md` §A.4).

## F. Phase 2 feasibility (`./run.sh src/g0_gpu.py memory` → `out/f_memory.json`)

**Setup.**
- Pass-3 LoRA, the generator and the e50 guesser (the predictor from the earlier stage) are trainable. The answer key is a frozen copy. The scorer and projector are frozen.
- Each step runs, under bf16 autocast:
  1. `agent._pad_model(features)` in train mode, so GridMask is on.
  2. **Upstream `agent.trajectory_loss_anchors`, called unchanged**, with an empty vocabulary list, i.e. the human term only (the code skips empty lists, `agent:237-238`).
  3. **Existing `players.game_loss`** from `ENC/02_encoder_build/src/players.py:231`.
- One backward on the sum, then no optimiser step.
- Planner inputs are picking-set features used as stand-ins; hide-and-guess inputs are train pairs from the frame cache.

**Result: both losses work in one step.** Non-zero gradients reach all 192 LoRA tensors and 41 generator tensors. The scorer and projector get none.

| Batch | Peak allocated | Peak reserved | s/step (3 steps, shared card) |
|---|---|---|---|
| 8 | 3.74 GiB | 3.98 GiB | 4.77 (first), 0.50, 0.80 |
| 16 | 5.02 GiB | 5.43 GiB | 0.63, 0.50, 0.63 |
| **32** | **7.52 GiB** | 8.24 GiB | 0.78, 0.99, 0.84 |

- **Batch 32 fits** with about 16–25 GiB free on the card. Optimiser state for the 30.8 M trainable parameters adds about 0.25 GB.
- Losses at the first step: generator about 0.67–0.75, hide-and-guess about 0.58. They are on the same scale.
- At about 0.85 s/step, one pass over 13,937 moments is about 436 steps, roughly 6 min.

## G. Baseline menu on the 103 picking moments (`./run.sh src/g0_gpu.py menu` → `out/g_menu.json`, per moment in `out/g_menu_per_moment.pkl`)

**Drift check first: all bit-identical on 103 / 103.** I rebuilt every moment through the evaluation wrapper's own path (`ADP.build_sample` → feature builders → `agent.forward` / `compute_trajectory`, as `ENC/02_encoder_build/src/f_baseline.py`), then compared:
- Features and human path: equal to the cached selection features.
- Original model: the wrapper's 32 candidates, grades and pick equal the cached-feature dump, and that dump equals the saved s2 zero-LoRA dump.
- Pass 3 (unmerged): the wrapper dump equals the cached-feature dump, which equals the saved s2 pass-3 dump. ADE reproduces at 1.8732948781716308.
- Merged against unmerged pass 3: maximum candidate shift 0.13 mm, 0 picks changed.

**Metric definitions:**
- All metrics use waypoints 1–6 (0.5–3 s).
- "Closest" is the candidate with the lowest ADE; its own FDE@3s is reported. The lowest FDE@3s over the 32 is shown separately.
- **Variety** is the mean over the 496 candidate pairs of the mean waypoint distance.
- **Worst 20 %** is the 21 moments (ceil(0.2 × 103)) with the highest closest-of-32 ADE under the original model. That list is fixed and stored in the JSON.

| Metric (mean / median) | Original | Pass 3 |
|---|---|---|
| 1a. Closest of 32: ADE (m) | 0.573 / 0.305 | 0.563 / 0.303 |
| 1b. Closest of 32: its FDE@3s (m) | 1.061 / 0.455 | 1.036 / 0.489 |
| (lowest FDE@3s over 32, m) | 0.899 / 0.303 | 0.878 / 0.346 |
| 2. Candidates with ADE < 1 m | 9.89 / 11 | 9.91 / 11 |
| 2. Candidates with ADE < 2 m | 18.01 / 20 | 18.26 / 20 |
| 3. Variety (m) | 2.832 / 3.000 | 2.822 / 2.975 |
| 4. Worst 20 % (n = 21): closest ADE / its FDE@3s (m) | 1.813 / 3.538 | 1.724 / 3.329 |
| (reference) Scorer's pick, ADE | 2.392 / 1.414 | 1.873 / 1.001 |

Pass 3 barely changes the menu. Its whole gain was in the pick, as s2/s3 found.

## Open questions (my recommendation first)

1. **Vocabulary half of the loss: no KIT grades exist.** *Recommend:* train with the human-path half only. That is the upstream function called unchanged with empty lists, as in §F. Name it "upstream generator loss, human term" in the write-up. Building KIT grades would need a NAVSIM-style metric cache from KIT maps and agents, which is a separate project.
2. **Scorer gradients into the generator** (upstream has them, the plan does not). *Recommend:* leave them out, as the plan says. Note it as a deviation from upstream.
3. **Projector: frozen or trained?** *Recommend:* frozen in both phases. That keeps "generator only" in phase 1 clean and keeps phase 2 as generator + LoRA. Upstream did train it at 1e-5, so list it as a deviation.
4. **Cache or live encoder in phase 1?** *Recommend:* live frozen encoder from the existing frame cache, with GridMask on as upstream does. Batch 32 fits easily and a pass is minutes. Use the 3.4 GiB fp16 projector cache only if GPU time is the constraint (it costs the augmentation). Either way, compute `lidar2img` once per scene with the adapter's ground-height estimate.
5. **Phase 1 settings.** *Recommend:* upstream's Adam, lr 1e-4, constant, no weight decay, no clipping, batch 32. Use all 13,937 moments (every 5th frame) with the `gt` command, as in evaluation. Check the 103 picking moments after every pass with metrics 1–4. Stop after 3 passes without improvement on metric 1a, or at 10 passes.
6. **Phase 2.** *Recommend:*
   - Loss = generator loss + 1.0 × hide-and-guess. They start at similar sizes (0.67 against 0.58).
   - LoRA at lr 2e-4 and the guesser at 1e-4, as in s2; generator at 1e-4.
   - Start from the pass-3 state, including the answer key and guesser (`state.pt`).
   - Keep the s2 momentum.
7. **Metric 1 definition.** *Recommend:* the FDE@3s of the ADE-closest candidate (1b), with the lowest FDE@3s over the 32 as a side number.
8. **Pass weights 0.001 / 0.01 / 0.1 / 1.** *Recommend:* keep them. Only the last pass's menu is judged and scored.

## Files

- `run.sh`: the s1/s2 interpreter, with `ENC/02_encoder_build/src` and `ENC/03_encoder_training/src` imported unchanged.
- `src/d_horizon.py`, `src/d_sign.py` (CPU), `src/g0_gpu.py` (`parts`, `menu`, `memory`).
- `out/`: `d_horizon.json`, `d_moments.json`, `d_sign.json`, `ce_parts.json`, `g_menu.json`, `g_menu_per_moment.pkl`, `f_memory.json`.
- `logs/`: one log per run.
