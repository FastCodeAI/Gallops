# Encoder LoRA on KIT — Stage 1 build (no real training)

Date 2026-10-16, shakaal.

**Folder.** This work lives at `experiments/drivejepa_kitscenes/02_encoder_build/`, next to `../01_encoder_survey/REPORT.md`, rather than `analysis_3.0/…`, because retraining work now lives in Gallops.

**How to run.** Every script runs through `./run.sh`. It uses the `drive-jepa` conda Python with the torch 2.8 add-on on `PYTHONPATH`, `DRIVE_JEPA_MSDA=pytorch` and `PYTHONDONTWRITEBYTECODE=1`, as in GB `06_first_training_run/run_s4.sh:2-4`.

**Abbreviations.** `DEVKIT`, `ADP`, `RUN` and `GB` mean what they mean in s0. Here `ADP`/`RUN` are **Naren's devkit copies that actually ran**: `DEVKIT/scripts/inference/{kitscenes_adapter,run_kitscenes_inference}.py`.

**Side effects.**
- We wrote only into this folder.
- Naren's tree was imported and read, never written. `find DEVKIT -newermt 00:50` is empty.
- GPU-holding processes ran about 6 min in total: about 3 min of checks, 1.5 min for Part F and 1 min of timing. No checkpoint was saved.
- Outputs total 0.9 MB, so nothing went to the NAS.

## 1. Verdict — READY-WITH-GAPS

Every piece is built and tested, and the training step works on the real models. Two things stand between this and a clean Stage 2:

1. **Gate F1 fails.** The baseline is not reproduced bit-for-bit. The first sample differs by up to 1.2 mm per waypoint; its inputs match exactly. Part F stopped there, as instructed (§F).
2. **On-the-fly reading from the NAS is about 50× too slow for the GPU.** It reads 1.27 pairs/s with 8 workers, against the GPU's 72 pairs/s at batch 32. That is about 20.5 h per pass, all of it waiting on data (§B, §G).

| Gate / test | Result | Evidence |
|---|---|---|
| B.2 loader = baseline feature path (3 selection keyframes) | **PASS**, bit-identical | `out/gpu_checks.json` B2 |
| T1 zero LoRA → encoder output bit-identical (3 pairs, fp32) | **PASS** (max diff 0.0) | gpu_checks T1 |
| T2 zero LoRA in full agent → trajectory, 32 proposals, logits bit-identical (3 keyframes) | **PASS** | gpu_checks T2 |
| T3 only the 192 LoRA tensors trainable; projector, generator, scorer and base weights none | **PASS** | gpu_checks T3 |
| T4 merged = unmerged (fp32, rtol 1e-5, atol 1e-5; whole-encoder swap 1e-4) | **PASS** | `tests/test_lora.py` |
| T5 momentum rule exact, base weights identical | **PASS** (error 0.0 on the real ViT-L) | tests + gpu_checks E3 |
| T6 planted bugs each break their test: non-zero B → T1, LoRA on projector → T3, LoRA on a non-target layer → T3, merge sign → T4, reversed momentum → T5 | **PASS** (5/5 fail as they should) | `tests/test_lora.py`, 12/12 tests pass |
| **F1** baseline bit-identical, ADE 2.3636 / FDE@3s 4.199 | **FAIL** at sample 1 (stopped) | `out/f_baseline.json` |
| F2 zero-LoRA dump = F1 dump | **NOT RUN** (Part F stopped). Covered for 3 selection keyframes by T2. | — |

## Part A — split file

- **`out/splits.json`** (file sha256 `11b2f40e…13a8`; embedded content hash `697fa0e6…7992`, re-checked on every load by `splits.load_splits`).
  - Scenes: `train.txt` lists 534, of which 532 are on disk across the NAS and `{bulk_disk_root}/…/kitscenes_data`. Missing: `0aef5c74-debd-67ee-c41a-72bb6c82b221` (also absent from the archive manifest) and `ad32c036-166e-4632-99c5-8995b5a03b64`.
  - Rule: selection = `random.Random(20261016).sample(sorted ids, 50)`; train = the other 482.
  - Overlap check: zero overlap with validation (117), overlap_train_val (23), test (206) and test-e2e (127), asserted in `src/splits.py` `build()` and `load_splits()`.
- **Locations** (`out/split_locations.json`, from each scene's `maps/origin.json` plus its first pose):
  - Train: Karlsruhe origin 243 scenes, Frankfurt 237, and a third origin (48.708, 8.981) 2.
  - Selection: 25 Karlsruhe, 25 Frankfurt.
  - 49 of the 50 selection scenes fall in a 1 km cell that also holds train scenes (train covers 59 cells, selection 28).
- **Does the geo split group by location?** The toolkit ships only the lists, with no grouping code (`kitscenes/dataset.py:19` points at the folder). Whether val sits in held-out areas is UNVERIFIED: answering it means reading val locations, which the data rules allow only in Part F.

## Part B — frame-pair loader (`src/pairs.py`)

- **What it yields.** `PairDataset` yields, for every frame t ≥ 5, `[t−5, t]` as a (2,3,256,512) float32 tensor. It calls the baseline's own code and reimplements no image step:
  - `ADP._load_camera("camera_ring_front", i)` (`:242-257`; crop `:165-182`, resize `:247`).
  - `DriveJEPAFeatureBuilder._get_camera_feature` (`drive_jepa_features.py:38-65`: `[28:-28]`, 512×256, ToTensor).
  - `DriveJEPAModel.make_transform` (ImageNet normalisation, `drive_jepa_model.py:42-47`).
  - Stacked older-then-newer as at `drive_jepa_model.py:57`.
  - No GridMask, which only acts in training anyway (`grid_mask.py:89`).
- **Adapter ground offset.** The adapter is built with a fixed `ground_offset`, which only moves camera extrinsics (`:235-240`), never pixels. This skips its lidar read.
- **Import detail.** The devkit's `scripts/` has no `__init__.py`, and this env installs an unrelated package called `scripts` (from the AlpaSim gRPC tree). So devkit files are loaded by path (`common.devkit_module`); the runner is loaded with its own `scripts.inference` names pre-registered (`common.kit_runner_module`).
- **Equality (B.2).** On selection scenes `03091b73`, `0d99e104` and `16a3d521` at keyframe 40, our tensor is **bit-identical** to the tensor entering `_backbone` when the baseline path (`ADP.build_sample` → feature builders → `agent.forward`) runs on the GPU.
- **Counts.** Train 93,784 pairs (96,194 frames); selection 11,315 (11,565 frames).
- **Read speed from the NAS** (`out/pairs_report.json`): 0.46 / 0.80 / 1.27 pairs/s with 1 / 4 / 8 workers.
  - Breakdown: reading one 1.6 MB JPEG takes 0.8 s. A pair costs 0.10 s once its files are in the page cache.
  - Raw NFS: 64 files read in 40 s serially and 22 s with 16 readers, i.e. 3–6 MB/s.
  - **On-the-fly decoding cannot feed the GPU.** See open decision 8.

## Scene lengths (added on request) — `out/scene_lengths.json`, `out/scene_lengths_hist.png`

- **Method.** Frames = lines in `timestamp.reference.txt`; for every scene this equals the front-camera JPEG count. Duration = last − first timestamp. Median frame step is 0.100 s.
- **Moments** use the baseline rule, called on the devkit method itself: `valid_keyframes(num_future_frames=10, stride=90, first_frame=40)` (`RUN:236-238`, `ADP:261-273`).

| Set | Scenes | Frames min / median / max | Duration s min / median / max | Moments (rule) | Pairs |
|---|---|---|---|---|---|
| train | 482 | 100 / 180 / 700 | 9.9 / 17.9 / 69.9 | 850 | 93,784 |
| selection | 50 | 100 / 200 / 600 | 9.9 / 19.9 / 59.9 | **103** | 11,315 |
| val | 117 | 100 / 150 / 600 | 9.9 / 14.9 / 59.9 | **186** | 20,980 |
| overlap_train_val | 23 | 100 / 170 / 490 | 9.9 / 16.9 / 48.9 | **37** | 4,445 |

- **Baseline check.** Val + overlap give **223 = 186 + 37** moments across 140 scenes. Every scene has at least one moment.
- **The keyframe set is identical to the baseline pickle** (no extras either way).
- **Moments per scene:**
  - val: 75 scenes have 1, 28 have 2, 6 have 3, 4 have 4, 3 have 5, 1 has 6.
  - overlap: 14 have 1, 6 have 2, 2 have 3, 1 has 5.
  - selection: 20 have 1, 17 have 2, 7 have 3, 4 have 4, 2 have 6.
- **Pair counts.** Recomputed as Σ(frames − 5), they match Part B exactly.

## Part C — the three players (`src/players.py`, `src/c_players.py` → `out/c_players.json`)

1. **Learner = the agent's fine-tuned encoder.**
   - Size: 292 tensors, 303,885,312 parameters, loaded strict from `agent._pad_model._backbone.img_backbone.*`.
   - File hashes: agent ckpt sha256 `740335c8…0d05` and e50 `46491827…668f`, both re-hashed.
   - State hash: learner `a463f524…` equals the agent ckpt's encoder, and differs from e50 `encoder` (`ff8a03c3…`) and `target_encoder` (`ca2173d5…`).
   - Relative difference vs e50 `target_encoder`: worst tensor **5.23 %**, overall 2.23 %. Vs e50 `encoder`: 17.6 % / 13.4 %.
   - Construction is the exact call in `vit_encoder.py:57-61`. Its `input_size` argument is swallowed by `**kwargs`; the non-square grid comes from `handle_nonsquare_inputs` (`vision_transformer.py:176-181`).
2. **Guesser = the e50 predictor** (22,086,016 params, strict load, 10 mask tokens).
   - Settings come from `pretrain-256px-16f.yaml:69-78`.
   - **One change was needed:** upstream `predictor.py:227-229` calls its blocks without the grid. RoPE then assumes a square grid of side 16 (`modules.py:291-321`), so on 16×32 token 300 reads as (frame 1, row 2, col 12) instead of (0, 9, 12). Passing the grid changes the output by up to 10.75.
   - `players.Guesser` overrides `forward` to pass T=1, H=16, W=32. Everything else is upstream verbatim.
   - Mask token 0 is used, as upstream's `PredictorMultiSeqWrapper` does for a single clip length (`src/utils/wrappers.py:35-41`).
   - Runs end-to-end on a real pair: 512 tokens, 132 context, 380 hidden, output (1, 380, 1024).
3. **Masks.** These use upstream's own `_MaskGenerator`, unchanged (`multiseq_multiblock3d.py:80-240`). With crop (256, 512), 2 frames and tubelet 2 it produces one 16×32 slice, so no 2D rewrite was needed.
   - PLACEHOLDERS: 8 blocks at 0.15 scale and 2 blocks at 0.7, aspect ratio 0.75–1.5.
   - Hidden fraction over 1,000 draws:

     | Config | Mean | p5 | p95 |
     |---|---|---|---|
     | 8 small blocks | 62 % | 50 % | 74 % |
     | 2 large blocks | 72 % | 53 % | 91 % |
     | Union of both | 85 % | — | — |

   - Upstream trims every sample in a batch to the batch minimum (`:212-213`). At batch 16 this lowers the hidden fraction to 49 % and 59 %.
   - On the 2:1 grid, block heights hit the 16-row limit, so the large blocks always run the full height and hide vertical bands (`out/masks/mask_example_*.png`).
4. **Targets** follow upstream `app/vjepa/train.py:413-435`: answer-key output, `F.layer_norm` over features, gathered at the hidden positions, L1 (`loss_exp` 1). The loss is averaged over the two mask configs. The answer key runs under `no_grad`. → `players.game_loss`.

## Part D — hand-written LoRA (`src/lora.py`)

- **Layer.** `LoRALinear`: y = W x + b + (α/r)·B(A x). A is Kaiming-uniform (a=√5), B is zero, there is no dropout, and the base is frozen.
- **Targets.** `blocks.N.{attn.qkv, attn.proj, mlp.fc1, mlp.fc2}`, giving 96 layers.
- **Size at rank 16, alpha 16:** **6,291,456** trainable parameters.

  | Layer type | Parameters |
  |---|---|
  | qkv | 1,572,864 |
  | proj | 786,432 |
  | fc1 | 1,966,080 |
  | fc2 | 1,966,080 |

- **Tests:** T1–T6 all pass; see the table in §1.
  - T1–T3 ran on the real ViT-L and agent on the GPU, in fp32, which is how the baseline runs.
  - T4–T6 ran on a 2-block ViT of the same class on the CPU, with the same check functions.

## Part E — one training step (`src/gpu_checks.py`, GPU, bf16 autocast, 2 train pairs)

- **Warm-up mode (LoRA held at zero):** loss 0.628.
  - The guesser has 151 of 160 tensors with non-zero gradients. The other 9 are the unused mask tokens.
  - LoRA A and B, the base encoder, the answer key and the agent all have no gradients.
- **Main mode, first step:**
  - B has non-zero gradients in 96/96 layers.
  - **A is exactly zero, as it must be while B = 0** (dL/dA ∝ Bᵀ).
  - The base encoder, answer key and agent (projector, generator, scorer) have no gradients.
- **After one AdamW step and one momentum update** (m = 0.99925):
  - The answer key's B equals m·key + (1−m)·learner exactly (error 0.0). Its base weights are identical.
  - The next backward pass gives non-zero gradients to both A and B in 96/96 layers. Loss on the same batch: 0.628 → 0.605.
- **Initial warm-up loss** (8 batches of 8 train pairs, no grad): **mean 0.613** (range 0.599–0.630). The e50 pretraining run ended at 0.485, with different data and frames. This 0.613 is the guesser mismatch the warm-up should bring down.

## Part F — baseline reproduction (`src/f_baseline.py` → `out/f_baseline.json`)

1. **Merged root.** `out/kit_root/data/{val,overlap_train_val}/<id>` holds 140 symlinks (117 + 23); nothing was copied.
2. **Runner wrapper.**
   - The sample loop is `RUN.main`'s (`RUN:184-275`) with the baseline arguments: `--splits val overlap_train_val --protocol e2e --command gt --eval-horizon 3`.
   - `RUN.build_agent`, `ADP`, `agent.compute_trajectory` and `RUN.displacement_errors` are called unchanged.
   - Beside `RUN:256`, the features are rebuilt and `agent.forward` runs under `no_grad`. Per sample it saves proposals (32×8×3), `pred_logit` (32×6), `pdm_score`, the chosen index and the trajectory.
   - It asserts the dumped trajectory equals `compute_trajectory`'s output bit-for-bit, so the hook adds no difference of its own.
3. **Gate F1: FAIL at the first sample.** The sample is `00fc1b9b-005e-554c-7733-c4c0adeca015` (val), keyframe 40.
   - Max pose difference **1.2 mm**. Per waypoint: 0.16, 0.02, 0.25, 0.20, 0.48, 1.22, 0.59, 0.65 mm.
   - ADE 0.57491 vs 0.57472.
   - Inputs that could be checked are identical: recorded future poses, ego speed 2.09366 and command. Both are float32.
   - **Why: UNVERIFIED.**
     - The error is floating-point-sized and consistent with a different torch/CUDA build.
     - The baseline's environment is no longer on shakaal. The bytecode in `DEVKIT/**/__pycache__` is `cpython-39`, but no Python 3.9 environment with torch remains under `{naren_workspace}` (miniconda base is 3.13; the other envs are 3.14).
     - `run.json` records no hardware or software (`results/kitscenes/drivejepa/legacy-kitscenes_e2e_final/run.json`).
     - Our run is deterministic: the same agent run twice is bit-identical (`control_plain_twice_bit_equal`).
   - Nothing was tuned. The aggregate over all 223 samples was not computed, because the loop stopped at the first difference as instructed.
4. **F2: not run.** Zero LoRA inside the agent is bit-identical on 3 selection keyframes (T2).
5. **Drift check** (`src/drift.py`).
   - It pairs proposal i with proposal i. This is valid from the code: proposal i is decoded from learned query rows [8i, 8i+8) (`drive_jepa_model.py:33,62`; `traj_refiner.py:24`). The refiner only reshapes and permutes the layout (`bev_refiner.py:113-142`). The pick is an argmax over that fixed index (`:85-91`).
   - Outputs per sample: mean and max proposal shift (m), mean |Δ grade|, and whether the pick changed.
   - `tests/test_drift.py`: a dump against itself gives all zeros, and a planted 3-4-5 m shift reads 5.000 m. F1 vs F2 was not produced (Part F stopped).
6. **Per-split baseline table** (from the pickle, read-only):

| Split | n | ADE mean / median | FDE@1s | FDE@2s | FDE@3s |
|---|---|---|---|---|---|
| val | 186 | 2.369 / 1.467 | 1.310 | 2.673 | 4.205 |
| overlap_train_val | 37 | 2.335 / 0.991 | 1.284 | 2.641 | 4.169 |
| all | 223 | 2.3636 / 1.392 | 1.306 | 2.668 | 4.199 |

## Part G — timing (`src/g_timing.py` → `out/g_timing.json`; main mode, bf16, 3 warm-up + 10 timed steps, a real batch reused)

| Batch | s/step mean (min–max) | Pairs/s | Peak allocated | Free before → after |
|---|---|---|---|---|
| 8 | 0.395 (0.33–0.62) | 20.3 | 4.3 GiB | 29.4 → 28.7 GiB |
| 16 | 0.378 (0.32–0.62) | 42.3 | 4.7 GiB | 29.3 → 28.3 GiB |
| 32 | 0.442 (0.38–0.62) | 72.4 | 5.1 GiB | 29.3 → 27.7 GiB |

- **Fit.** All three batch sizes fit. Activation checkpointing is on, from upstream `in1k.yaml`. The step time is nearly flat, so batch 32 or more is cheap.
- **Per pass over the 93,784 train pairs.** The GPU alone needs 1.28 / 0.62 / 0.36 h at batch 8 / 16 / 32. Reading from the NAS needs 56.6 / 32.6 / **20.5 h** with 1 / 4 / 8 workers, so **every configuration is data-bound**.
- **Others on the card.** 8 other users' processes held about 63 GiB (largest 20.5, 15.7 and 8.4 GiB). Utilisation read 13–100 % during our steps, and our step times varied ±40 % with it. pid 2575588 in the "others" list is our own process.

## Open decisions for Sid before Stage 2

1. **The baseline to compare against** (F1 failed).
   - (a) Re-run the baseline in this env and use that as the "before" number, so both sides share one software stack. Cost: about 2 min of GPU, plus NAS reads.
   - (b) Find or rebuild Naren's August env and retry bit-equality.
   - (c) Accept the original pickle and state the 1–2 mm noise floor.
   - Evidence: §F.3.
2. **LoRA rank, alpha and targets.**
   - Placeholders: r16 / α16 on all four layer types (6.29 M parameters).
   - Memory is not a constraint: 5.1 GiB at batch 32.
   - Options: keep; attention-only (qkv + proj, 2.36 M); r8 or r32.
3. **Mask settings.**
   - Upstream square-grid blocks on a 2:1 grid give full-height vertical bands for the large blocks.
   - About 85 % of tokens are hidden per draw, falling to 49–59 % after batch trimming.
   - Options: keep the upstream placeholders; size blocks relative to 16×16 and tile; make large-block width proportional; stop the batch-minimum trimming by padding.
   - Evidence: §C.3, `out/masks/`.
4. **Learning rates.**
   - Upstream pretraining uses AdamW 5.25e-4 with wd 0.04 (`pretrain-256px-16f.yaml:86-98`). Upstream planner fine-tuning ran the encoder at 1e-5 (s0 Q6a).
   - Options: guesser 1e-4 to 5e-4; LoRA 1e-4 to 1e-3 (LoRA usually runs higher than full fine-tuning).
5. **Momentum.**
   - Upstream uses a constant 0.99925 (`:87-89`). The answer key's half-life is about 924 steps, roughly one third of a pass at batch 32 (2,931 steps).
   - Options: keep; 0.999; or ramp to 1.0.
6. **Warm-up length rule.**
   - The starting loss is 0.613 against e50's 0.485, and one step on the same batch moved 0.628 → 0.605.
   - Options: a fixed number of steps; stop when selection-set loss improves by < x % over N steps; cap at k passes.
7. **Batch size.** 32 fits (0.44 s/step); larger is likely fine. Options: 32, 64.
8. **Data path / number of passes.**
   - From the NAS, a pass is about 20.5 h of reading for 0.36 h of GPU.
   - Options:
     - (a) Cache the 512×256 frames produced by this exact loader as uint8. That is 107,759 frames × 393 KB ≈ 42 GB in large shard files on the NAS, or on `{p4}` (54 GB free). Building it means one slow pass, and the cache needs its own bit-equality check against `pairs.build_pair`.
     - (b) Train on a subset that is read once.
     - (c) Fix the pass budget around the reader.
9. **Checkpoint and selection cadence.**
   - The selection set gives 103 baseline-rule moments (each reads 5 camera images from the NAS, about 5 s, so about 9 min per check from the NAS; the GPU cost is negligible).
   - Options: every N steps or every pass; drift check plus ADE on the 103 moments; keep the top-k checkpoints by selection ADE.

## Files

- **`src/`:**
  - `common.py` (paths, env, devkit loaders)
  - `splits.py`
  - `pairs.py`, `pairs_report.py`
  - `players.py`, `c_players.py`
  - `lora.py`
  - `gpu_checks.py`
  - `scene_lengths.py`
  - `drift.py`
  - `f_baseline.py`
  - `g_timing.py`
- **`tests/`:** `test_lora.py`, `test_drift.py`.
- **`out/`:** JSON results, `splits.json`, mask and scene-length images, the symlink root, F1 rows and dumps (1 sample).
- **`run/`:** GPU run logs.
