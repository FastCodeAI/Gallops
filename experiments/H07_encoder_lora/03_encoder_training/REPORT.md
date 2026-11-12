# Encoder LoRA on KIT — Stage 2: cache, masks, warm-up, training, one final evaluation

Date 2026-10-16 (finished 23:45), shakaal. Folder `experiments/drivejepa_kitscenes/03_encoder_training/`.

**How it runs.** `./run.sh` is s1's interpreter and settings, with `02_encoder_build/src` imported unchanged. The pipeline was `chain.sh`, and after a crash, `chain2.sh` (§E.0).

**Where things live.** Large files are on the local NVMe at `{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/`, recorded in `out/DATA_POINTER.txt`:

| What | Size |
|---|---|
| Frame cache | 7.9 GB |
| Checkpoints | 1.7 GB |
| Selection features | 1.9 GB |
| Merged encoder | 1.2 GB |

## 1. Verdict — DONE, all gates pass

**Headline: selection ADE 2.392 → 1.873 m (pass 3, chosen). On val + overlap_train_val (223 moments), ADE went from 2.3636 m (August) to 1.8793 m with LoRA (−20.5 %). FDE@3s went from 4.199 to 3.442 m.**

- **The gain is almost all along the direction of travel.** The plans got faster, closer to the human's speed. Error at 3 s along the direction of travel fell from 3.96 to 3.21 m; sideways error barely moved, 0.83 → 0.75 m.
- **The candidates themselves did not get better.** The 32-candidate menu is essentially unchanged on selection: best-of-32 ADE 0.573 → 0.563. The scorer now picks differently from the same menu. Pick regret (chosen ADE minus best-of-32 ADE) fell from 1.82 to 1.31 m.
- **The median moves less than the mean.**
  - Val median: 1.467 → 1.249.
  - Overlap median got worse: 0.991 → 1.122.
  - 55 of 223 moments got worse by more than 10 cm, and 95 got better by more than 10 cm.

| Gate / stop condition | Result | Evidence |
|---|---|---|
| A: 300 cached pairs bit-identical to s1 `pairs.build_pair` | **PASS**: 300/300, max diff 0.0; 11/11 shard checksums | `out/a_gate.json` |
| B.2(i): hidden fraction within ±5 points of the square-grid target | **PASS**: −0.71 and −0.12 points | `out/b_masks.json` |
| B.2(ii): large blocks full-height in < 30 % of draws | **PASS**: 22.1 % (s1: 100 %) | same |
| B.3: batch-32 hidden fraction within ±3 points of per-sample | **PASS**: −0.10 and +0.14 points | same |
| C.1: zero LoRA = baseline path on all 103 selection moments | **PASS**: 103/103 bit-identical, drift 0 | `out/c_reference.json` C1 |
| D: warm-up loss falls in the first 500 steps | **PASS**: 0.5988 → 0.5007 | `out/d_warmup.json` |
| No NaN/inf loss or gradient norm | **PASS**: none in 1,000 + 3,522 steps | `logs/*_steps.jsonl` |
| Drift/ADE not identical to zero LoRA after a main pass | **PASS**: drift > 0 after every pass | `out/e_passes.json` |
| F.1: merged = unmerged within 1e-4 m (103 moments) | **PASS**: 2.3e-5 m, 103/103 picks equal | `out/f_final.json` |
| Data rules | **Held.** Val and overlap were read once, for pass 3 only. Test and test-e2e were never read. Naren's tree was not modified (`find -newermt` is empty), and the pickle sha256 is unchanged (`b40a14e9…`). | — |

## Settings (decided; saved as `settings.json` beside every checkpoint)

These are in `src/s2common.py` `SETTINGS`.

- **Frames:** every 5th frame; pair (t−5, t) with t a multiple of 5 and t ≥ 5.
- **LoRA:** rank 16, alpha 16, on `blocks.N.{attn.qkv,attn.proj,mlp.fc1,mlp.fc2}` for N = 0..23; 6,291,456 trainable parameters.
- **The three players:**
  - Learner: the agent's own encoder, wrapped in place (`src/train.py` `Setup`), so each pass's driving check runs the current unmerged LoRA, as in s1 T2.
  - Answer key: a frozen copy, updated by momentum m = 0.99925 on its LoRA tensors only (`train.py:146`).
  - Guesser: the e50 predictor with the s1 grid fix.
- **Optimiser:** AdamW, betas (0.9, 0.999), eps 1e-8, no gradient clipping.
  - LoRA: lr 2e-4, weight decay 0.
  - Guesser: lr 1e-4, weight decay 0.04.
  - Learning-rate ramp: linear over the first 200 steps of each phase, then constant (`train.py:68`). Gradient norms are logged every step.
- **Run:** bf16 autocast for training, fp32 for driving. Batch 32. Seed 20261016. Loss = s1 `players.game_loss`.
- **Two implementation choices to note:**
  1. **Guesser weight decay** follows upstream `init_opt` (`vjepa2/app/vjepa/utils.py:224-236`): no decay on biases or 1-D tensors (`train.py:59`).
  2. **Fresh optimiser for main.** The main phase builds a fresh optimiser, so its ramp starts again from 0.

## Part A — frame cache (`src/cache.py`, `src/a_cache.py`)

1. **What is stored.**
   - The baseline makes each camera feature as `cv2.resize(image[28:-28], (512, 256))`, then `transforms.ToTensor()` (`drive_jepa_features.py:48,51,58`), on the uint8 array from `_load_camera` (`kitscenes_adapter.py:242-257`).
   - We store that uint8 (256, 512, 3) array (`cache.py:40`). The rest is ToTensor, then Normalize, then stack, exactly as in s1 (`cache.py:48`).
   - ToTensor of a uint8 array is a pure function, so the float tensor is exactly reproducible. Gate A confirms this.
2. **Location.**
   - `{p4}` (54 GB free) is owned by root; no directory there is writable by our user.
   - So the cache went to the local NVMe: `/` had 71 GB free, at `{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/frame_cache`.
3. **Layout.**
   - 11 shards of up to 2,000 frames (~786 MB each). Each has a sha256 in `shard_NNN.json`.
   - `index.json` maps scene:frame to (shard, slot). A shard is marked done only after its `.bin` is complete, so a rerun resumes.
   - **Build:** 16 parallel readers, 21,552 frames, **8.47 GB, 1 h 44 min**. The rate was 1.8–2.2 frames/s while the NAS was busy and 13–15 frames/s at the end (`out/a_build_log.json`).
4. **Gate A: PASS.** On 250 train + 50 selection random pairs, all 300 are bit-identical to s1 `pairs.build_pair`.
   - **New pair counts:** train **18,757** pairs (482 scenes, 19,239 frames); selection 2,263 (50 scenes, 2,313 frames).
   - **Reading from the cache keeps up with the GPU:** data wait averaged 2–3 s per pass of 587 steps.

## Part B — covering pattern for the 16×32 grid (`src/masks2.py`, `src/b_masks.py`)

1. **Target.** Upstream's `_MaskGenerator`, unchanged, on its native 16×16 grid with the s1 placeholder settings (5,000 draws).
2. **Shape change (one argument).**
   - Upstream sizes a block from `n = int(H·W·scale)`, `h = round(√(n·ar))`, `w = round(√(n/ar))`, clipped to the grid (`multiseq_multiblock3d.py:136-152`).
   - Area is already a share of the grid. On 16×32, though, `ar ∈ (0.75, 1.5)` pushes h past 16 every time.
   - We pass `ar·H/W`, i.e. (0.375, 0.75), so the aspect ratio is in grid units (`masks2.py:40`). Each block is now the square-grid block stretched 2× in width.
   - The upstream class is used unchanged for block size.
3. **No batch trimming.**
   - Upstream cuts every sample to the batch minimum (`:212-213`).
   - Ours draws the block size once per batch, as upstream does, and draws sample 0's positions uniformly, as upstream does (`:156-158`). Every other sample is then redrawn (vectorised) until it hides exactly as many tokens as sample 0 (`masks2.py:85`).
   - Sample 0's count is a draw from the per-sample distribution, and the others are conditioned on it. So each sample keeps the per-sample distribution, and nothing is added or cut.
   - **Cost:** 2–16 ms per batch on average.
   - **Fallback, added after the main-phase crash in §E.0:** if no match turns up in 500 rounds of 4,096 tries, sample 0 is redrawn (`masks2.py:109`).

| Config | Square-grid target (mean / p5 / p95, full-height share) | s1 wide, per sample | s1 wide, batch 32 after trimming | **New, per sample** | **New, batch 32** |
|---|---|---|---|---|---|
| 8 blocks × 0.15 | 62.0 / 49.2 / 74.2 %, 0 % | 62.3 %, 0 % | 46.6 % | **61.3 %**, 0 % | **61.2 %** |
| 2 blocks × 0.70 | 81.0 / 70.3 / 93.8 %, 21.4 % | 71.7 %, **100 %** | 56.6 % | **80.9 %, 22.1 %** | **81.0 %**, 23.3 % |

4. **Images:** `out/masks/mask_example_0..7.png` show 3 of 32 samples per config, all with equal counts. Tests: `tests/test_masks2.py`, 4/4 pass.

## Part C — selection "before" numbers (zero LoRA)

1. **The 103 moments.** These are 50 scenes under the baseline rule, run through the s1 baseline wrapper path: `build_sample(..., "gt")`, `compute_trajectory`, `displacement_errors(3 s)` (`src/c_reference.py`).
   - The agent features for each moment are saved once (`selection_features.pt`). Each pass then reruns only `agent.forward`, through `evalsets.forward_dump`, which mirrors `abstract_agent.py:76-94`.
   - Our saved-feature run matches `compute_trajectory` exactly, bit for bit, on **103/103** moments.
   - Zero LoRA inside the agent is bit-identical to the plain agent on 103/103.
   - **Before:** ADE mean / median **2.392 / 1.414**; FDE@1/2/3 s **1.275 / 2.689 / 4.346**.
   - Outputs: `out/c_selection_zero_dump.pkl` (32 proposals, grades, pick, trajectory) and `out/c_selection_zero_rows.pkl`.
2. **Fixed guesser-loss set.** 2,000 selection pairs (seeded sample) in 63 batches, each with its own fixed masks (`evalsets.py`).
   - Zero-LoRA loss **0.5988**; a rerun gives the identical value. Items sha256 `fd268f7e…`.

## Part D — warm-up (guesser only, LoRA at zero)

| Step | 0 | 250 | 500 | 750 | 1,000 |
|---|---|---|---|---|---|
| Fixed-set loss | 0.5988 | 0.5056 | 0.5007 | 0.4988 | **0.4973** |

- **Stopped at step 1,000 by the rule:** the loss improved 0.67 % over steps 500–1,000, under the 1 % threshold.
- Wall time 15 min, no GPU wait.
- Almost all the gain comes in the first 250 steps. The guesser re-fits to the agent's encoder (which fine-tuning moved away from e50) and to KIT images.
- Plot: `out/plots/d_warmup.png`.

## Part E — main training (LoRA + guesser)

**E.0 Crash and resume.**
- At main step 1,848 (pass 4), the mask equaliser drew a hidden count of 431: 8 blocks of 5×14 with almost no overlap. No other sample matched in 500 × 4,096 tries, and it raised an error at 07:25. The chain stopped as designed. **I only saw this at 23:09.**
- **Fix:** the fallback in §B.3. I checked that 616 earlier steps reproduce their logged hidden counts exactly, so every batch that worked before is unchanged.
- Training resumed from the pass 3 checkpoint (`chain2.sh`), and pass 4 was rerun from its start.
- The rerun's step 1,800 loss was 0.4782, against 0.4784 the first time. GPU kernels are not bit-deterministic, so the reruns are close but not identical.

**Per pass.** Selection = 103 moments, fp32, unmerged. Drift is against the Part C dump. Grade = the scorer's `pdm_score` per proposal.

| Pass | Steps | Train loss | Fixed-set loss | Sel ADE mean / median | FDE@1/2/3 s | Best-of-32 ADE | Mean / max proposal shift (m) | Mean \|Δgrade\| vs zero / vs previous pass | Picks changed vs zero | Train / eval / GPU wait (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| zero | — | — | 0.4973 (after warm-up) | 2.392 / 1.414 | 1.275 / 2.689 / 4.346 | 0.573 | — | — | — | — |
| 1 | 587 | 0.4848 | 0.4886 | 2.091 / 1.136 | 1.165 / 2.337 / 3.730 | 0.560 | 0.205 / 5.61 | 0.069 / 0.069 | 57 | 787 / 80 / **1,800** |
| 2 | 1,174 | 0.4790 | 0.4859 | 1.933 / 1.090 | 1.059 / 2.160 / 3.456 | 0.560 | 0.230 / 5.16 | 0.077 / 0.029 | 62 | 732 / 69 / 0 |
| **3** | 1,761 | 0.4760 | **0.4853** | **1.873 / 1.001** | 0.963 / 2.082 / 3.477 | 0.563 | 0.267 / 5.99 | 0.090 / 0.024 | 74 | 684 / 59 / 0 |
| 4 | 2,348 | 0.4751 | 0.4874 | 2.009 / 1.167 | 1.080 / 2.232 / 3.647 | 0.559 | 0.303 / 5.70 | 0.093 / 0.020 | 77 | 366 / 43 / 0 |
| 5 | 2,935 | 0.4753 | 0.4902 | 2.045 / 1.300 | 1.116 / 2.278 / 3.679 | 0.562 | 0.321 / 6.82 | 0.097 / 0.024 | 75 | 344 / 23 / 0 |
| 6 | 3,522 | 0.4758 | 0.4922 | 2.025 / 1.130 | 1.092 / 2.253 / 3.673 | 0.559 | 0.348 / 9.00 | 0.099 / 0.024 | 79 | 350 / 25 / 0 |

- **Stop:** the rule fired, with selection ADE worse than pass 3 for passes 4, 5 and 6.
- **Checkpoints kept:** passes 2, 3, 4 and 6 (best 3 + latest); pass 1 was deleted.
- **Wall time:**
  - The pass 1 wait was 30 min BLOCKED-GPU (06:12–06:42, retried every 10 min).
  - Passes 1–3 took 11–13 min each on the busy shared card; passes 4–6 took about 6 min.
  - In total, about 1.4 h holding the GPU across C–F.
- **Plots:** `out/plots/e_loss.png` (losses, gradient norms, ‖B‖) and `out/plots/e_selection_drift.png` (ADE, shift, |Δgrade|, picks).
- **Gradient norms** (50-step mean) were steady at LoRA ≈ 0.02–0.03 and guesser ≈ 0.03–0.05. ‖B‖ grew 9.8 → 15.3 → 19.6 over passes 1–3.

## Part F — final evaluation (pass 3 only, once)

1. **Merge.**
   - `W + (α/r)·B·A` is written into plain `nn.Linear` layers (`src/f_final.py` `merge`), with no LoRA modules left. The merged state keys equal the plain encoder's.
   - On 103 selection moments, merged and unmerged differ by at most **2.3e-5 m**, with 103/103 picks equal. Selection ADE is 1.8732948 merged against 1.8732949 unmerged.
   - The merged encoder is saved as `merged_encoder_pass_03.pt`.
2. **The 223 moments** went through the s1 wrapper loop (`RUN.main`'s calls with `--splits val overlap_train_val --protocol e2e --command gt --eval-horizon 3`): 79 s.
   - For every moment, the recorded human future equals the pickle's (asserted).
3. **Results** (`out/f_final.json`, rows in `out/f_final_rows.pkl`):

| Split | n | ADE mean / median, August → LoRA | FDE@1s | FDE@2s | FDE@3s | ΔADE mean / median | Better > 1 cm / worse > 1 cm / within ±1 cm |
|---|---|---|---|---|---|---|---|
| val | 186 | 2.369 / 1.467 → **1.903 / 1.249** | 1.310 → 1.018 | 2.673 → 2.108 | 4.205 → 3.490 | −0.466 / −0.057 | 106 / 68 / 12 |
| overlap_train_val | 37 | 2.335 / 0.991 → **1.759 / 1.122** | 1.284 → 0.922 | 2.641 → 1.974 | 4.169 → 3.202 | −0.576 / −0.030 | 21 / 16 / 0 |
| all | 223 | 2.3636 / 1.392 → **1.8793 / 1.246** | 1.306 → 1.002 | 2.668 → 2.085 | 4.199 → 3.442 | −0.484 / −0.050 | 127 / 84 / 12 |

- **Software noise.** The August pickle came from Naren's August environment, which no longer exists. On the one moment checked in s1 (`00fc1b9b`, keyframe 40), our software differs from it by up to 1.2 mm per waypoint (ADE 0.2 mm). That is three orders of magnitude below the changes here.

## Extra comparisons (added at Sid's request) — `src/g_extra.py` → `out/g_extra.json`

No model was run; these come from the saved dumps.
- **Before:** zero LoRA on selection; the August pickle on the 223 moments.
- **After:** pass 3.
- **Plan speed** = path length through the 8 waypoints ÷ 4 s. **End distance** = |xy| at 4 s.

| | Selection 103 | Val 186 | Overlap 37 | All 223 |
|---|---|---|---|---|
| **1.** ADE mean, before → after | 2.392 → 1.873 | 2.369 → 1.903 | 2.335 → 1.759 | 2.364 → 1.879 |
| ADE median, before → after | 1.414 → 1.001 | 1.467 → 1.249 | 0.991 → **1.122** | 1.392 → 1.246 |
| Better / worse by > 10 cm (rest within ±10 cm) | 48 / 31 (24) | 83 / 45 (58) | 12 / 10 (15) | 95 / 55 (73) |
| **2.** Pick changed (index) | 74 of 103 | n/a (pickle has no index) | n/a | n/a |
| Old → new chosen trajectory distance, mean / median (m) | changed picks 2.16 / 1.66; same index 0.22 / 0.17 | all moments 1.35 / 0.65 | 1.25 / 0.31 | 1.34 / 0.50 (218 moved > 1 cm) |
| ΔADE where the pick changed / stayed the same | −0.737 / +0.038 | — | — | — |
| **3.** Best-of-32 ADE mean, before → after | **0.573 → 0.563** (better > 10 cm: 5, worse: 7) | UNAVAILABLE → 0.392 | UNAVAILABLE → 0.311 | UNAVAILABLE → 0.378 |
| Pick regret (chosen − best-of-32), before → after | 1.82 → 1.31 | ? → 1.51 | ? → 1.45 | ? → 1.50 |
| **4.** Plan speed m/s, before → after (human) | 5.87 → 6.17 (6.69) | 6.13 → 6.49 (7.00) | 6.97 → 7.50 (7.63) | 6.27 → 6.66 (7.10) |
| End distance at 4 s (m), before → after (human) | 23.4 → 24.6 (26.6) | 24.4 → 25.9 (27.8) | 27.7 → 30.0 (30.5) | 25.0 → 26.5 (28.3) |
| Share of plans that got faster | 65 % | 69 % | 78 % | 70 % |
| ΔADE mean where the plan got faster / did not | −0.85 / +0.09 | −0.65 / −0.07 | −0.75 / +0.05 | −0.66 / −0.06 |
| Correlation of Δspeed with ΔADE | −0.59 | −0.56 | −0.83 | −0.62 |
| Error at 3 s along the direction of travel (x), mean, before → after | 4.21 → 3.33 | 3.95 → 3.24 | 4.03 → 3.05 | 3.96 → 3.21 |
| Error at 3 s sideways (y), mean, before → after | 0.64 → 0.64 | 0.88 → 0.79 | 0.58 → 0.54 | 0.83 → 0.75 |
| **5.** Mean \|Δgrade\| per pass | in the §E table (vs zero and vs previous pass) | | | |

**What this says:**
- **(3) The scorer is choosing differently from the same menu.** On selection, best-of-32 ADE barely moved (0.573 → 0.563, with 5 better and 7 worse by more than 10 cm). The improvement is all in the pick: regret fell from 1.82 to 1.31 m, and moments whose pick index stayed the same got slightly worse (+0.04 m).
- **(4) The pick moved toward faster, longer plans, closer to the human's.** The gain sits almost entirely in moments where the plan got faster, and in error along the direction of travel. Sideways error is flat.
- **So the before → after gain is mostly the scorer no longer under-planning speed on KIT.** The menu got no better, and steering did not change.
- **Gap on val/overlap:** best-of-32 "before" is not available there, because the pickle stores only the chosen trajectory.

## Surprising, and open questions for Sid (not decided)

1. **Best-of-32 "before" on val/overlap, and a same-software baseline.** Getting either means running zero LoRA on the 223 moments. That is the original model in our software, but the Stage 2 rule says no other checkpoint on val/overlap. Is a zero-LoRA run allowed? It would also replace the "1.2 mm noise" caveat with an exact before.
2. **The gain is in the scorer and in speed.** This came from self-supervised image training alone, with no driving labels. The encoder changed enough that the scorer now prefers faster plans, yet the generator's menu barely moved.
   - Is "plans closer to human speed on KIT" the improvement you want?
   - Or should the next check be per-situation, e.g. stops and slow traffic, where the 55 moments that got worse by more than 10 cm may sit?
3. **The median does not move like the mean.** Overlap's median got worse (0.991 → 1.122) while its mean improved. 84 of 223 moments got worse by more than 1 cm. The mean gain is carried by large-error moments.
4. **Selection peaked at pass 3.** After that, the fixed-set guesser loss rose (0.4853 → 0.4922), while training loss stayed flat and drift kept growing. Caveat: the answer key moves, so the loss is not strictly comparable across passes. Options include a lower LoRA learning rate, a learning-rate decay, a different rank, or stopping by fixed-set loss. These are listed, not chosen.
5. **The mask fallback (§B.3, §E.0)** was added mid-run. It changes only extreme-tail batches. Accept it as is?
6. **Selection-based picking bias.** Pass 3 was chosen as the best of 6 on 103 moments. The val/overlap number is the honest one, read once, but it is still a single draw.
7. **The KIT-tuned encoder is KIT-specific.** Whether it helps or hurts in AlpaSim / NuRec closed loop is untested.

## Files

- **`src/`:**
  - `s2common.py` (paths, settings, GPU wait)
  - `cache.py`, `a_cache.py`
  - `masks2.py`, `b_masks.py`
  - `evalsets.py`, `c_reference.py`
  - `train.py`
  - `f_final.py`
  - `g_extra.py`
  - `plots.py`
- **`tests/test_masks2.py`.**
- **`out/`:**
  - JSON for each part
  - zero and per-pass selection dumps
  - `f_final_rows.pkl` / `f_final_dump.pkl`
  - `masks/`, `plots/`
- **`logs/`:** build, gate, per-step JSONL for warm-up and main, chain logs.
- **Checkpoints** at `{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/checkpoints/pass_{02,03,04,06}/`, each with `state.pt` (LoRA, answer-key LoRA, guesser, optimiser) and `settings.json`, plus `warmup_final/`.
