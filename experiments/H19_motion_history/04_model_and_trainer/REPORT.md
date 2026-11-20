# Approach 4, build 2: model, loader and training script (no training run)

Date 2026-11-18. Design: `PREREG_APPROACH4.md` with amendment 1 (sha256 `f8f4b975…a450`): training uses the frozen encoder's cached tokens, no GridMask, no hide-and-guess. Every CPU job ran as `taskset -c 0-1 nice -n 19`. The GPU was used only for the step-0 check and the two dry runs, each after the memory gate (5-min load < 32 and available RAM > 40 GB).

**Ready to train: yes.** Estimated wall time per arm, assuming the run stops where 0.1.8b stopped (5,580 steps, 7 checks): **arm A ≈ 7.3 h, arm B ≈ 4.6 h**. Worst case, all 3 passes (13,950 steps, 16 checks): **A ≈ 18 h, B ≈ 11 h**. The two arms **fit on the GPU together** (3.7 GB peak each; about 70 GB free at the time). They do **not** fit together on the 2 pinned CPU cores: grading is the bottleneck, so running both at once would roughly double each arm's time unless more cores are allowed.

## 1b. Local token copy

- Location: `{approach4_store}/tokens/` (the exception approved for this lane).
  - `tokens.npy`: (126,981, 512, 256) fp32, 66.57 GB.
  - `pooled.npy`: (126,981, 33, 256) fp32, 4.29 GB.
  - `index.parquet`: 30 MB, maps sample_id to row and also holds the stored car state and human trajectory.
  - `camera.npz`, `copy.json`.
  - Total about 70.9 GB.
- Wall time: 1,480 s for the single pass over the 4,015 manifest clips, plus 718 s for the finish step.
  - The first run crashed after the copy was complete, while building the index: a pandas attribute bug, `out.clip` should have been `out["clip"]`.
  - The finish step rebuilt the index from the copied files without copying again. This is recorded in `copy.json`.
- Checks:
  - 640 random rows are bit-identical to `torch.load` of the source cache files.
  - The row count equals the manifest's 126,981 moments.
  - Every pooled row equals approach 3's `pooled_training.npy`.
  - No all-zero token rows.
  - Each file's sha256 was taken while writing; the temp-name-then-rename rule was followed.
- Reader benchmark (32 random rows, 20 batches, cold): **1.50 s per batch** (median 1.50, max 1.56). The NAS reader took 3.17 s per batch, which caused the earlier stop.

## 2. Loader (`src/a4_data.py`)

- Sample list and sampler are 0.1.8b's, imported unchanged: `pai_train.manifests()`, then `pai_data.sampler_017` / `batches_017`, batch 32.
- Per moment it returns:
  - tokens and pooled, read from the local arrays;
  - car state: the newest stored row, with acceleration slots 5:7 zeroed;
  - the camera matrices of the clip's first moment;
  - the human trajectory, float64;
  - the label-2 flag and the metric-cache path;
  - the (8, 6) car-state history from the side file. Its root is one config value, `a4_config.SIDE`.
- Any moment missing from the token index, the history file or the camera table raises an error. There is no fallback.
- **Equality against 0.1.8b's `planner_dataset` on 100 random training moments:** 100/100 on every field (car state, trajectory, camera matrices, image shape, label-2 flag, metric-cache path, sample id), with dtypes and shapes equal.
  - The first version cast the trajectory to fp32 and scored 0/100 on it. It now keeps float64 as stored. Fixed and re-checked: `out/data_check_equality.json`.
- **Full pass:** one pass of 0.1.8b's sampler, pass 1, seed 0.
  - 4,650 batches, 148,796 draws; all 126,981 manifest moments served.
  - **4,858 s, 1.04 s per batch** on the 2 pinned cores, with a background thread building batches 4 ahead (`out/data_check.json`).
  - In training this is hidden behind the step: the loader wait was 0 s on dry-run steps 2–3.

## 3. Model (`src/model_a4.py`, wraps p06, edits nothing upstream)

- Motion encoder 48→256→GELU→256. A learned "no history" vector of 256 values, starting at zero.
- Hiding is seeded: `hide_prob` is 0.5 for arm A and 1.0 for arm B. At evaluation A never hides and B always does.
- **Link 1:** a zero-initialised Linear 256→256, added to the plan starting points at the same place in both the main forward and the replay.
- **Link 2:** the first Linear of the scorer is widened, computed in two pieces as W·x + b + W_new·s so that step 0 is bit-exact. W_new is 1024×256, starts at zero and has no bias.
- Everything else is p06, and the comfort re-score is untouched. `load_from_p06()` asserts all 495/495 tensors.
- **New parameters: 406,528:**
  - motion 78,336;
  - no-history 256;
  - link 1 65,792;
  - link 2 262,144.
- Trainable 7,492,643 (generator 41 tensors, scorer 30, plus the new blocks); frozen 304,156,416.

## 4. Step-0 check (GPU, fp32, batch 1, eval, no previous plan, 1,131 graded picking moments)

The reference is p06 through its own original path (live images through the frozen encoder), compared with the wrapper at step 0 reading cached tokens and the real history.

| | arm A | arm B |
|---|---|---|
| picks equal | 1131 / 1131 | 1131 / 1131 |
| plans equal | 1131 / 1131 | 1131 / 1131 |
| rating logits, max abs diff | **0.0** | **0.0** |
| pick grade mean | 0.768489 | 0.768489 |
| moments hidden | 0 | 1131 |

- The pick grade equals 0.1.8b's `run.json` value for p0.6, and the grades equal 0.1.8b's stored p0.6 grades.
- **Cached tokens vs the live encoder:** 1131/1131 bit-identical (max diff 0.0). Car state, camera and metric-cache path are also 1131/1131. No finding.
- **Link sanity:** link weights set to N(0, 0.01²).
  - Arm A's logits moved on 100/100 moments (max 3.80) and changed 54 picks. A also moved on 99/100 when the histories were shuffled.
  - Arm B moved on 0/100 (diff 0.0), and B with shuffled histories equals B unshuffled exactly.
- Output: `out/step0_check.json`. PASS.

## 5. Training script (`src/train_a4.py`)

- 0.1.8b attempt 2's recipe with amendment 1.
- Loss: generator (`generator_loss_s10`) + grade + collision + drivable + human. The human term has weight 0.5 and pull-away ×2, plus the hard-brake rule, through `Model018c.scorer_lessons`. There is no hide-and-guess term and no GridMask.
- Schedule, safety checks and selection rule are 0.1.8b's:
  - checks every 930 steps; up to 3 passes;
  - stop after 6 checks without improvement, or after the hard gate fails 2 checks in a row;
  - pull-away go gate ≥ 0.66;
  - `np.random.seed(SEED+pass)` at each pass.
- Optimiser: Adam, as in 0.1.8b, at the same rates (generator 3e-5, scorer 1e-4).
- Trainable: motion encoder, no-history vector, both links, generator and scorer. Selected with `--arm A|B`.
- **Dry runs** (3 steps, GPU, memory gate passed: A load 19.3 / 71 GB, B 26.2 / 69 GB):

| | arm A | arm B |
|---|---|---|
| losses finite | yes | yes |
| encoder gradients | 0 | 0 |
| peak GPU memory | 3.58 GB (3.71 reserved) | 3.58 GB (3.71 reserved) |
| s/step, steps 2–3 | **4.05** (3.80, 4.31) | **2.31** (3.27, 1.36) |
| of which waiting on grading | 3.30 | 2.24 |
| new tensors that moved | 8 / 8 | 1 / 8 (link-1 bias only) |

- Step 1 (49 s for A, 23 s for B) is start-up of the grading pool and is excluded.
- Steps 2–3 use the same batches in both arms. The difference between arms is grading-time noise on 2 shared cores, not model cost.
- **Full-run estimate per arm:**
  - Method: s/step × steps, plus checks at about 500 s each. A check is about 415 s of grading on 2 cores, measured in step 4, plus the forward and the shares.
  - At 0.1.8b's actual length (5,580 steps, 7 checks): **A ≈ 7.3 h, B ≈ 4.6 h**.
  - Worst case (3 × 4,650 = 13,950 steps, 16 checks): **A ≈ 18 h, B ≈ 11 h**.
  - For comparison, 0.1.8b's whole run took about 4.7 h. Its grading wait averaged 0.18 s per step on unpinned cores, and its check grading took about 24 s against our about 415 s.

## Flags for the reviewer

1. **Arm B trains no history pathway, and this follows from the design.** B always reads the no-history vector, which starts at zero, and both links start at zero. So the motion encoder, the no-history vector and both link weight matrices get exactly zero gradient and stay zero for the whole run. Only link 1's bias moves: a constant offset on the plan starting points. B is therefore "p06 retrained on the cached-token path". That matches amendment 1 ("Arm B measures the effect of this training path on its own"). I am naming it so nobody reads the zeros as a bug.
2. **The new blocks use Adam at 1e-4.** 0.1.8b has no such blocks, so this rate is a choice not covered by its recipe. It matches the scorer's rate.
3. **No gradient reaches link 1 through the replay.** This is inherited from p06, which builds the replay's starting points under no-grad. Link 1 learns only from the main forward.
4. **Timing is inflated by the 2-core pin.** Grading takes about 3 s per step against 0.18 s in 0.1.8b. Checks take about 8 min each, against 0.1.8b's 24 s of grading. Grading is synchronous because nothing else is left to overlap it with.
5. **No resume support.** A crash means a restart, which costs up to the full estimate. 0.1.8b needed one resume.
6. **Two bugs found and fixed during the build:** the crash in the copy's index step (the data was not affected; recovered by the finish step, recorded in `copy.json`), and the loader's trajectory dtype (fp32 → float64, re-checked 100/100).
7. **Two runs cannot share cores 0-1 without slowing each other.** Running the arms side by side needs either a second pinned core pair or sequential runs: about 12 h sequential at 0.1.8b's length, up to about 29 h worst case.

Outputs: `out/copy.json` (in the tokens folder), `out/local_reader_bench.json`, `out/data_check.json`, `out/data_check_equality.json`, `out/step0_check.json`, `out/dry_run_A.json`, `out/dry_run_B.json`.
