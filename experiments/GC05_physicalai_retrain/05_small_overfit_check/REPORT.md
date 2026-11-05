# S3.5 — 1% smoke training

**Status 2026-10-07: Steps 1–4 done. VERDICT by the pre-registered rule: BROKEN (S1 fails), and S2 fails too. S3, S4, S5, S6 hold.**
My reading of the same numbers is that the pipeline works and the run is under-trained, not broken: every loss falls steadily for 30
epochs on both the slice and the held-out set, and held-out was still improving at the end. Details in "Steps 2–4" below.

## Step 1 — wiring

Vendored files are untouched (sha256 of the reference copies in `train_patch/upstream_ref/` equals the vendored files). All changes
live in two patched copies plus one helper:

| file | patched copy of | diff |
|---|---|---|
| `train_patch/drive_jepa_agent_s35.py` | `navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py` | `train_patch/drive_jepa_agent.diff` |
| `train_patch/dataset_s35.py` | `navsim/planning/training/dataset.py` (`CacheOnlyDataset`) | `train_patch/dataset.diff` |
| `train_patch/s35_common.py` | — (paths, flags, split guard, builders) | — |

Tests: `test_wiring.py unit` → `wiring_checks.json`, `wiring_unit.log`; `test_wiring.py full640` → `wiring_full640.json`.
Interpreter: conda drive-jepa python, CPU.

### Wiring table

| # | edit | upstream lines replaced | what the patch does | flag |
|---|---|---|---|---|
| 1a-1 | skip the metric-cache lookup | agent `:45-47` (global `MetricCacheLoader`), `:103-115` (`metric_cache_paths[token]` for every token); dataloader `dataset.py:111-132` | dataloader adds `has_label2` and `metric_cache_path` to each sample's targets; `compute_score` scores only the labelled rows; a raw-only token is never looked up | `has_label2=` manifest dict, else file presence |
| 1a-2 | drop the vocabulary pseudo-target | agent `:276-278`, consumed at `:236-261` | raw-only rows get an empty anchor index; upstream's own `if idx_arr.shape[0] == 0: continue` (`:237-238`) then skips the term | — |
| 1a-3 | mask the scorer loss | agent `:289-300`, summed at `:313-323` | all five scorer-head terms (sub-score, final score, key-agent CE and L1, ego-area) are computed on labelled rows only, same mean reductions; a batch with no labelled row contributes 0 | — |
| 1a-4 | anchor-index path and grader | `compute_navsim_score.py:52-76` (`get_sub_score`, index path by string replace `:54-55`) | index read from `anchors_scores_index_{orig,ext}.npy` next to the metric cache; grading goes through `04_training_set/s3_scorer.score` (S3's D2 real box + D3 exemption), so training-time grades match label 2 | `scorer_impl=s3` (default) / `upstream` |
| 1b | extended vocabulary | agent `:49-50` (`np.load("./data/8192.npy")`) | vocabulary path is an argument; index kind must match it; an index row beyond the vocabulary is refused | `vocab=ext` (default) / `orig` |
| 1c | zero acceleration | `dataset.py:120-124`; slots defined at `drive_jepa_features.py:79-84` | `ego_status[..., 5:7] = 0` at load time | `zero_accel=True` (default) |
| 1d | init + freeze | agent `:59-66` (`initialize`), `:356-364` (`get_optimizers`) | `init_checkpoint` loaded strict with upstream's key mapping; encoder `requires_grad=False` and dropped from the optimiser | `init=True`, `freeze_encoder=True` (defaults) |
| extra | one ego_status row | `dataset.py:120-124` | keep only the newest row (see Findings) | always |

### Checks (all pass)

| check | result |
|---|---|
| Raw-only batch (4 masked samples), forward + backward | loss 8.92, finite; all 30 scorer parameter tensors have gradient `None`; generator gradient max 2.59; encoder gradient none; 0 metric-cache lookups |
| Labelled batch (4 S2 samples) | loss 20.37, finite; scorer gradient max 0.31; one scoring call |
| Mixed batch (raw, labelled, raw, labelled) | gradient on the scorer outputs of the raw rows is exactly 0 on all three heads, non-zero on the labelled rows; the five scorer terms equal those of the labelled rows alone (max diff 0.0) |
| Same as upstream when every sample is labelled | vendored `pad_loss` vs patched, same predictions and seed, original vocabulary: all 13 loss keys differ by 0.0, with `scorer_impl=upstream` and with `s3` |
| One raw-only Adam step | scorer weights bit-identical, generator moved, encoder bit-identical |
| Labelled step, then raw-only step | scorer moves on the labelled step only; encoder never moves |
| Vocabulary file | (10240, 40, 3) float32; first 8,192 rows bit-identical to `data/8192.npy`; sha256 `4b0bc3c9…39e785` (orig `cc44a31e…38ad35`) |
| Tensors that depend on vocabulary size | none in the model or checkpoint (495 tensors scanned). Only `self.anchors` (agent `:49-50`, used at `:246`); `init_feature` is (256, 256) = 32 proposals × 8 poses |
| Vocabulary flag in the loss | labelled-batch pseudo-target loss 13.10 (orig index) vs 17.94 (ext index); raw-only loss identical under both (8.92); an extended index against the original vocabulary is refused |
| S3 layout, real extended index (2 graded S3 samples at 37.0 and 34.1 m/s) | loss finite, scorer gets gradient; index sizes 20 and 23, **all of them added rows (≥ 8192)** |
| Acceleration slots | slots 5:7 equal `ax, ay` of `samples.csv` to 4e-16 before zeroing; zero after; the other 9 slots untouched |
| All 640 masked samples through the dataloader | 640 loaded; max abs acceleration 0.0; none has label 2; split guard passed |
| All 640 masked samples, forward + backward (80 batches of 8) | every loss finite (min 0.06, mean 8.37, max 57.3); scorer gradient `None` on every batch; 0 metric-cache lookups; 8.1 s per batch on CPU |
| Checkpoint | `{runs}/alpasim_bulk/navsim-exp/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt`, sha256 `740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05` (recomputed, matches `real_model.py`); strict load, 495 tensors, 0 missing, 0 unexpected |
| Encoder | **upstream trains it**, at 0.1 × lr (`get_optimizers :356-364`; the group is the whole `_backbone` = ViT-L + its 1024→256 projector). Frozen here: 304.1 M parameters, 0 trainable; optimiser holds one group of 75 tensors at 1e-4 |

Parameter counts: encoder 304,147,712 (frozen); generator 2,389,827 (2,381,123 trainable — 8,704 are fixed upstream); scorer 4,704,992.

### Findings

- **ego_status has 2–4 rows per sample in our caches** (navtrain always has 4), so the stock collate cannot build a batch. The model
  reads only the newest row (`drive_jepa_model.py:51`), so the patched dataloader keeps that one. Without this no batch forms.
- **The scorer loss also trains the generator trunk.** The scorer heads read `bev_feature`, which is not detached, so on labelled rows
  the scorer loss reaches the trajectory head. This is upstream behaviour and is kept. The reverse does not happen: the trajectory
  loss never reaches the scorer heads, which is why a raw-only batch leaves them with no gradient at all.
- **S4 can be checked per step.** Torch's Adam skips parameters whose gradient is `None`, so a raw-only step leaves the scorer
  bit-identical even after labelled steps (measured). This needs `zero_grad(set_to_none=True)`, the default.
- The frozen encoder has no dropout and no drop-path; its only train-mode behaviour is GridMask (p = 0.7), an upstream augmentation,
  left on.
- The S2 pilot caches have no anchor index, so the labelled-batch tests use **dummy index fixtures** (`test_fixtures/`, wiring only).
  The S3-layout check uses real indices.

## Flags

1. **GPU (blocks Step 3).** The model only builds in the conda drive-jepa env (python 3.9, torch 2.1.0+cu121, mmcv-full 1.7.2), and
   that torch cannot run on the Blackwell card. CPU costs about 8 s per forward + backward at batch 8 (frozen encoder). Options: (a) a second env
   with torch ≥ 2.7 cu128, relying on the pure-PyTorch deformable-attention fallback (`DRIVE_JEPA_MSDA=auto`), with a forward-parity
   check against the current env before any training; (b) the JarvisLabs H100, which cu121 supports. Not started; needs your call.
2. **Scorer in the loop is S3's, not upstream's** (edit 1a-4). Required for the grades to match label 2; identical to upstream on
   Pacifica-box caches without the D3 flag (checked). `scorer_impl=upstream` restores the vendored path.
3. **Scorer loss is averaged over labelled rows only**, so its weight per labelled sample equals upstream's whenever a batch has at
   least one labelled row; batches with none contribute no scorer term.
4. Upstream quirk kept as is: `trajectory_loss_anchors` zips the 4 refinement stages with the per-sample index list (`:226`), so a
   batch smaller than 4 would silently drop stages. Batch size stays ≥ 4.
5. Upstream's validation step (`agent_lightning_module.py:54-88`) scores every token against a metric cache; Step 3 needs its own
   evaluation loop for raw-only held-out samples (not written yet).
6. The brief names `03_grade_candidates/s1_masked/`; on disk the 640 samples are `{runs}/drivejepa_retrain/03_grade_candidates/s1_masked/cache`
   (work-folder record `03_grade_candidates/s21_masked/`).
7. Nothing was written under `adapters/`, `runs/` or `drafts/`. S3 files were read only (two graded samples, `vocab_10240.npy`,
   `s3_scorer.py`, `s3_common.py`).


---

# Steps 2–4 — slice, training, readouts (2026-10-06/07)

Pre-condition: S3's own verdict was "not SET-READY" (Z3, Z5); Sid accepted both (recorded in `04_training_set/REPORT.md`, S3.1) and asked
for Step 2 onward. Inputs as found on disk: train manifest `04_training_set/manifest.parquet` (126,981 rows; the brief calls it
`manifest_train.parquet`, which does not exist), val manifest `04_training_set/val/manifest_val.parquet` (608 rows), `vocab_10240.npy`.

| id | statement | result | verdict |
|---|---|---|---|
| S1 | train loss at epoch 30 ≤ 25 % of epoch 0 | 41.0 % (slice loss 1.795 → 0.735) | **fails** |
| S2 | held-out loss bottoms before epoch 30, then rises ≥ 5 % | minimum at epoch 29, then +3.2 % | **fails** |
| S3 | held-out minimum < held-out at epoch 0 | 1.68 vs 4.23 | holds |
| S4 | scorer unchanged by raw-only steps | bit-identical on 3 of 3 steps after training and on all 12 raw-only batches met during training | holds |
| S5 | highway: fastest proposal at the best epoch exceeds the epoch-0 ceiling on ≥ 50 % of anchors | 100 % of 64 (ceiling 15.8 m/s → mean fastest 24.4 m/s) | holds |
| S6 | held-out and train-slice drives disjoint | 0 shared chunks, clips, samples | holds |

**Verdict by the rule: BROKEN** (S1 fails). It is not LEAK: held-out loss (1.73) stays 2.4 × the slice loss (0.74). It is not
GENERATOR-STUCK.

**Why I do not think it is broken.** S1 and S2 both assume the model memorises 1,223 samples within 30 epochs. It did not get there:
the slice loss and the held-out loss are both still trending down at epoch 30 (held-out over epochs 25–30: 1.88, 1.84, 1.72, 2.28,
1.68, 1.73; noisy, with one spike at epoch 28), so there is no overfitting turn yet for S2 to detect. Three things slow memorisation here and are part of the brief or the
upstream recipe: the encoder is frozen (7.1 M trainable of 311 M parameters), GridMask augmentation is on, and lr is the upstream 1e-4
at batch 32 (41 steps per epoch, 1,230 steps in total). The loss also has a floor above zero by construction (a best-of-32 distance
plus pseudo-target and scorer terms). Whether to rerun longer or with a higher learning rate to see the turn is Sid's call; nothing was tuned.

## Step 2 — slice

Seed 20261007, 1 % of clips per (type, source), `s35_slice.py` → `slice_train.parquet`, `heldout_val.parquet`.

| type | source | clips in train | clips in slice | samples | with label 2 |
|---|---|---|---|---|---|
| A (highway) | nurec | 205 | 2 | 64 | 64 |
| B_pass | nurec | 89 | 1 | 32 | 32 |
| B_raw (pull-out) | raw | 283 | 3 | 96 | 0 |
| ordinary | nurec | 438 | 4 | 71 | 71 |
| ordinary | raw | 3,000 | 30 | 960 | 0 |
| **slice** | | 4,015 | **40** | **1,223** (0.96 %) | 167 |

Held-out = the entire val manifest, not sub-sampled: 20 clips, 608 samples (A 64, B_pass 65, ordinary 479), 225 with label 2, 20
chunks. Asserted: the slice passes the train guard and held-out the val guard (no exam / test); 0 chunks shared between the 20 val chunks
and the full list of 1,398 train chunks; 0 shared clips or samples. The assertions run again at training start.

## Step 3 — training setup

- **GPU route (the Step 1 blocker).** torch 2.8.0+cu128 is layered over the conda drive-jepa env through a PYTHONPATH overlay
  (`{runs}/alpasim_bulk/conda-envs/dj-torch-cu128-overlay`, 6.4 GB; the env itself is untouched). mmcv's compiled
  extension cannot load under that torch, so `s35_common._stub_mmcv_ext` puts a stub in its place and forces the pure-PyTorch
  deformable attention. **Parity before training** (`s35_parity.py`, 8 samples, eval mode, vs the shipped env on CPU): proposals differ by
  at most 0.13 mm, scorer outputs by 2e-5, same chosen plan on 8 of 8. The pure-PyTorch attention equals the compiled one exactly on CPU.
- **Recipe.** Upstream: Adam from `agent.get_optimizers`, lr 1e-4, batch 32, 16-bit mixed precision, no gradient clipping, GridMask on.
  Per brief: 30 epochs, encoder frozen, extended vocabulary, acceleration slots zeroed, init from the shipped checkpoint.
- **Vocabulary shim.** The hard-coded `./data/8192.npy` (`drive_jepa_agent.py:49`) is replaced by the path argument of the patched agent
  (Step 1b). Logged at start: `vocab_10240.npy shape (10240, 40, 3) float32`, `agent.anchors (10240, 8, 3)`.
- **Sample weights.** Every type-A sample is drawn twice per epoch: 1,287 draws from 1,223 samples. Share actually seen, identical every
  epoch: ordinary 80.1 %, highway 9.9 %, pull-out 7.5 %, pass 2.5 % (unweighted: 84.3 / 5.2 / 7.8 / 2.6 %).
- **Evaluation** every epoch in eval mode on the whole slice and the whole held-out set (`s35_train.py::evaluate`), plus once before training (epoch 0).
- Script `s35_train.py`, entry `run_s35.sh`; outputs `run/` (`train.log`, `curve.csv`, `readouts.parquet`, `summary.json`, `tables.md`, PNGs).

## Step 4 — readouts

### R1 — loss per epoch (total and per head)

| epoch | train total (training mode) | train generator | train scorer | slice total (eval mode) | slice generator | slice scorer (labelled) | held-out total | held-out generator | held-out scorer (labelled) |
|---|---|---|---|---|---|---|---|---|---|
| 0 |  |  |  | 1.79 | 1.66 | 0.543 | 4.23 | 3.85 | 0.59 |
| 1 | 2.7 | 1.58 | 0.616 | 1.27 | 1.14 | 0.525 | 2.81 | 2.43 | 0.569 |
| 2 | 2.2 | 1.31 | 0.527 | 1.14 | 1.01 | 0.516 | 2.61 | 2.26 | 0.546 |
| 3 | 1.97 | 1.21 | 0.505 | 1.1 | 0.984 | 0.473 | 2.53 | 2.18 | 0.543 |
| 5 | 1.83 | 1.08 | 0.464 | 1.08 | 0.97 | 0.44 | 2.45 | 2.08 | 0.587 |
| 10 | 1.55 | 0.953 | 0.441 | 0.952 | 0.844 | 0.438 | 2.11 | 1.76 | 0.55 |
| 15 | 1.5 | 0.869 | 0.419 | 0.804 | 0.708 | 0.391 | 1.88 | 1.53 | 0.546 |
| 20 | 1.36 | 0.834 | 0.387 | 0.822 | 0.715 | 0.421 | 1.84 | 1.5 | 0.536 |
| 25 | 1.25 | 0.778 | 0.379 | 0.87 | 0.776 | 0.38 | 1.88 | 1.5 | 0.622 |
| 29 | 1.27 | 0.758 | 0.363 | 0.742 | 0.647 | 0.356 | 1.68 | 1.34 | 0.536 |
| 30 | 1.22 | 0.726 | 0.38 | 0.735 | 0.637 | 0.382 | 1.73 | 1.33 | 0.565 |

Full curve: `run/curve.csv`; plots `run/curves_loss.png`, `run/curves_readouts.png`. Training-mode loss includes the GridMask augmentation; eval-mode numbers do not. Best held-out epoch: **29**.

### R2 — generator: distance between the chosen plan and the human path at 4 s (m), mean per type

| set | type | samples | epoch 0 chosen | epoch 0 closest of 32 | best (29) chosen | best (29) closest of 32 | last (30) chosen | last (30) closest of 32 |
|---|---|---|---|---|---|---|---|---|
| heldout | A | 64 | 39.2 | 32.4 | 22.8 | 11.6 | 20.2 | 11.5 |
| heldout | B_pass | 65 | 9.03 | 1.11 | 7.28 | 1 | 7.79 | 0.83 |
| heldout | ordinary | 479 | 8.53 | 1.61 | 6.53 | 0.65 | 6.21 | 0.65 |
| train | A | 64 | 29.4 | 18.7 | 12.2 | 1.58 | 8.44 | 1.49 |
| train | B_pass | 32 | 9.45 | 1.32 | 8.79 | 0.68 | 8.37 | 0.67 |
| train | B_raw | 96 | 5 | 0.9 | 4.71 | 0.46 | 4.53 | 0.42 |
| train | ordinary | 1031 | 8.14 | 1.23 | 5.52 | 0.6 | 5.74 | 0.65 |

Held-out has no pull-out (B_raw) clips, so the pull-out readout exists on the train slice only.

### R3 — scorer on samples with label 2

| set | epoch | samples | top1_agreement | top1_strict | rank_correlation_mean | anchors_with_flat_referee | referee_grade_of_chosen | referee_best_of_32 |
|---|---|---|---|---|---|---|---|---|
| heldout | epoch 0 | 225 | 23.6 % | 4.9 % | 0.046 | 4 | 0.793 | 0.876 |
| heldout | best (29) | 225 | 32.9 % | 5.3 % | 0.245 | 1 | 0.795 | 0.9 |
| heldout | last (30) | 225 | 29.8 % | 4.4 % | 0.281 | 1 | 0.791 | 0.887 |
| train | epoch 0 | 167 | 34.7 % | 0.0 % | 0.116 | 1 | 0.798 | 0.903 |
| train | best (29) | 167 | 34.7 % | 5.4 % | 0.48 | 2 | 0.892 | 0.943 |
| train | last (30) | 167 | 38.3 % | 4.2 % | 0.504 | 3 | 0.882 | 0.935 |

top-1 agreement = the scorer's pick has the referee's top grade among the 32 (ties count); strict = same index as the referee's first maximum. Rank correlation is Spearman per sample, averaged over samples where both vary.

| set | type | epoch | samples | top1 | rank_corr | ref_chosen |
|---|---|---|---|---|---|---|
| heldout | A | epoch 0 | 64 | 0.0 % | 0.083 | 0.675 |
| heldout | A | best (29) | 64 | 4.7 % | 0.123 | 0.783 |
| heldout | A | last (30) | 64 | 7.8 % | 0.123 | 0.747 |
| heldout | B_pass | epoch 0 | 65 | 38.5 % | 0.048 | 0.736 |
| heldout | B_pass | best (29) | 65 | 47.7 % | 0.297 | 0.69 |
| heldout | B_pass | last (30) | 65 | 46.2 % | 0.245 | 0.68 |
| heldout | ordinary | epoch 0 | 96 | 29.2 % | 0.02 | 0.909 |
| heldout | ordinary | best (29) | 96 | 41.7 % | 0.291 | 0.874 |
| heldout | ordinary | last (30) | 96 | 33.3 % | 0.411 | 0.896 |
| train | A | epoch 0 | 64 | 0.0 % | -0.211 | 0.761 |
| train | A | best (29) | 64 | 0.0 % | 0.397 | 0.896 |
| train | A | last (30) | 64 | 9.4 % | 0.482 | 0.885 |
| train | B_pass | epoch 0 | 32 | 56.2 % | 0.387 | 0.82 |
| train | B_pass | best (29) | 32 | 40.6 % | 0.514 | 0.861 |
| train | B_pass | last (30) | 32 | 43.8 % | 0.417 | 0.857 |
| train | ordinary | epoch 0 | 71 | 56.3 % | 0.289 | 0.821 |
| train | ordinary | best (29) | 71 | 63.4 % | 0.539 | 0.904 |
| train | ordinary | last (30) | 71 | 62.0 % | 0.562 | 0.891 |

### R4 — highway (type A) samples: fastest of the 32 proposals vs the human

| set | epoch | anchors | human_end_speed | fastest_mean | fastest_max | chosen_mean | fastest_over_human | chosen_over_human |
|---|---|---|---|---|---|---|---|---|
| heldout | epoch 0 | 64 | 21.9 | 14.2 | 17.1 | 11.1 | 0.66 | 0.49 |
| heldout | best (29) | 64 | 21.9 | 24.3 | 29 | 16 | 1.09 | 0.72 |
| heldout | last (30) | 64 | 21.9 | 22.8 | 31.2 | 14.4 | 1.02 | 0.62 |
| train | epoch 0 | 64 | 19.3 | 13.5 | 15.8 | 10.2 | 0.7 | 0.53 |
| train | best (29) | 64 | 19.3 | 24.4 | 31.3 | 15.6 | 1.23 | 0.82 |
| train | last (30) | 64 | 19.3 | 24.3 | 34.1 | 16.8 | 1.23 | 0.9 |

### GPU, time, projection

| item | value |
|---|---|
| epochs completed | 30 of 30 |
| wall time | 0.51 h of the 2 h cap |
| GPU memory, peak reserved / allocated | 4.5 / 4.5 GiB of the 24 GiB cap |
| seconds per step (batch 32), median / mean | 0.57 / 0.53 |
| steps | 1230 |
| per-epoch draws / samples | 1287 / 1223 |
| evaluation per epoch (slice + held-out) | 32 s |
| device / torch | cuda / 2.8.0+cu128 |
| vocabulary loaded | (10240, 40, 3) |

Effective per-type share actually seen per epoch (type A drawn twice): ordinary 80.1 %, B_raw 7.5 %, A 9.9 %, B_pass 2.5 % (identical every epoch; sampler plan {'ordinary': 0.8011, 'A': 0.0995, 'B_raw': 0.0746, 'B_pass': 0.0249}).

| epochs | steps | hours_training_only |
|---|---|---|
| 5 | 20845 | 3.3 |
| 10 | 41690 | 6.6 |
| 20 | 83380 | 13.1 |

Projection basis: full train manifest 126,981 samples + 6,415 repeated highway samples = 133,396 draws = 4,169 steps per epoch at batch 32, at the measured 0.57 s per step. It excludes per-epoch evaluation and assumes the same labelled share per batch; the full set is 17.2 % labelled against 13.7 % in the slice, and the referee call on labelled rows is part of the step time.

## Flags (Steps 2–4)

- **A. The pre-registered verdict is BROKEN, on S1 alone.** See the reading above; the evidence against a wiring fault is S3–S6, the
  parity check and the 15 Step-1 checks.
- **B. Highway is where the generator learned most.** Fastest proposal on highway anchors went from 14 m/s to 24 m/s on both the
  slice and held-out, and the chosen plan's 4 s error fell from 29 m to 8 m (slice) and 39 m to 20 m (held-out). The scorer still picks slow:
  chosen end speed is 0.62–0.72 of the human's on held-out.
- **C. Scorer agreement stays low.** Top-1 agreement with the referee is 30–33 % on held-out and 35–38 % on the slice; rank correlation rises
  from 0.05 to 0.25–0.28 (held-out) and 0.12 to 0.50 (slice). On highway held-out it is 5–8 % top-1.
- **D. No pull-out readout on held-out** (val has no pull-out clips); the slice's 96 pull-out samples go from 5.0 m to 4.5 m at 4 s.
- **E. Chosen-plan error barely moves for pass and pull-out samples** (9.5 → 8.4 m, 5.0 → 4.5 m), while the closest of the 32 is under
  0.7 m: the menu contains a good plan and the scorer does not pick it.
- **F. GPU route is an overlay, not a clean env.** It works and is parity-checked, but it depends on the mmcv stub; a call into any other
  compiled mmcv op would raise.
- **G. Epoch-0 "train loss" is measured in eval mode** (no optimiser step has run yet); S1 compares eval-mode slice loss at epoch 0 and 30.
  In training mode the ratio is 45 % (epoch 1 → 30).
- **H. Projection excludes evaluation and NAS read time at scale.** The slice was read with 12 workers; 4,169 steps per epoch on the full set
  will stress the NAS more than 41 did.
- **I. S5 is judged against the single epoch-0 ceiling** (the fastest proposal on any slice highway anchor, 15.8 m/s). It also holds
  per anchor (100 %).
- **J. Nothing was written under adapters/, runs/ or drafts/.** The vendored tree is untouched; `04_training_set` was read only.
