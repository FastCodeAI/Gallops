# Data

What exists, what does not, and what a training run has to record about it.

**No paths here.** Where a dataset lives is `gallops.toml`, and that file is
gitignored — see [../CLAUDE.md](../CLAUDE.md) for the invariant and
`tests/test_portability.py` for the gate that enforces it. Use the same dataset
*names* paddock uses, because those names are what a checkpoint's `trained_on`
records and what paddock cross-checks a benchmark against.

## What we hold today

Measured on this workspace, not assumed.

| dataset | held here | usable for training |
|---|---|---|
| **nuScenes** | full 850 scenes, 34,149 keyframes, ~54 GB | **Yes** — the one real corpus we have. |
| **KITScenes** | 127 of 1,006 sequences from `val` / `overlap_train_val`, and since 2026-09 at least 373 of the 533 `train` sequences; 532 by 2026-10-16 | **Partly** — enough to train the re-ranker (5,711 keyframes from those 373) and the KITScenes retraining (13,937 moments from 482 scenes). |
| **NAVSIM / OpenScene** | `navtest` only: 136 logs, 12,146 scenes | **No** — evaluation split only. |

### nuScenes — usable, with caveats

Camera-only in practice. The local copy has **no `sweeps/` directory**, all five
radar directories are **empty**, and lidar is **7,044 keyframes short** of the
keyframe count. Camera-only training is viable; sensor fusion is not, and
generating `nuscenes_infos*.pkl` against a keyframe-only copy degrades silently
rather than failing.

### KITScenes — part of `train`

The full public release is ~4.2 TB across 1,006 sequences: 533 `train`
(2.38 TB), 206 `test`, 117 `val`, 127 `test-e2e`, 23 `overlap_train_val`. **We
hold part of `train`:** at least the 373 sequences the re-ranker trained on, the
first to finish downloading in 2026-09. A full pull only fits on the largest
available volume. Since the subset follows no rule, a checkpoint trained on it
records the scenes it read (`notes.train_scenes`), not just the split.

By 2026-10-16, 532 of the 534 scenes the `default_geo_split_v1_0` `train.txt`
lists were on disk (two are missing). The KITScenes retraining
([H07](../experiments/H07_encoder_lora/README.md)) trained on 482 of them and
held 50 out to choose checkpoints; both lists are committed in
`experiments/H07_encoder_lora/02_encoder_build/out/splits.json`. It read
`val` and `overlap_train_val` only for its 223-moment held-out comparison, and
never read `test` or `test-e2e`.

Two properties that matter for training:

- **Tiers are not necessarily nested.** The copies here are genuinely disjoint
  in places — one tier holds a pruned working set of one scene list, another
  holds a different scene list with the full sensor suite. `Dataset.resolve()`
  returns the *first tier that exists*, so "the dataset" is not a well-defined
  object. **Record the scene list you actually read.**
- **Rigs are recalibrated between sessions.** Paddock found three distinct
  front-camera rotations across 60 `val` scenes, up to a degree apart. Anything
  derived from calibration is per-episode state.

Also worth knowing: the upstream release is marked **v1.0.1, "Early release"**,
with a note that files, annotations and splits may still change and that final
benchmark reporting should wait for a more stable release.

### NAVSIM — the most valuable corpus we do not have

`navtrain` / `trainval_sensor` is a separate, much larger download and is
absent, as is `synthetic_scenes/` (needed for the two-stage scoring the public
leaderboard reports). Since NAVSIM at n = 12,146 is the highest-resolution
benchmark in the suite, **its training split is probably the most valuable data
acquisition available to us.**

Note OpenScene is a redistribution of **nuPlan**, not nuScenes — a different
collection campaign, rig, cities and map format. The nuScenes copy cannot serve
it.

## NVIDIA Physical AI AV — used by the Drive-JEPA retrain

Not one of the three corpora above, and not held the way they are: it is
an access-restricted Hugging Face dataset, read by range request and cached per step, never
mirrored whole. What the retrain read, and how it split it, is recorded in
[GC05's account](../experiments/GC05_physicalai_retrain/README.md) and, step by step, under
`experiments/GC05_physicalai_retrain/`.

| | |
|---|---|
| Access | restricted; each site accepts the terms on its own account. No frames, clips, trajectories, coordinates or arrays from it are in this repository. Tables derived from it are: lists of clip ids, a per-clip table of id, chunk, split and country (`04_training_set/clips.csv`), and per-event measurements computed from its ego motion and object labels, such as speeds, clearances and actor sizes (`01_find_clips/d3_around.csv`). Whether the dataset's terms allow publishing those is for whoever accepted the terms to confirm before this repository is opened beyond the people who have |
| Splits | NVIDIA's own `train` / `val` / `test`, by chunk. Every one of the 3,146 chunks lies in exactly one split (asserted) |
| Read for training | `train` only: 4,015 clips → 126,981 samples (4,018 converted; three yielded none). The list of converted clips is `experiments/GC05_physicalai_retrain/04_training_set/clips.csv` |
| Read for choosing a checkpoint | `val`: 801 clips → 24,657 examples |
| Never read | `test`; and the AlpaSim simulator's 913 public scenes (`experiments/GC05_physicalai_retrain/01_find_clips/exam_exclusions.txt`), which are reconstructions of clips from this dataset and are what paddock's AlpaSim benchmark scores on |
| Name in a sidecar | `physicalai_av` |

**`physicalai_av` is not a key in paddock's `datasets.yaml`.** Everything above
about names matching paddock's applies in reverse here: a checkpoint that
declares `trained_on: physicalai_av` is making a declaration paddock cannot yet
cross-check against a benchmark's scene list. The exclusion of the evaluation
scenes is enforced by assertions in the training scripts and re-checked from
the committed lists by `tests/test_drivejepa_retrain.py`. Registering the
dataset in paddock, so that the check runs there, is open.

## Synthetic data

`cosmos-transfer2.5` and `Cosmos-Drive-Dreams` are already on this machine,
unevaluated. Given the size of the real-data gap, generation is a live option
rather than a hypothetical one, and it is worth costing before committing to a
multi-terabyte download.

## What a training run must record

In the checkpoint sidecar (`trained_on`), and in its own logs:

- Every `(dataset, split)` that contributed, **including splits used only for
  early stopping**. A split you selected on is not a split you can report on.
- The **scene or sample list**, not just the dataset name — because of the tier
  problem above. A hash of the sorted sample-id list is the cheap version, and
  it is exactly what paddock stores as `sequences_sha256` so the two can be
  compared.
- Any subsetting or filtering, and why.

Absent that, a result cannot be reconciled with anyone else's, which paddock
records as the single most common cause of numbers that will not agree.
