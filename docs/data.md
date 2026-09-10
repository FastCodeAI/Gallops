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
| **KITScenes** | 127 of 1,006 sequences, all `val` / `overlap_train_val` | **No** — roughly 40 minutes of front-camera driving, and no `train` split at all. |
| **NAVSIM / OpenScene** | `navtest` only: 136 logs, 12,146 scenes | **No** — evaluation split only. |

### nuScenes — usable, with caveats

Camera-only in practice. The local copy has **no `sweeps/` directory**, all five
radar directories are **empty**, and lidar is **7,044 keyframes short** of the
keyframe count. Camera-only training is viable; sensor fusion is not, and
generating `nuscenes_infos*.pkl` against a keyframe-only copy degrades silently
rather than failing.

### KITScenes — not a training set here

The full public release is ~4.2 TB across 1,006 sequences: 533 `train`
(2.38 TB), 206 `test`, 117 `val`, 127 `test-e2e`, 23 `overlap_train_val`. **We
hold none of `train`.** A full pull only fits on the largest available volume.

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
