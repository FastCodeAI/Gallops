# Training the scorer on cached features

What came after s3 ([swerve-finetune.md](swerve-finetune.md)): can Drive-JEPA's
scorer learn to pick a safe plan on a scene it has never seen? Measurements
unless marked as an argument, as [conventions.md](conventions.md) asks.

> **Correction (2026-10-12): every cache here was camera-blind.** `scorer_cache`
> rebuilds inputs with `swerve_data.py`, which placed the cameras relative to
> nuScenes' LIDAR_TOP (yawed −90°), so no image reached the refiner. Found by testing
> whether the scorer's answers change when only the hazard car changes: on GT drives,
> where 30 runs of a scene share every pose, the cached tokens were bit-identical
> across runs whose images differ by up to 23% of pixels. So the scorer learned from
> plan geometry and ego motion only — the "place, not the car" result below. Caches
> v1, v2 and v3_nusc must be rebuilt; the grader, labels' geometry, vocabulary and
> the real-nuScenes conversion are unaffected. A cache now records a key (camera
> frame, code sha, `--batch`, vocabulary sha256, checkpoint, calibration source) in
> its manifest and in every file: `scorer_cache` refuses to resume under another
> key, `scorer_labels` and `scorer_folds` refuse a cache without `camera_frame:
> ego`, and labels and `folds_input.npz` are reused only for the build they came from.

**Status (2026-10-12): not yet.** Open loop, a scorer trained on cached features
ranks a held-out scene's plans far better than chance for the first time, and
rescues some frontal moments. It does not survive real nuScenes data, and the
one closed-loop gain on 0103 turned out to be a drift set off by the scene, not
an avoidance.

## Why a cache

s1–s3 retrained the scorer on four rendered scenes, and its held-out ranking loss
ended where a random ranking lands (3.45–3.58, against log 32 = 3.47): it
memorised the scenes it saw. Each step also reran the 300 M-parameter encoder, so
a run saw only about 2,000 steps. With the encoder, queries and refiner frozen,
everything upstream of the scorer is a fixed function of the tick, so it is
computed once and training becomes minutes over arrays.

## What was built

| module | what it does |
|---|---|
| [expert/fastgrade.py](../src/gallops/expert/fastgrade.py) | the grader over arrays, held exactly to `grader.grade` by tests and a per-episode re-check; adds the sub-scores the scorer's unused logits have slots for (hit within 5 s, time to collision, …) |
| [train/scorer_cache.py](../src/gallops/train/scorer_cache.py) | pass 1 runs the released model; pass 2 reruns only the refiner's last stage for a vocabulary of extra plans; stores what the scorer reads |
| [train/scorer_labels.py](../src/gallops/train/scorer_labels.py) | grades every cached plan against **every** vehicle, the lane centre and the drivable raster |
| [train/scorer_folds.py](../src/gallops/train/scorer_folds.py) | leave-one-scene-out training: a control (s1's target and ranking loss) against sub-score heads, and a gated pick (the released plan unless the collision head vetoes it) |
| [data/nuscenes_real.py](../src/gallops/data/nuscenes_real.py) | real nuScenes train scenes written as episodes the cache reads, graded against where their vehicles really went |

```bash
python -m gallops.train.scorer_cache  --source <labelled dir> <name> [--source ...] --vocab <vocab.py> \
                                      --batch 1 --out <cache dir>        # batch 1: see Open
python -m gallops.train.scorer_labels --cache <cache dir> --drivable <dir>
python -m gallops.train.scorer_folds  --cache <cache dir> --labelled <name> <labelled dir> [--labelled ...] \
                                      --checkpoint <released ckpt> --out folds.json
```

The cache holds 450 rendered episodes, 21,934 ticks × 64 plans (32 native, 32
vocabulary). The released scorer on the cached features reproduces its own picks
100% (logit drift 0.0012).

## What it measured

All open loop, each scene held out in turn, 5 seeds. "Rescued": the released
pick hits, the hazard is in view, and the new pick is safe for 5 s.

- **The scorer generalises a little for the first time.** Held-out collision
  ranking AUC 0.66–0.83, against 0.50–0.59 for the released final score. Ranking
  AUC is about 0.8 from a single training scene; rescues rise from 2–26% to
  17–36% with four scenes and have not levelled off.
- **Gated pick** (keep the released plan unless the collision head is under 0.5):
  kept safe 90–100% per held-out scene, frontal rescued 2–29%, stationary 4–48%.
  The two Singapore scenes (0053, 0058) are weakest.
- **Frontal is limited by what the plans see.** More than 2 s before contact the
  car is 35–45 m away; the released pick reaches 8–13 m and straight swerves 15–25
  m, so no plan's features cover it. Long probe paths, attention and plan geometry
  lifted the average but not the frontal rescues beyond 2 s.
- **Real nuScenes data breaks it either way.** With real scenes in training, the
  rendered frontal lesson disappears (real logs teach "car ahead, short braking
  plan, safe"). With real scenes masked from the collision label, the rendered
  scorer works but fires on 55% of real held-out ticks and keeps only 68% of them
  safe. At 0.2 weight: no middle ground found.
- **Closed loop, 0103, relative rule** (switch when the best plan's collision
  score beats the released one's by 0.2): frontal 2.13 / 5.0 / 5.0 against
  2.07 / 2.26 / 2.08. **Not an avoidance.** Every switch happens in the first 3
  s, identically in the frontal, stationary and side runs, so the scene or the
  episode start sets it off, not the car; it drifts the ego about 2 m left, and the
  oncoming car misses an ego already displaced. Any closed-loop gain must be
  checked for switches that differ between scenarios before it is credited.

## Open

- **The vendor model mixes ticks within a batch; the cause is known now.** At batch
  1 the rebuilt plans match the driven ones to 5e-5 m; at batch 4, by up to 1.0 m.
  The vendor's spatial cross-attention chooses which queries sample each camera from
  `mask_per_img[0]`, batch element 0's mask, for every element of the batch (verified
  by reading `bevformer/spatial_cross_attention.py`; reproduced on that module alone
  in #8's review: a tick second in a batch behind one whose points miss the frame
  takes no image features). Every gallops training run so far used batch 8, so each
  tick after the first in a batch read the image along another tick's plans. v1 was
  built at batch 4 and v2 at batch 1. Every pass is now one tick (`TICKS_PER_PASS` in
  `swerve_data.py`): `scorer_cache`, the trainer, `reach` and the preflight refuse
  `--batch` above 1 until the selection is made per element.
- **More scenes, half of them Singapore**, if any closed-loop result holds up.
- **An explicit motion cue** (a lead car against an oncoming one) for the scorer,
  argued, not built.
