# GC01 — the KITScenes re-ranker

The first thing gallops trained. It is a small selector that picks among released
Drive-JEPA's 32 candidate trajectories. The model itself stays frozen. This page
summarises facts from the sources below and adds no analysis of its own. Every
number is copied from them, and the full account is paddock's model page.

Sources:
- [docs/STRATEGY.md](../../docs/STRATEGY.md), the section "First target — taken:
  Drive-JEPA's selection, not its head".
- The re-ranker row of this repo's README, "Training so far".
- The docstrings of [src/gallops/train/rerank.py](../../src/gallops/train/rerank.py)
  and [src/gallops/data/rerank.py](../../src/gallops/data/rerank.py).
- In paddock: `docs/models/drivejepa_rerank.md`, `docs/ROADMAP.md`,
  `docs/models/comparison.md` and
  `src/paddock/adapters/models/drivejepa_rerank/model.yaml`.
- On paddock branch `analysis/drivejepa-ncap-candidates`:
  `docs/models/drivejepa-neuroncap.md`.
- The checkpoint's own `checkpoint.json`.

## Why a selector

From 16 to 19 Sep 2026 paddock measured why Drive-JEPA is last on KITScenes. The
cause is not a speed head:
- The best of its 32 candidates lands 0.37 m ADE from the human on average.
- The candidate its scorer picks lands 2.36 m away, about 15th of 32 on average.
- That scorer was trained only on NAVSIM's PDM score.

The inputs were checked first: acceleration, route command, ego origin and lens
distortion. None moved the pick beyond noise. So the first target became a
re-ranker, and not the longitudinal head the strategy first proposed.

## What was built

- **Model.** A one-hidden-layer MLP with 128 units, written in numpy. It scores
  each candidate from 18 features, all taken from what the model already outputs:
  - the scorer's six logits
  - the trajectory's shape
  - ego speed
- **Label.** Each candidate's ADE@3s to the logged future, turned into a soft
  target with softmax(−ADE). The loss is listwise cross-entropy.
- **Epoch.** Chosen by the pick's ADE on 55 held-out *train* scenes, not by the
  loss.
- **Data.** 5,711 keyframes from 373 KITScenes train scenes: 318 scenes to train
  on and 55 to choose the epoch. The scenes are listed in
  [rerank_train_scenes.txt](../../src/gallops/data/rerank_train_scenes.txt).
  Paddock dumped the candidates through its own KITScenes harness, with the
  cameras cropped onto the OpenScene pinhole.
- **Settings.** Hidden size and temperature won a sweep over
  {16, 32, 64, 128} × {0.3, 0.5, 1.0}, ranked on the held-out train scenes. The
  sweep also computed every configuration's benchmark score, and the held-out
  winner was the benchmark's best as well. Paddock's page therefore notes that
  the benchmark "was not unseen".
- **Checkpoint.** `{weights}/drivejepa_rerank/kitscenes-train-h128-t1/step_000000325/`,
  trained at gallops commit `dfa7459`. Its `reranker.npz` has sha256 `2050b799e8e8…`.
  Paddock's `drivejepa_rerank` adapter pins this directory.

## What it measured

| Where | Re-ranker | Against | Paired | Kind |
|---|---:|---|---|---|
| KITScenes ADE@3s, 223 samples | **0.973** | 2.521, the model's own pick on the same path | −1.548 m, SE 0.174, t −8.0 | adapted, open loop |
| KITScenes ADE@3s, 223 samples | 0.973 | 1.159, the candidate closest to constant velocity | −0.186 m, t −4.9 | adapted, open loop |
| nuScenes val cumulative L2, 5,119 samples | **0.587** | 0.641, constant velocity | −0.054 m, t −2.8 | zero-shot, open loop |
| nuScenes val cumulative L2 | 0.587 | 1.360, `drivejepa` with the camera fix | −0.772 m, t −16, but mixes the camera crop with the selector | zero-shot, open loop |
| neuroNCAP scene 0103, 9 episodes, seeing model | 1.85 | 3.01, `drivejepa` | hit in 6 of 9 against 5 | smoke, a subset |

- **Where it wins on KITScenes:** the model picks a plan at about half the
  current speed on an open road.
- **Where it loses:** the human slowed for something ahead, and the re-ranker
  kept the current speed. Its only view of the scene is the scorer's logits.
- **Before the seeing-model smoke:** an offline replay over camera-blind
  neuroNCAP candidate dumps pointed the same way. It rescued 0 of the 322 ticks
  where a clear candidate existed, and it never picks a stop.

## Limits, as the sources state them

- **Open loop only.** The selector learned closeness to a human path and nothing
  about collisions. Its `model.yaml` declares only KITScenes and nuScenes until a
  full neuroNCAP run pairs it with `drivejepa` at the scenario-pair level.
- **Adapted on KITScenes, not zero-shot.** It saw that dataset's train split,
  though none of the benchmark's scenes.
- **NAVSIM is expected to get worse.** NAVSIM scores PDM, which the original
  scorer was trained for. This has not been measured.
- **Not tracked:** the neuroNCAP 0103 run
  (`neuroncap/drivejepa_rerank/20261012T231620-01570a`). Only its numbers are on
  paddock's model page.
