<h1 align="center">
  <img src="docs/assets/gallops.svg" height="44" alt=""> Gallops
</h1>

<p align="center"><strong>Gallops trains driving policies. <a href="https://github.com/FastCodeAI/Paddock">Paddock</a> scores them. Neither repo imports the other.</strong></p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="#quickstart"><img src="https://img.shields.io/badge/gate-make_check_(no_CI_yet)-555555.svg" alt="Gate: make check, no CI yet"></a>
  <a href="#training-so-far"><img src="https://img.shields.io/badge/training_so_far-6_efforts-2E8B4E.svg" alt="Training so far: 6 efforts"></a>
  <a href="docs/reviewing.md"><img src="https://img.shields.io/badge/reviews-severity_scale-B4560A.svg" alt="Review standard"></a>
</p>

Gallops takes a driving model, trains it, and writes a checkpoint that paddock
scores on the same benchmarks, with the same machinery, that scored four vendor
models first. So far every run has started from the released Drive-JEPA and
retrained part of it. Our own model, gallops-v0, is designed
([docs/architecture.md](docs/architecture.md)) but not built yet.

Every checkpoint written here says what it is: which model, at what step, from
which commit, trained on which scenes. That record is how a number on paddock's
leaderboard traces back to the training that produced it, and most of the rules
in this repo exist to keep it true.

The two repos meet at a directory on disk:

```
{weights}/<model>/<train_run>/step_<n>/
    checkpoint.json      <- the entire contract
    model.safetensors
```

Gallops writes it. Paddock reads it, scores the model, and puts the result in
the same table as every other model. That separation is the point; see
[why](#why-two-repos).

| Where | What |
|---|---|
| [src/gallops/train/](src/gallops/train/) | the trainers: the KITScenes re-ranker, the swerve finetune, the scorer on cached features, and the preflight and safety checks they must pass |
| [src/gallops/expert/](src/gallops/expert/) | the expert that draws evasion and braking targets offline, and the grader every plan is judged by |
| [src/gallops/data/](src/gallops/data/) | training-data builders: the re-ranker's dataset, real nuScenes scenes as episodes |
| [src/gallops/drivejepa_b2d/](src/gallops/drivejepa_b2d/), [recipes/](recipes/) | the Bench2Drive training, as a recipe that prints or runs its commands |
| [experiments/drivejepa_retrain/](experiments/drivejepa_retrain/) | the Physical AI AV retrain, as the scripts that ran |
| [checkpoint.py](src/gallops/checkpoint.py), [config.py](src/gallops/config.py) | the checkpoint contract, and the only code that knows where anything lives on disk |

The names: a paddock is the enclosure where a horse is paraded and judged
before a race. The gallops is the training ground where it is made fast.

---

## Status

**gallops-v0, our own model, is not built yet.** Every training run so far has
started from the released Drive-JEPA checkpoint and retrained part of it, or
trained something small on top of it. [Training so far](#training-so-far) lists
all seven efforts. The loop is proven end to end: `gallops train-rerank` writes a
checkpoint with its sidecar, and paddock's `drivejepa_rerank` adapter loads it
and scores it like any other model.

Read [docs/STRATEGY.md](docs/STRATEGY.md) first. It carries what paddock's
results already established about what to build, and what this machine can and
cannot actually do today.

## Training so far

Status at 2026-10-21. Everything below starts from the released Drive-JEPA
checkpoint, with its image encoder frozen. Each row's doc has the full account;
the numbers here are copied from it, never re-derived.

| What was trained | Retrained on top of the release | Trained on | Headline | How far it holds | Doc |
|---|---|---|---|---|---|
| **Re-ranker** (`drivejepa_rerank`) | nothing: a new small MLP chooses among the frozen model's 32 candidates | KITScenes `train`: 318 scenes, plus 55 to choose the epoch | KITScenes ADE@3s **0.973 m**, against **2.521** for the model's own pick (223 samples) | **Strong, open loop:** paired t −8.0. **Worse in closed loop:** 1.85 against 3.01 on 9 neuroNCAP episodes | [STRATEGY.md](docs/STRATEGY.md#first-target--taken-drive-jepas-selection-not-its-head), paddock's `docs/models/drivejepa_rerank.md` |
| **Bench2Drive Split-v4** | the scorer head (1.32 M parameters) | Bench2Drive base set, 150 clips held out | **DS 68.1** on 50 routes | **Selection on the test routes**, one run per route; paddock cannot score it yet | [drivejepa_b2d.md](docs/drivejepa_b2d.md) |
| **Bench2Drive Final** | the proposal generator, the split head, the area head | as Split-v4, plus MTD-H teacher paths | **DS 73.6** on 50 routes | as Split-v4. Its three training stages have not been rerun from this repo | [drivejepa_b2d.md](docs/drivejepa_b2d.md) |
| **Physical AI AV retrain** | run 1: generator and scorer; run 2, shipped: generator only | NVIDIA Physical AI AV `train`: 126,981 samples from 4,015 clips | AlpaSim mean `scene_score` **0.5433 → 0.7697** on 30 scenes | **One drive per scene.** 3 at-fault collisions, 2 of them new | [drivejepa-retrain.md](docs/drivejepa-retrain.md) |
| **Swerve finetune**, 11 runs | the refiner, the scorer, or both | 5 reconstructed nuScenes train scenes with inserted hazards: 9,495 + 5,426 + 9,263 labelled ticks | 0103: v4 **4.53** against **4.05** released. Full neuroNCAP set, s1: **3.671** against **3.568** | **Camera-blind.** 0103 is what chose v4, and its gain is one episode; s1's is +0.10 with SE 0.08. No evasion learned | [swerve-finetune.md](docs/swerve-finetune.md) |
| **Auxiliary heads** (`drivejepa_aux`), 9 runs | the refiner path (2.39 M); upstream's map and box heads switched on and two new heads, all for training only | nuScenes `train`: 16,952 keyframes of 577 scenes, 102 more held out | Picked plan's L2 at 3 s on the held-out scenes: T2 **2.93 m**, against **3.76** for the same fine-tune without heads and **3.79** released (2,996 keyframes, a fixed epoch) | **Open loop, in domain.** The gain is in which candidate the frozen scorer picks; best-of-32 is unchanged. Inputs not yet checked against paddock's, no gate run | [drivejepa-aux-heads.md](docs/drivejepa-aux-heads.md#results-2026-10-21) |
| **Scorer on cached features** | the scorer, with sub-score heads | the same rendered episodes, from a feature cache (21,934 ticks × 64 plans) | Held-out collision ranking AUC **0.66–0.83**, against 0.50–0.59 released | **Camera-blind, open loop only.** Breaks on real nuScenes logs. Its one closed-loop gain was a drift, not an avoidance | [scorer-cache.md](docs/scorer-cache.md) |

How to read it:

- **Nothing here is yet a closed-loop safety gain** shown on a split nothing was
  chosen on. The strongest result is the re-ranker's open-loop gain on KITScenes,
  and the same weights do worse on neuroNCAP.
- **The auxiliary heads and the re-ranker say the same thing:** Drive-JEPA's
  error sits in which of its 32 candidates it picks, not in the candidates.
  Both moved the pick; neither has been through paddock's gate.
- **Every swerve and scorer-cache run was camera-blind.** Paddock and the
  trainer both placed the cameras relative to nuScenes' LIDAR_TOP, so no image
  reached the model; a black frame gave the same plan. Those numbers are what
  the runs measured, but they say nothing about learning to *see* a hazard. The
  trainer is fixed and its preflight now refuses a model that a blanked camera
  does not move. Paddock's half is #62, still open.
- **The swerve runs also trained at batch 8, and scorer cache v1 was built at
  batch 4.** The vendor model mixes ticks within a batch, which did not matter
  while it was blind but would from the first run that sees. Every forward pass
  is now one tick, so any rerun trains differently (see swerve-finetune.md).
- **Several shipped sidecars are known to be wrong about their own run,** and
  were left as written rather than hand-edited. Each doc lists them.
- **Bench2Drive and AlpaSim runs are one run per route or scene,** so a
  difference of a few points is not a result on its own.

## Quickstart

```bash
make setup                                  # .venv + install (no torch)
source .venv/bin/activate

cp gallops.toml.example gallops.toml
$EDITOR gallops.toml                        # point at YOUR weights, runs and data

gallops config                              # what this machine resolved
gallops show <checkpoint dir>               # what a checkpoint claims about itself
gallops train-rerank <dataset.npz> --train-run <name>   # see gallops/train/rerank.py
python -m gallops.data.rerank keyframes|dataset ...     # its training data; see gallops/data/rerank.py
```

Point `[roots.weights]` at **the same directory as paddock's
`[roots.weights]`**. That is the whole mechanism by which paddock finds a model
this repo trained.

The training stack is a separate extra, so a machine with no GPU can still run
the gate:

```bash
uv pip install --python .venv/bin/python -e ".[train]"
```

## Why two repos

Four reasons, in increasing order of importance:

1. **Engineering.** Paddock's dependency list is six packages and its `make
   check` runs in a clean environment with no GPU. Putting torch in there makes
   its gate heavy and breaks the invariant that paddock never imports a model.
2. **Data isolation.** The strategy is to attract OEMs who bring their own data.
   That data arrives under terms that do not permit it to sit beside a repo we
   publish results from. The separation has to exist before the first
   conversation, not after.
3. **Our own numbers stay trustworthy to us.** Paddock's claim is that an ADE
   and an NCAP score are "statements about the same model, run the same way".
   The moment our model gets a privilege the vendors' do not have — in-process
   access, a special data path, a shortcut past the HTTP node — we lose the
   ability to tell whether we are actually gaining on them.
4. **Paddock's credibility is the commercial asset.** A leaderboard maintained
   by the party that also trains the winning entry is marketing. A leaderboard
   where our model is one adapter among six, scored by machinery that predates
   it and scored four competitors first, is evidence. **Keeping the scorer
   separate from the trainee is what makes the numbers persuasive to somebody
   who does not work here.**

## Documentation

| Doc | Read it when |
|---|---|
| [docs/STRATEGY.md](docs/STRATEGY.md) | **start here** — why this exists, what the evidence says, what to build |
| [docs/architecture.md](docs/architecture.md) | designing or changing the model — what gallops-v0 is, and which choices are measured |
| [docs/checkpoints.md](docs/checkpoints.md) | writing or reading a checkpoint; the paddock contract |
| [docs/conventions.md](docs/conventions.md) | before training anything — frames, splits, what counts as a result |
| [docs/data.md](docs/data.md) | what data exists, what does not, and what must be fetched |
| [docs/reviewing.md](docs/reviewing.md) | reviewing a PR — what gets checked, and the 🔴 / 🟠 / 🟡 scale |
| [docs/swerve-finetune.md](docs/swerve-finetune.md) | the first training experiment end to end — results, and the defects the measurements exposed |
| [docs/scorer-cache.md](docs/scorer-cache.md) | training Drive-JEPA's scorer on cached features — what it measured, and why it is camera-blind |
| [docs/drivejepa-retrain.md](docs/drivejepa-retrain.md) | the Drive-JEPA retrain on Physical AI AV — what was trained, on what, and what it measured |
| [docs/drivejepa_b2d.md](docs/drivejepa_b2d.md) | the Drive-JEPA Bench2Drive recipe — lineage, what it reads, what was verified |

## Licence

Apache-2.0, matching paddock.
