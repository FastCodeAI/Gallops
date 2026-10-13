# Gallops

**Training for driving policies. Scored by [paddock](../paddock); imported by nobody.**

Paddock is the enclosure where a horse is saddled and paraded before a race —
where you look at it and judge it. The gallops is the training ground, where it
is made fast.

The two repos never import each other. They meet at a directory on disk:

```
{weights}/<model>/<train_run>/step_<n>/
    checkpoint.json      <- the entire contract
    model.safetensors
```

Gallops writes that. Paddock reads it, scores the model on the same benchmarks
that scored four vendor models before it, and puts the result in the same table.
That separation is the point — see [why](#why-two-repos).

---

## Status

**One trainer, and the loop proven end to end.** `gallops train-rerank` fits a
selector over Drive-JEPA's 32 candidate trajectories and writes a checkpoint
with its sidecar. Paddock's `drivejepa_rerank` adapter loads it and scores it
like any other model. It is small on purpose: its job was to prove
train → checkpoint → sidecar → paddock row, and it also closes a measured gap
(see [docs/STRATEGY.md](docs/STRATEGY.md)). Beyond that there is the
site-config split, the checkpoint contract, and the conventions a training run
has to respect for its numbers to mean anything.

**The Drive-JEPA retrain on NVIDIA Physical AI AV is recorded outside the
package.** It is under
[experiments/drivejepa_retrain/](experiments/drivejepa_retrain/) as the scripts
that ran, in folders numbered in the order the work was done, and is described
in the next section.

**One recipe:** the training behind Drive-JEPA's Bench2Drive results. It is not
gallops-v0; it was moved here so it is tracked and portable. See
[docs/drivejepa_b2d.md](docs/drivejepa_b2d.md).

Read [docs/STRATEGY.md](docs/STRATEGY.md) first. It carries what paddock's
results already established about what to build, and what this machine can and
cannot actually do today.

## Drive-JEPA, retrained on Physical AI AV

The full account is [docs/drivejepa-retrain.md](docs/drivejepa-retrain.md).

**Approach.** Start from the released Drive-JEPA checkpoint and train it
further on NVIDIA's Physical AI AV driving clips, keeping its image encoder
frozen.

Drive-JEPA has two parts: a generator that proposes 32 candidate paths, and a
scorer that rates them and picks one. Both were trained together in the first
training run. The scorer did not pass that run's own quality check, so it was
frozen for the second training run, which retrained only the generator. The
shipped checkpoint therefore contains the first run's scorer unchanged.

Three things change in what the model is given, in training and again when it
drives. The bottom 37 rows of each camera frame are blacked out (the bonnet
mask: the recording car's own bonnet is in every training frame and the
simulator never draws one). Acceleration is fed as zero. And the list of path
shapes the generator builds its candidates from grows from 8,192 to 10,240, to
add shapes taken from human driving at 14 m/s and above.

**Data and splits.** NVIDIA Physical AI AV. Training reads the `train` split
only: 126,981 samples from 4,015 clips. Each run makes several passes (a pass
is one sweep through the training data) and keeps one of them: which pass of
the first run to keep was chosen on `val`; which pass of the second, on 145
`train` clips held out of that run. `test` is excluded. So are the simulator's 913 public scenes, which
include the 30 scenes the result below is measured on; every step's scripts
assert both exclusions, and `tests/test_drivejepa_retrain.py` re-checks them
from the committed clip lists.

**Result.** A closed-loop test: the model's plan steers a simulated car, and
what it sees next depends on where that plan took it. 30 scenes in the AlpaSim
simulator, one drive per scene:

<!-- table:results -->
| Adapter | Weights | Mean `scene_score` | At-fault collisions | Any collision | Off-road | Stalls | Mean progress | Mean distance (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1.4b | released checkpoint, momentum-aware selection | 0.5433 | 4 / 30 | 10 / 30 | 4 / 30 | 3 / 30 | 0.5366 | 75.8 |
| 0.1.6 | intermediate checkpoint, not shipped (first training run) | 0.7032 | 2 / 30 | 7 / 30 | 3 / 30 | 4 / 30 | 0.6977 | 115.4 |
| 0.1.7 | retrained checkpoint (second training run, generator only) | 0.7697 | 3 / 30 | 3 / 30 | 2 / 30 | 2 / 30 | 0.8284 | 151.6 |
<!-- /table:results -->

Read the first row against the last. The version numbers name the driver that
feeds the model in the simulator: 0.1.4b is the released checkpoint with
momentum (a step from the Drive-JEPA paper that favours candidates consistent
with the plan already being driven); 0.1.7 is the same driver with the
retrained checkpoint. The middle row, 0.1.6, drove the first training run's
checkpoint, was never shipped, and is shown once so the two runs can be told
apart. An at-fault collision is one the scoring attributes to the driven car.
For scale, a constant-velocity baseline (a car that keeps its current speed and
heading, with no model) scores 0.4876 on the same 30 scenes; it ran with one
simulator setting different and is a reference only.

Each column is narrower than its heading. Collisions and off-road are counted
only up to the point where a scene's scoring stops; over the whole clip 0.1.7
has 6 scenes with a collision and 4 off-road, not 3 and 2. "Stalls" does not
count the two scenes where the car never moves under any version. One scene
scores 0 for any driver, and one "at-fault" collision is pedestrians walking
into the stationary car. And 0.1.7 against 0.1.6 is 0.066 with a standard
error of 0.068, so the table does not show that the second run beat the first.
The definitions are in the experiment doc, under "How to read the columns".

Most steps of the experiment did not pass the checks they set themselves
beforehand, and the experiment doc has that table too.

Stated plainly:

- **The spacing term was switched off.** The second run's code has a loss term
  that penalises unevenly spaced waypoints. It exists in the code but was
  switched off (weight 0) for the shipped checkpoint.
- **0.1.7 has three at-fault collisions and two of them are new.** The mean
  rose; one drive per scene cannot say it is safer.
- **The committed code has not been re-run** since its machine paths were
  replaced by placeholders.

**Running it.** Gallops holds no driver. The weights are run in the simulator
by paddock's Drive-JEPA driver at `tools/models/drivejepa/alpasim_driver/`
(version 0.1.7; paddock's `docs/models/drivejepa.md` says what it feeds the
model and how to launch it).

**Weights.** `{weights}/drivejepa_physicalai/second-run-generator/step_000000002/drive_jepa_physicalai_generator.ckpt`,
1,273,002,536 bytes, sha256
`29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916`, with a
`checkpoint.json` beside it that declares `trained_on: physicalai_av / train`.
Paddock has no `datasets.yaml` key for that dataset yet, and the sidecar's
`train_git_sha` was set after the fact; both are explained in the experiment
doc. Every other artefact is in
[experiments/drivejepa_retrain/POINTERS.md](experiments/drivejepa_retrain/POINTERS.md).

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
| [docs/swerve-finetune.md](docs/swerve-finetune.md) | the first training experiment end to end — results, and the defects the measurements exposed |
| [docs/drivejepa-retrain.md](docs/drivejepa-retrain.md) | the Drive-JEPA retrain on Physical AI AV — what was trained, on what, and what it measured |
| [docs/drivejepa_b2d.md](docs/drivejepa_b2d.md) | the Drive-JEPA Bench2Drive recipe — lineage, what it reads, what was verified |

## Licence

Apache-2.0, matching paddock.
