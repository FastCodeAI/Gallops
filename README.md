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

Read [docs/STRATEGY.md](docs/STRATEGY.md) first. It carries what paddock's
results already established about what to build, and what this machine can and
cannot actually do today.

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

## Licence

Apache-2.0, matching paddock.
