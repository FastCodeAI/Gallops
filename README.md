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

**Skeleton, plus one recipe.** gallops-v0 has no training code yet,
deliberately. What exists is the site-config split, the checkpoint contract, and
the conventions a training run has to respect in order for its numbers to mean
anything. Those are cheap now and expensive to retrofit.

The first recipe is not gallops-v0: it is the training behind Drive-JEPA's
Bench2Drive results, moved here so it is tracked and portable — see
[docs/drivejepa_b2d.md](docs/drivejepa_b2d.md).

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
| [docs/drivejepa_b2d.md](docs/drivejepa_b2d.md) | the Drive-JEPA Bench2Drive recipe — lineage, what it reads, what was verified |

## Licence

Apache-2.0, matching paddock.
