# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this repo is

Gallops trains driving policies. [Paddock](../paddock) scores them. **Neither
imports the other**, and the whole interface between them is a JSON file beside
a checkpoint — see [docs/checkpoints.md](docs/checkpoints.md).

If you are new here, read [docs/STRATEGY.md](docs/STRATEGY.md) before writing
code. It carries what paddock's existing results already established about what
is worth building, and the two constraints that bound everything: one shared,
saturated GPU, and almost no training data on disk.

## Commands

```bash
make setup          # uv venv at .venv + editable install of [dev]  (no torch)
make check          # lint + tests. The pre-PR gate — and exactly what CI runs.
make test           # pytest only
make lint           # ruff check + ruff format --check
make fmt            # ruff check --fix + ruff format
make typecheck      # mypy. ADVISORY: not in `make check`, never blocks a PR.
make which          # which interpreter the targets will use

uv pip install --python .venv/bin/python -e ".[train]"   # add the training stack
```

A single test: `.venv/bin/python -m pytest tests/test_checkpoint.py -q`.

CLI (after `source .venv/bin/activate`, or as `.venv/bin/gallops`):

```bash
gallops config              # what this machine resolved — run this first
gallops show <ckpt-dir>     # what a checkpoint claims about itself
gallops train-rerank <dataset.npz> --train-run <name>   # re-ranker over a frozen model's candidates
```

Set `GALLOPS_CONFIG` to use a config file other than `./gallops.toml`.

## The invariant that shapes everything

**Tracked files say what a thing is; `gallops.toml` (gitignored) says where it
lives.** Tracked files name locations only through template variables —
`{weights}`, `{runs}`, `{data.NAME}`, `{data.NAME.tiers}`, `{site}` — expanded
at load by `config.py`.

[tests/test_portability.py](tests/test_portability.py) fails the build if a
tracked file contains a real machine root. (That test necessarily contains the
very patterns it bans and exempts itself; do not restate them elsewhere or that
file joins the failure list.) When it fails, the fix is almost always a template
variable, **not** an exemption.

This is paddock's invariant, adopted on purpose: it is the single mechanism that
kept that repo working at a second site, and a training repo has strictly more
opportunity to hardcode a path than an evaluation one does.

## Architecture

**`checkpoint.py` is the contract.** Everything else is negotiable; this is not.
A checkpoint directory carries a `checkpoint.json` recording which model, at
what step, from which training commit, trained on what. Paddock reads it for
provenance, for the contamination declaration on the leaderboard, and for the
model card. Write it **in the same function that saves the weights** — a sidecar
written later is one that gets skipped when a job is killed, and an unlabelled
checkpoint is worth less than no checkpoint, because it looks usable.

**`config.py` is the only place that knows where anything lives.** Point
`[roots.weights]` at the same directory paddock uses; that is how paddock finds
what this repo produced without either side learning the other's layout.

**Contracts are imported, never restated.** Trajectory conventions, `Command`,
`constant_velocity` and the frame definition belong to
`paddock.policy.protocol`. Import them. A client running in a foreign
interpreter can still `import paddock.*`; if it cannot, fix the environment
rather than copying a constant. Restating a convention is how the two repos
silently disagree about which way `y` points.

## Conventions

- `Trajectory` is ego-frame, **x forward, y left**, metres, evenly spaced by
  `waypoint_dt` starting at `t = waypoint_dt`. Waypoint *k* is at
  `(k+1)·waypoint_dt`; there is no waypoint for the current position.
- An **all-zero trajectory is a commanded emergency stop**, not "no plan". A
  model that has not warmed up returns `constant_velocity(obs.speed)`. Zeros
  score as a deliberate brake, which on a collision benchmark looks like a good
  result for the wrong reason.
- **Select on one split, report on another.** Never report a headline number
  from the split you selected checkpoints on. Paddock's report sorts by score,
  so a sweep is a `max()` over draws — at neuroNCAP's paired standard error of
  0.15, the best of 50 checkpoints carries roughly +0.3 of pure selection bias,
  larger than the entire measured gap between the top three models. This cannot
  be corrected after the fact.
- **Closed loop is a gate, not a training signal.** A 200-sample neuroNCAP run
  costs about 10.6 hours. Sweep on open-loop and NAVSIM; gate the two or three
  checkpoints you would actually ship on closed loop.
- **Record the scene list you read, not the tier you read it from.** Dataset
  tiers are not necessarily nested — two tiers can hold different scenes.
- Comments explain **why**, especially where the reason is expensive to
  rediscover.
- Match the surrounding file's voice over any global preference.

## Before any training run

**No run trains until the preflight passes.** `gallops.train.swerve` runs
[train/preflight.py](src/gallops/train/preflight.py) after building the model and
before its first step, and stops with every reason listed if any check fails:

- **code** — `src/` and `tests/` are committed, so the sidecar's `train_git_sha`
  names the code that ran, and the test suite passes under the interpreter that
  trains (the drive-jepa env is Python 3.9 — no `zip(strict=)`, no `match`).
- **grading** — on 48 ticks sampled across episodes and visited late tick first,
  the trainer's own `Referee` reproduces the labels *exactly*: the expert's
  verdict, the driven plan's hit time, all 32 candidate verdicts. This is the
  check that would have caught defect 6 in
  [docs/swerve-finetune.md](docs/swerve-finetune.md): v1–v4 graded every tick from
  the first tick's pose, and on round 0 the preflight rejects that referee on
  28 of 48 ticks.
- **fidelity** — the starting weights, fed the trainer's rebuilt inputs, reproduce
  the plan they drove at collection: the pick's verdict on ≥95% of ticks, and a
  median gap under 1% of four seconds' travel (measured 0.29%). Its first real
  run was refused, and rightly: an episode's first half second needs a priming
  frame the logger never saved, so those ticks are now skipped as unrebuildable.

Do not add a way to skip it. A check that can be skipped is skipped the one time
it matters. If it fails on data you believe is right, the disagreement is the
finding: find which side is wrong before training on either. Pass `--dumps` (the
candidate dumps the labels were made from); the report lands in the checkpoint
sidecar beside `labelled` and `dumps`.

**Any new code that grades, labels or builds inputs gets a check here**, in the
form "what the trainer computes equals what was recorded". Testing the parts
separately is not enough: in defect 6 every part was right on its own.

## Before opening a PR

`make check` must pass, portability gate included; check `$?` directly rather
than piping to `tail`. `make typecheck` is the deliberate exception — advisory,
absent from `make check`, blocking nowhere — for the same reason paddock keeps
it advisory: a check that reddens unrelated work is one people learn to ignore.

When reporting results, **separate what you verified from what you argued**, and
state the strength of each claim. That standard is paddock's
(`docs/reviewing.md`, `docs/ROADMAP.md`) and it is the reason its numbers are
worth anything.
