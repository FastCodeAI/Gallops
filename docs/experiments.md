# Experiments: how they are named, recorded and decided

Gallops is building a driving model that has to win on evidence, not on assertion. This page is
the set of rules that keeps that evidence traceable: from a number in the README, to the run that
produced it, to the checkpoint by its sha256, to the data it trained on and the code that trained
it, to the decision the team took because of it.

The mechanism is small. Every experiment has a folder `experiments/<ID>_<slug>/` holding an
`experiment.toml`. Everything a reader sees is rendered from those files by
`gallops experiments render --write`: the README's tables and picture, a one-page `CARD.md` in each
folder that the README's IDs link to, and the gate and decision logs in `experiments/README.md`. `make check` fails if the README differs from what
the registry renders, if a record is malformed, if it points at a file that does not exist, or if
a module in `src/gallops` was used by no experiment. The validation is in
[src/gallops/experiments.py](../src/gallops/experiments.py); this page explains the reasons.

## Contents

- [Two phases, three tracks](#tracks)
- [IDs and names](#ids)
- [Runs](#runs)
- [Checkpoints and tags](#checkpoints)
- [Versions](#versions)
- [Claim strength and flags](#strength)
- [Status and verdict](#status)
- [Predictions](#predictions)
- [Data](#data)
- [Where code goes](#code)
- [The regression gate](#gate)
- [Deciding a hypothesis](#deciding)
- [The decision log](#decisions)
- [Recording an experiment, step by step](#how-to)
- [What a results PR contains](#results-pr)
- [Moving an open branch onto the registry](#open-branches)

<a id="tracks"></a>

## Two phases, three tracks

| Track | ID | What it is |
|---|---|---|
| `gap-closing` | `GCnn` | Phase 1, closed. Each person took some benchmarks and tried to close released Drive-JEPA's gap there. |
| `hypothesis` | `Hnn` | Phase 2, from the 4-camera approach (16 Oct) on. One structural claim, tested on the agreed base and decided by the regression gate. |
| `enabler` | `Hnn` | Phase 2 work that a hypothesis needs before it can be decided: the gate itself, shared storage, a benchmark, teacher data. |

Hypotheses and enablers share one numbering because the team prioritised them together on
21 Oct; H01–H11 are that list, in that order.

<a id="ids"></a>

## IDs and names

- **An ID is never reused and never renumbered.** `H07` means one experiment for the life of
  the repository, in commit messages, sidecars, PR titles and conversation. New experiments take
  the next free number in the order they are registered; abandoning one marks it `dropped`, it
  does not free its number.
- **Priority is a field, not the ID.** When the team reorders its plans, `priority` changes and
  the IDs stay.
- **The folder is `<ID>_<slug>`,** the slug lower-case words joined by underscores
  (`H05_ssm_memory`). The test checks the folder against the `id` inside it.
- **Names inside a record keep the names the work was done under** (`v4`, `run3a`, `exp2b`,
  `1B`). Renaming them after the fact is how two documents come to disagree; the run ID
  (below) makes them unambiguous instead.
- **Short labels collide.** `S1` (a retrain step) and `s1` (a swerve run), four meanings of `v4`,
  two of `0.1.5`: write the run ID (`GC03.s1`) wherever a label could be read two ways.

<a id="runs"></a>

## Runs

A run is one training or evaluation that produced a result. Its ID is `<experiment ID>.<label>`:
`GC03.v4`, `GC05.run2`, `H07.exp3`. Old runs keep their labels; new runs are `r1`, `r2`, … in
the order they start.

Each run records what it started from (another run's ID, a checkpoint tag, or `released`), which
parts it trained and which it froze, the commit that trained it (`train_git_sha`), the code and
the data (below), the checkpoint it wrote, its results, and any gate verdict.

**A secondary run stays under its hypothesis when it tests the same claim:** another seed, an
ablation, a fix after a FAIL, the same change on another benchmark. Add it as the next `[[runs]]`,
add what you expect to `PREDICTIONS.md` first, and keep one `headline` for the experiment. **A run
that tests a different claim gets its own `Hnn`,** with `parents` naming where it came from. The
test: if its result would change this hypothesis's verdict, it belongs here; if it needs a verdict
of its own, it does not. A diagnostic that wrote no checkpoint and only informed a choice is a
`[[checks]]` entry, not a run.

<a id="checkpoints"></a>

## Checkpoints and tags

A checkpoint lives where [docs/checkpoints.md](checkpoints.md) says, `{weights}/<model>/<train_run>/step_<n>/`,
with `model` the model family: a paddock adapter, or a variant of one. For new runs, `train_run` is `<id>-<run>-<slug>` in lower
case, e.g. `h04-r1-subscore`, so a directory names its experiment.

**A checkpoint is identified by its weights' sha256, not its path.** A path names a directory,
which can be overwritten or copied between sites; paddock's `drivejepa_rerank` adapter recorded the
digest for the same reason. The **tag** is the first 12 hex digits (`29d5f9517c2e`); the record
holds all 64. `gallops experiments checkpoint <step dir>` prints the record's
`[runs.checkpoint]` block. It hashes the weights every time and fails if the sidecar's digest
disagrees, because the record pins the bytes on disk, not what the sidecar remembers;
`--trust-sidecar` copies the sidecar's digest instead, and says so in the block.

**A step directory with several weights files is refused until you say what the checkpoint is:**

- `--file <name>` pins the file paddock loads. The swerve trainer writes `model.pt` and
  `agent.ckpt`, and paddock loads `agent.ckpt`, the smaller one. The other files go in `members`,
  each with its own digest.
- `--ensemble` pins the files together, for an ensemble of seeds (`seed_0.pt` … `seed_4.pt`). The
  digest is sha256 over the members' `sha256  name` lines in name order, which
  `sha256sum seed_*.pt | sha256sum` reproduces from a shell.

`members` lists every weights file as `sha256sum` prints it, and `make check` refuses a
`weights_sha256` that is neither the digest of the member `file` names nor the digest over all of
them.

`location` is the template location, `{weights}/<model>/<train_run>/step_<n>/`: where the checkpoint
lives, or where its `POINTERS.md` says a copy belongs if it has not been moved into that layout yet.
Where it sits at the training site today goes in the run's `notes`.

Record `step_unit`, because `step` has meant an epoch (the re-ranker), a pass (the retrain) and
an optimizer step (Bench2Drive). When a sidecar is known to be wrong about its run, say so in
`known_wrong`; do not hand-edit the sidecar.

Every checkpoint that went through the gate must carry its sha256. A verdict about weights nobody
can identify is not evidence.

<a id="versions"></a>

## Versions

`gallops-v0.N` is **earned, never assigned.** A checkpoint becomes `gallops-v0.1` when the gate
says SHIP against the current base, and each later SHIP against the then-current version bumps
`N`. The promotion is a row in the decision log with `version` set, and the README's "Where we
are" reads it from there. Until the first SHIP there is no gallops version, and the README says so.

The queue compares with its own baselines, and today those are released Drive-JEPA only: a
`--baseline` job takes no `--env`. So a SHIP from the queue beats the released model, not the base,
and earns no version. Record it and decide on it; `version` waits until the base has baselines there.

Paddock's driver versions (`0.1.4b`, `0.1.7`) name the driver that feeds a model in AlpaSim, not
a checkpoint. A record names them in `shipped_as`.

<a id="strength"></a>

## Claim strength and flags

Every result carries a `strength` from paddock's ladder, the one
[docs/conventions.md](conventions.md) already uses: **proof** (it cannot be otherwise, e.g. a
bit-for-bit comparison), **strong**, **moderate**, **argued** (reasoned, not measured), or
**unverified**. Older documents in this repository use other words — *reproduced*, *measured*,
*verified by reading*, *not supported*; a record translates them into this ladder and keeps the
original wording in `note`.

`argued` is for a claim reasoned from other evidence. A number read from a run is not argued,
however thin the run: its flags say how thin (`smoke`, `no-record`).

Flags say what kind of number it is:

| Flag | Means |
|---|---|
| `adapted` | trained on the benchmark's own domain |
| `blind` | measured before the 12 Oct camera fix, when no image reached the model |
| `smoke` | one scene or a few episodes: a pipeline check |
| `selected-on-report` | the variant was chosen on the split it is reported on |
| `open-loop` | measured against logged driving |
| `no-record` | the evidence is not committed to any repository yet |

Exactly one result per experiment is its `headline`: the one that best states how it came out.
The README's table row shows it; the card shows every result.

<a id="status"></a>

## Status and verdict

`status` is where the work is: `planned`, `running`, `done`, `parked`, `dropped`. `verdict` is how
it came out: `positive`, `negative`, `mixed`, `inconclusive`, or `none` (not finished, or an
enabler). A finished experiment states what it `learned` in a sentence or two, limits included.

<a id="predictions"></a>

## Predictions

A hypothesis experiment writes `PREDICTIONS.md` and commits it **before anything trains**: what
will be measured, the effect expected, what counts as support and as refutation, what happens
next in each case, and which split selects versus which reports. Its commit date is the evidence
that the prediction came first. `make check` refuses a hypothesis with results and no
`predictions`; work that ran before this rule existed says so honestly, as
`predictions = "none: <why>"`, and its card marks it ⚠️ not pre-registered.

<a id="data"></a>

## Data

Each run lists what it read in `[[runs.data]]`: dataset, split, `role` (`train`, `selection`,
`held-out`, `eval`), count and unit. Record **the scene list you read** (`scene_list`), not the
tier you read it from: tiers are not nested. Data a run built — labels, manifests, caches — stays
off the repository in `{runs}` or `{data}`, listed with bytes and sha256 in the experiment's
`POINTERS.md`; the record names its builder (`built_by`) and its `manifest_sha256`.

<a id="code"></a>

## Where code goes

Code lives where its job puts it, and each run's `[runs.code]` links it to exactly the code that
produced it.

| Kind of code | Where it lives | Linted and tested |
|---|---|---|
| Shared and maintained: graders, label builders, losses, model parts, data loaders, the contract | `src/gallops/<area>/`, named by what it does, never by experiment ID, because the pieces that win become the model | yes |
| A re-runnable pipeline for one model or data family | a package `src/gallops/<model_dataset>/` plus `recipes/<name>.toml` as its entry point | yes; `tests/test_recipe.py` resolves its stages |
| One-off analysis, diagnostics, data manipulation, as they ran | `experiments/<ID>_<slug>/scripts/` (or an existing record's step folders), committed as run, machine paths replaced by placeholders | no: a record, outside lint |
| Patches to vendor code (Drive-JEPA, WA-JEPA) | diffs or an overlay against a pinned upstream commit, with a sha256 manifest, beside the recipe that applies them | the manifest is checked |
| Built data: labels, manifests, caches, dumps | off the repository in `{runs}` or `{data}`; `POINTERS.md` lists bytes and sha256 | sha256 pinned |

`[runs.code]` takes `modules` (dotted names under `src/`), a `recipe` (`recipes/x.toml#stage`,
which claims that stage's module), an `entry` script, `as_run` folders and `upstream` patches.
`make check` fails if any of them does not exist, or if a module under `src/gallops` is claimed by
no run and is not one of the few shared ones (`checkpoint`, `config`, `cli`, `recipe`,
`experiments`). [experiments/README.md](../experiments/README.md) renders the resulting code map.

Existing trainers stay where they are. Recipes, preflight, the Python 3.9 training interpreter
and the commands quoted in each account all name their current modules, and the link in
`[runs.code]` gives the traceability that moving them would.

<a id="gate"></a>

## The regression gate

Paddock's gate decides every hypothesis. Submit the checkpoint from a paddock checkout (paddock
`docs/queue.md`); one worker runs the jobs in turn, against the queue's own baselines:

```bash
python -m tools.gate_queue submit --model <adapter> --env <ADAPTER>_CHECKPOINT=<path> --note "H07.r1"
python -m tools.gate_queue submit ... --tier all --note "H07.r1"
python -m tools.gate_queue ls
```

The default **quick** tier takes about 40 minutes: run it on every checkpoint you would keep. `--tier
all` adds the closed-loop subsets: run it on the two or three you would ship. A checkpoint needs a
paddock adapter, and `--env` must name a key that adapter's `model.yaml` sets. `--model` takes that
adapter, not the record's `model`: `H07.exp3` is recorded as `drivejepa_kitscenes` and ran as `--model
drivejepa`. `submit` refuses a model with no queue baselines, so a new adapter needs a `--baseline`
job first (paddock `docs/queue.md`). Paddock `docs/gate.md` says what each subset measures and how
to read a verdict.

A finished job writes `<queue root>/jobs/<job>/verdict.json`, with the gate's manifest in `smoke/`.
Copy the verdict into the run that produced the checkpoint, verbatim. `{paddock_site}` below is a
label for paddock's site folder; gallops does not resolve it.

```toml
  [[runs.gate]]
  date = <the day it ran>
  tier = "<quick | closed | all>"
  baseline = "<what the candidate was paired against>"
  paddock_commit = "<the paddock sha the gate ran at>"
  manifest = "{paddock_site}/queue/jobs/<job>/smoke/manifest.toml"
  verdict = "<SHIP | HOLD | FAIL>"

    [[runs.gate.pairs]]           # one per benchmark, copied from the job's verdict.json
    bench = "<navsim | kitscenes | nuscenes | neuroncap | bench2drive | alpasim>"
    metric = "<the benchmark's own score>"
    baseline = <baseline value>
    candidate = <candidate value>
    t = <t>
    mde = <the MDE it printed>
    units = <scenes, pairs or routes>
    verdict = "<GAIN | NEUTRAL | REGRESSION | an undecided verdict, as printed | NOT RUN>"
```

The gate log in `experiments/README.md` has one row per gated checkpoint. The gate needs the checkpoint's sha256 in
the record, so the verdict can be traced to the weights it judged.

<a id="deciding"></a>

## Deciding a hypothesis

The gate gives a checkpoint one of three verdicts: SHIP, HOLD or FAIL (paddock `docs/gate.md` owns
the exact rule). Each has a next step, and it is written down, so a hypothesis never just goes quiet.
HOLD has four causes, and the job's output says which.

| Gate says | Means | What you do |
|---|---|---|
| **FAIL** | a pair REGRESSED. On one pair alone that is a reason to look, not proof: an unchanged model FAILs about 10% of the time on the quick tier and 16% on `--tier all` | Read the failing pair's numbers first. If the regression stands, do not build on this checkpoint. Add a decision row, then either run a fix as the next run, or close the hypothesis: `status = "done"`, `verdict = "negative"`. |
| **HOLD**, `not finished: <subset>` | a subset did not finish | No verdict yet. `python -m tools.gate_queue retry <job>`; finished runs are kept. |
| **HOLD**, a pair undecided | a pair did not reach GAIN, NEUTRAL or REGRESSION (a clean crash check's TOO FEW UNITS is normal) | No verdict yet. Fix that pair, then run it again. Paddock `docs/gate.md` says what each verdict means. |
| **HOLD**, on the quick tier | every pair decided, and this is the best quick can do: its closed-loop pairs may not ship | The expected good result. Run `--tier all` if you would ship this checkpoint. |
| **HOLD**, on the closed tier | every pair decided, nothing regressed, and no closed-loop pair that may ship gained | Read each NEUTRAL with its MDE: it may only mean the subset is too small. Add a decision row: run a stronger version as the next run, or close it as `inconclusive`. |
| **SHIP** | every pair decided, none regressed, and a closed-loop pair that may ship gained | Add a decision row, with `baseline` if the base changes. `status = "done"`, `verdict = "positive"`. Set `version` only for a SHIP against the current base: see [Versions](#versions). |

`make check` refuses a `done` hypothesis with no gate verdict on any run, or with no decision citing
it dated on or after its last gate. It also refuses `verdict = "none"`, and `positive` without a SHIP.
Work that never reached the gate ends as `parked` or `dropped`, with what it `learned`.

**A change that trains nothing still goes through the gate.** A paddock-side input change on the
same weights is gated like any other, and paddock accepts identical weights for a code change. Set
`trained = []` on the run and pin the weights it ran on; only such a run may share a digest.

<a id="decisions"></a>

## The decision log

`experiments/decisions.toml` is the linear story: each decision the team took, in date order,
numbered `D01`, `D02`, … and never renumbered. A decision names who took it (`by`), the runs or
experiments that were its `evidence`, what it led to (`next`), and its `source`. Set `baseline`
when it changes the model every hypothesis is measured against, and `version` when it promotes a
checkpoint to `gallops-v0.N`. `experiments/README.md` lists the log in full.

<a id="how-to"></a>

## Recording an experiment, step by step

1. **Register it before it runs.** `gallops experiments new H15 my_slug --owner Naren` copies
   `experiments/_template/`; fill in `title`, `short`, `question`, `hypothesis`, `benchmarks`,
   `parents` and `priority`.
2. **Write `PREDICTIONS.md` and commit it** before the first training step.
3. **Put maintained code in `src/`, one-off scripts in the folder's `scripts/`,** and a recipe
   if the run should be re-runnable.
4. **When a run starts,** add its `[[runs]]` block: `start_from`, `trained`/`frozen`, `data`,
   `code`, `train_git_sha`.
5. **When it writes a checkpoint,** paste `gallops experiments checkpoint <step dir>` into the run.
6. **When results come in,** add `[[runs.results]]` with their strength, flags and source; mark
   one `headline`.
7. **When the gate runs,** copy its verdict into `[[runs.gate]]`.
8. **When the team decides,** add the decision to `experiments/decisions.toml`, and set `status`,
   `verdict` and `learned`.
9. `gallops experiments render --write` (the README and the cards; never edit `CARD.md` by hand), then
   `make check`, then open the PR.

<a id="results-pr"></a>

## What a results PR contains

One PR per experiment, or one per run if it is large. Reviewers check each of these:

- [ ] `experiment.toml` is complete: owner, question, hypothesis, data with split and counts, code.
- [ ] `PREDICTIONS.md` was committed before the run, or `predictions = "none: <why>"` says why not.
- [ ] Every checkpoint is pinned by sha256; off-repo files are listed in `POINTERS.md` with their sha256.
- [ ] Every result has a `source`, a `strength` and its flags, and one is the `headline`.
- [ ] The gate verdict is copied in, or the PR says why the gate has not run yet.
- [ ] `README.md` says why, what was done, what it showed and what was decided.
- [ ] A decision row is in `experiments/decisions.toml` if this result changes what the team does next.
- [ ] `gallops experiments render --write` was run, and `make check` passes.
- [ ] The title and description come from `gallops experiments pr`, with every `TODO(author):` replaced.

A results PR is any PR that adds, changes or removes a result or a gate verdict. Its title and the top of its
description are rendered from the record, like the README, so the ID and the numbers are copied, never
typed:

```bash
gallops experiments pr H04.run3a > /tmp/pr.md   # H04 alone takes the run that holds the headline
$EDITOR /tmp/pr.md                              # replace each TODO(author):
gh pr create --title "$(head -1 /tmp/pr.md)" --body-file <(tail -n +3 /tmp/pr.md)
```

The title is `<run>: <benchmark> <metric> <baseline> → <value>, gate <verdict>`. The description gives
the hypothesis, the result with its strength and flags, the gate, the decision and the card. Anything
longer belongs in the experiment's `README.md`.

The `results-pr` check in CI re-renders both, from the PR merged into its base. It fails a PR when:

- it changes a result or a gate verdict and has no rendered block;
- its title or block differs from the record, which is fixed by re-running the command;
- a `TODO(author):` is left in it;
- it also changes the results of a second experiment;
- the run it reports is not one it changed;
- it has more than 30 lines written by hand.

A PR that changes a result without reporting one, such as a correction across several records, says so
on a line of its own: `Not a results PR: <why>`. The reviewer judges that reason, as they judge
`predictions = "none: <why>"`.

Merge with a merge commit, never a squash, when another PR is stacked on this one. A squash lands
the PR as one new commit, so whatever it gained after the upper PR was cut conflicts there, even in
files the upper PR never touched. If it must be squashed, merge its head into the upper PR first.

<a id="open-branches"></a>

## Moving an open branch onto the registry

Several branches were opened before the registry existed; each adds a row to the old README table,
and they conflict with each other there. The registry already holds an `experiment.toml` for each,
with `branch` set so the card links to the record on that branch. If a branch is deleted without
merging, set `branch` to the full commit its closed PR keeps (`refs/pull/<n>/head`), so the links
still work. When one merges:

1. Merge `main` into the branch; do not rebase, because other PRs may sit on it. Take main's README
   when resolving the conflict (`git checkout --theirs README.md && git add README.md`): your row is
   generated now.
2. Move the branch's record into its ID folder (`git mv`), and its account to that folder's
   `README.md`. If the folder had a README of its own, it becomes `RECORD.md`. Keep the renames in
   their own commit, so history follows them.
3. In `experiment.toml`, clear `branch` and point `narrative`, `code` and `scene_list` paths at
   their new places; `make check` now verifies that they exist. Update paths in the branch's tests
   and links.
4. Maintained code stays in `src/`. Every new module under `src/gallops` must be named in some run's
   `[runs.code]`, or `make check` fails.
5. `gallops experiments render --write` and `make check`, then check the PR against
   [What a results PR contains](#results-pr).
