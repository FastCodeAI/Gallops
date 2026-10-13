# Reviewing a pull request

How to review a gallops PR so that the checkpoint it produces, and every number
quoted about that checkpoint, can be trusted by someone who was not in the room.

The method and the severity scale are paddock's (its `docs/reviewing.md`), and
they are kept the same here on purpose: the same markers and the same merge
rules, so a 🟠 means one thing in both repos. What differs is *what goes wrong*.
Paddock's failures live in adapters and scorers. Gallops' failures live in what
a trainer was fed, what it computed from that, and what its checkpoint then
claims. Every section below comes from something that went wrong in this repo,
and most of them passed every check that existed at the time.

The guiding rule is the same: **separate what you verified from what you
argued.** "Rebuilt the weights bit-for-bit from the sidecar" is verified. "The
hashes look right" is not. Mark the strength of each claim — *reproduced*,
*measured*, *verified by reading*, *argued* — and mark every finding 🔴 / 🟠 / 🟡
(§12).

`/pr-review` runs this procedure. `/pr-fix` answers the review it writes.

---

## 1. Run it, in its own environment

```bash
git fetch origin pull/<N>/head:pr-<N>
git worktree add /tmp/pr-<N> pr-<N>
cd /tmp/pr-<N>
uv venv -q --python 3.11 .venv && uv pip install -q --python .venv/bin/python -e ".[dev]"
env -u VIRTUAL_ENV -u CONDA_PREFIX -u GALLOPS_CONFIG make check; echo "exit=$?"
```

- **The worktree needs its own venv.** The main checkout's `.venv` installs
  gallops editable from the *main checkout's* `src/`, and `tests/` is a
  package, so pytest does not put the worktree's `src/` first. With that venv
  active, the PR's tests run against whatever branch the main checkout has out —
  and can pass.
- **Check `$?` directly.** `make check | tail` reports *tail's* exit code.
- **A green run is the portability check.** `tests/test_portability.py` is part
  of `make check`, and it scans untracked files too.
- **There may be no CI.** On 2026-10-13 there was none, although the Makefile
  says CI runs `make check`. If `.github/workflows/` is still absent, your run is
  the only one the PR has had. Say so.

**A green `make check` says nothing about torch code.** Its environment has no
torch, so nothing constructs a model, a `TickDataset` or a `Referee`. Three times
in one session a method or a test vanished in an edit while `make check` stayed
green. One run died eight minutes in, after the model had loaded. For a PR that
touches torch code:

- The structural guard has to cover every class the PR touches. That guard is a
  torch-free test that checks each class defines every method the training loop
  calls on it.
- The evidence that the code runs is a smoke run of a few steps under the
  training interpreter. A shadowed argument (`trainable` as both the argument
  and a counter) surfaced only there.
- Say which of the two you ran.

Code that runs **inside a model's own interpreter** (Drive-JEPA's is Python 3.9)
has a trap of its own: ruff's pyupgrade rewrites it to 3.11-only forms, and
`make check` runs 3.11. Check those files against the interpreter they run in.

## 2. Contracts that must be imported, not restated

| Contract | Lives in | What restating it looks like |
|---|---|---|
| checkpoint layout | `SiteConfig.checkpoint_dir` | `weights / model / run / f"step_{n:09d}"` |
| repo root | `config.REPO_ROOT` | `Path(__file__).resolve().parents[2]` |
| the sidecar | `gallops.checkpoint` | a hand-built `checkpoint.json` |
| trajectory, `Command`, `constant_velocity`, the ego frame | `paddock.policy.protocol` | a local constant, a sign flip |

A copy is correct today and silently wrong the day the definition moves. Its
tests usually restate the same thing, so they keep passing after the move.

Some things genuinely cannot be imported across the two repos: a weights-file
format string, a forward pass, a feature list. A copy of those is acceptable
**only with a check that runs on every load**. The re-ranker's probe is the
pattern: the weights carry inputs and the scores the trainer gave them, and
paddock replays them and refuses the file on any mismatch.

**A transcription shares its source's bugs.** The swerve trainer's input builder
was transcribed from paddock's Drive-JEPA adapter. Both composed nuScenes'
`LIDAR_TOP` into each camera's extrinsics, so every collection and every
checkpoint was camera-blind, and nothing on either side noticed (§4). When a PR
copies code from paddock or from a vendor, review the copy as new code. It is
not pre-approved by its origin.

## 3. A checkpoint is whole and true, or it does not exist

This is paddock's "a crash must not become a score", seen from the trainer's
side. Paddock reads `{weights}` while gallops writes to it, so:

- **The weights and the sidecar are written together.** Build and validate the
  `Checkpoint` before writing anything. Stage the step directory, then rename it
  into place. A failure or a kill must leave a whole checkpoint or none — never
  weights with no sidecar, and never new weights beside an old sidecar.
  `checkpoint.staged()` does this, and refuses a step directory that already
  exists. A trainer that writes a step directory any other way is a finding.
- **One fit is one run.** A `--train-run` that already exists is refused. If it
  is not, two unrelated fits sit side by side, and paddock groups them as one
  sweep.
- **Every claim in the sidecar is checked against the weights, never against
  the note.** A run recorded as "refiner only" had trained its scorer: its
  scorer weights differed from the frozen ones by up to 0.032. That note had
  been carried over stale. Check a freezing claim by diffing the parameters.
  Check `precision` against the dtypes actually saved.
- **Record the scene list you read, not the tier** (conventions.md).
  - `TrainedOn.fraction=None` means *all of it*. A run that saw 373 of a split's
    533 scenes has to say so.
  - Exclusions go in the sidecar too: an `--exclude-scene` that is not recorded
    there is invisible later.
  - So does anything about the inputs that changes what the weights mean, such
    as an image geometry or a camera variant.
- **Never hand-edit a sidecar** to make it agree with the weights or the text.
  Regenerate it.

## 4. A check is only as independent as its reference

This is the lesson that cost the most here, so it gets its own section.

- **Every check that existed passed, and the training targets were still
  wrong.** The grader reproduced paddock's 38,080 verdicts exactly, and
  re-grading reproduced the labeller's labels exactly. Each check tested a part
  that was right. Nothing compared *what the trainer computes* with *what the
  labeller recorded*. A per-episode cache was grading every tick from the
  wrong pose, wrong on up to 36.6% of verdicts (swerve-finetune.md, defect 6).
- **The fidelity check passed on a camera-blind model,** because it compared the
  trainer against a recording made by code with the same bug (§2).
- **The safety check graded against the logged human path while the labels used
  the lane centre.** Only a training scene exposed that. Its guard now refuses
  to report below 99.5% agreement with the labels.

So, for every check a PR cites as evidence, ask what it compares against, and
where that reference came from. If the reference came from the code under test,
from a copy of it, or from a recording that code made, the check is *argued*,
not verified. Independent references are paddock's reader at the ref that will
merge, the vendor's own forward pass, the labels as recorded (not recomputed),
and a physical limit.

## 5. The model must have had what it is trained on

- **Rebuilt inputs must reproduce what the model saw.** A PR that changes how
  inputs are built cites the rebuild-against-recording check (the preflight's
  fidelity test), run after the change. That check has caught three bugs so far:
  - lidar-to-ego assumed to be the identity: up to **9.16 m** of disagreement;
  - an `img_shape` off by 416 px, which moves every projection;
  - missing priming frames: about 7% of ticks would have trained on a past the
    model never had.
- **The output must depend on the input.** If you blank the camera, the plan must
  move; the preflight's `sight` test checks this. A PR that touches cameras,
  frames or extrinsics shows it.
- **No target may ask for a reaction to something the model cannot perceive.**
  400 of 9,495 ticks taught a swerve away from a hazard outside the camera's
  field of view. That maps "empty road" to "swerve", which is worse than no
  training.
- **Every cache needs a key that covers everything its value depends on.** The
  wrong-pose defect in §4 was a per-episode cache of a per-tick value.

## 6. Targets and baselines

- **Targets must be physically possible by the module's own limits.** Three
  targets once were not:
  - a recovery that jumped 3 m back to the path in half a second: 24.5 m/s²,
    six times the comfort limit;
  - a swerve drawn at a 1.7 m turn radius;
  - a "brake" that was really a coast.

  Check new targets against the comfort and curvature limits. An unreachable
  target must be refused, not drawn.
- **Ask what "good" is measured from.**
  - Reach compared against zero produced one wrong diagnosis, because a correct
    plan on a curving road sweeps 1.93 m.
  - Progress scored against free-running speed taught the scorer to prefer
    swerving over braking, the opposite of the expert.

  Compare against the expert's own geometry.
- **Measure distance to a polyline, not to its nearest vertex.** With 5 m
  between keyframes, a trajectory exactly on the path read 2.00 m off it.

## 7. Averages hide mechanisms

- **Ask for the case, not only the mean.** Three of the six swerve defects were
  found in a picture, and four more by plotting one collided episode tick by
  tick. For a PR that claims a change in behaviour, the evidence includes one
  case: the camera view beside the top-down view of the same tick, or the two
  runs' videos side by side.
- **A correctness fix can make the aggregate worse.** Fixing those four defects
  took rescues from 469 to 437, because some "rescues" had been trajectories no
  car could follow. Before calling it a regression, check what the aggregate was
  counting.
- **New data that does not change behaviour means something downstream overrides
  it.** 1,200 targets changed from swerve to brake and the behaviour did not
  change at all, because the scorer, which picks what gets driven, wanted the
  opposite.

## 8. Every number traces to a run

- **Select on one split, report on another** (conventions.md). A trainer that
  prints its selection metric labelled "held-out":
  - is a 🟡 when the bias has been measured and is small;
  - is a 🟠 when that number is quoted as the result.
- **Every row of a results table traces to a `run.json` whose checkpoint digest
  matches.** That is how two rows turned out wrong: one run had never been made
  (no override run matched its checkpoint), and one mean was 2.50, not 2.56.
- **Know what the benchmark can resolve** before calling a difference a result:
  neuroNCAP's paired standard error is 0.15 (conventions.md). Closed loop is a
  gate on the two or three checkpoints you would ship, not a signal to sweep on.

## 9. Docs and the PR description agree with the diff

- **Check every prose claim against the diff.** Prose is what the next person
  reads first.
- **STRATEGY.md and data.md repeat each other.** A change that falsifies one
  usually falsifies both. Corrections go at the top of STRATEGY.md, where a
  reader starting there meets them.
- **Check where paddock's half of the change has actually merged.** A gallops
  doc that says "paddock loads it" while paddock's half sits on an open paddock
  PR is false at the other site until that PR merges.
- **A link to a file that exists only on a PR higher in the stack** breaks on
  `main` until that PR lands.

## 10. Stacked PRs

Gallops PRs stack: a PR's base is often another PR's branch, and a PR based on
`main` can still carry an earlier open PR's commits.

- **Review what this PR adds,** against its base (`git diff origin/<base>...`).
  Say which commits belong to other PRs, and leave those to their reviews.
- **A fix on one PR does not reach the PRs above it,** and it conflicts with them
  wherever they touch the same files. Say which ones.
- **End with a merge note when it matters.** Use a merge commit, never a squash
  or a rebase: those rewrite the commits the PRs above are built on. With
  head-branch auto-delete off, the PR above is not retargeted to `main` until
  its base branch is deleted.

## 11. Shared-machine hygiene

The box is shared, and its one GPU is saturated (STRATEGY.md).

- **A review never uses the GPU.**
- **Before anything heavier than `make check`,** check `uptime`, cap BLAS
  threads (`OMP_NUM_THREADS=4`) and `nice` it. Ask before anything longer than a
  few minutes.
- **Never write into the real `{weights}`.** Paddock reads it. Every
  verification run passes a temporary weights root.
- **Kill only the process group you started.** Never `pkill` by name.

## 12. Writing the review

The scale is paddock's, unchanged. If the two ever diverge, raise it; do not
quietly follow one.

| Marker | Meaning | Merge rule |
|---|---|---|
| 🔴 **Blocker** | wrong numbers, a model trained on inputs it never had, harm to other users of the box, or the feature cannot run as documented | fixed before merge |
| 🟠 **Should fix** | a correctness or contract gap that will bite the next run or the next site: a restated contract, a sidecar that is wrong about its weights, a check that is not independent, docs that do not match the diff | fixed in this PR, or a tracked follow-up before merge |
| 🟡 **Nit** | quality, efficiency, wording | the author's call; never blocks |
| ✅ **Verified** | you ran it, or read the code it depends on | label the evidence, not the finding |

Rules of thumb for the scale, for a trainer:

- **If paddock would show a number that is not what was measured, it is 🔴,**
  however small the fix. That includes weights sitting under another run's
  sidecar, a model that was camera-blind, and a contamination declaration that
  claims *less* than the model saw.
- **If it works only on the author's machine, it is 🔴.** That covers a path in
  a tracked file and a module on the author's `PYTHONPATH`.
- **If the weights are right but the record is wrong, it is 🟠.** That covers a
  stale note, a wrong `precision`, and a `trained_on` that claims *more* than the
  model saw.
- **Anything that could be a follow-up PR without changing a number is 🟡.**

The structure:

- **Open with the verdict and a legend line.** Then say what is *right* about the
  PR, specifically enough that the author knows what to keep doing.
- **Next, a ✅ table of what you actually verified, with the result.** "Nothing
  beyond the diff" is an honest row.
- **Then the findings, grouped 🔴 → 🟠 → 🟡 and numbered.** Anchor each to
  `file:line` in the head commit you reviewed. Give each a concrete failure
  scenario and a concrete fix. A finding with no failure scenario is a
  preference, and goes in 🟡.
- **Mirror each marker on the matching inline comment,** so the file view and
  the summary agree.
- **Write as the most senior person in the room would:** no hedging on what you
  verified, no certainty on what you only argued.

## Reviewer's quick list

- [ ] own worktree and venv; `make check` run, exit code checked; CI presence noted
- [ ] torch code: structural guard covers it, or a smoke run, stated which
- [ ] layout, repo root, sidecar and protocol imported; cross-repo copies checked on every load
- [ ] code copied from paddock or a vendor reviewed as new code
- [ ] checkpoint written whole or not at all; an existing run refused
- [ ] sidecar checked against the weights: frozen parts, precision, scenes, exclusions, input variant
- [ ] every cited check has a reference independent of the code under test
- [ ] rebuilt inputs reproduce the recording; blanking the input moves the output
- [ ] targets physically possible; baselines measured from the expert's geometry
- [ ] at least one case looked at, not only the mean
- [ ] every number traces to a run with a matching checkpoint digest; selection ≠ report split
- [ ] docs, STRATEGY.md, data.md and the PR description agree with the diff and with paddock's merged state
- [ ] stack: reviewed against the base; other PRs' commits named; merge note
- [ ] no GPU used, `{weights}` untouched, processes cleaned up
- [ ] review states verified vs argued; every finding carries 🔴 / 🟠 / 🟡
