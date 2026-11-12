# Conventions

The rules that make a checkpoint usable and a number meaningful. Most were
learned expensively in paddock; the citations are there so you can check them
rather than take them on trust.

## Contracts are imported, never restated

Trajectory geometry, `Command`, `CANBUS_*` and `constant_velocity` belong to
`paddock.policy.protocol`. Import them:

```python
from paddock.policy.protocol import Command, Trajectory, constant_velocity
```

A process running in a foreign interpreter can still `import paddock.*` — that
is how every model adapter works. If it cannot, fix the environment rather than
copying a constant. **Restating a convention is how two repos silently come to
disagree about which way `y` points**, and that disagreement shows up as a
plausible-looking score.

## Trajectory geometry

- Ego frame, **x forward, y left**, metres. Not y-right. Not world frame.
- Waypoint *k* is at `(k+1) · waypoint_dt`. **There is no waypoint for the
  current position.**
- Evenly spaced by `waypoint_dt`, starting at `t = waypoint_dt`.
- neuroNCAP wants **exactly six** waypoints — it builds six timestamps and
  indexes them positionally, so a seventh is dropped and a fifth raises
  `IndexError`.
- **Six is the closed-loop slice, not the model's output width.** NAVSIM's
  converter builds eight poses at 0.5 s and *raises* rather than extrapolating,
  so a model that plans 3 s is locked out of the highest-resolution benchmark
  in the suite. Plan **eight** waypoints at `dt = 0.5 s`; closed loop takes the
  first six and drops the rest.
- An **all-zero trajectory is a commanded emergency stop**, not "no plan". A
  model that has not warmed up returns `constant_velocity(obs.speed)`. Zeros
  score as a deliberate brake, which on a collision-avoidance benchmark looks
  like a *good* result for the wrong reason.

## Splits and selection

**Select on one split, report on another. Never report a headline number from
the split you selected on.**

This is the single easiest way to produce a fake result, and it cannot be
corrected afterwards. `paddock report` emits one row per run **sorted by score**,
so evaluating a sweep and reading off the top is a `max()` over draws. At
neuroNCAP's paired standard error of 0.15, the best of 50 checkpoints sits about
2 SE above the truth by selection alone — roughly +0.3, which is larger than the
entire measured gap between the top three models in the matrix.

Record the selection split in the checkpoint sidecar (`selection_split`) so a
reader can tell which number is which.

## What to sweep on, and what to gate on

**Closed loop is a gate, not a training signal.** A 200-sample neuroNCAP run
costs about 10.6 hours, and paddock's `architecture.md` says outright that
closed-loop "cannot be the fast feedback signal" — it is orders of magnitude
more expensive and has far higher variance.

- **Sweep** on open-loop metrics and NAVSIM, which resolve fine differences at
  n = 12,146.
- **Gate** the two or three checkpoints you would actually ship on closed loop.
- Treat **ADE as a regression tripwire, not a target** — a constant-velocity
  baseline beats a genuinely better planner on it.

## Prefer paired comparisons

Paddock guarantees every model sees identical episodes and identical seeds
derived from the run index, so a comparison is paired rather than merely
averaged. A paired delta over identical samples is far more sensitive than two
means: paddock measured an alpamayo change worth +0.207 EPDMS at p = 0.030 over
just 60 scenarios, because it was paired.

For "did this checkpoint improve", this makes checkpoint-to-checkpoint
comparison much cheaper than a full-split run — as long as the sample set is
identical, which `sequences_sha256` proves.

## Expect a family of adapters, not one with knobs

`gallops`, `gallops_mono`, `gallops_int8`. Paddock treats camera count and
precision as per-adapter constants and hashes them into the run id, so each is
genuinely a different measurement. `viz/palette.py` colours by model **name**,
so a family also gets distinct colours in every comparison figure for free.

## Data hygiene

- **Record the scene list you read, not the tier you read it from.** Dataset
  tiers are a search path and are *not necessarily nested* — two tiers can hold
  different scenes. See [data.md](data.md).
- **Anything derived from calibration is episode state.** KITScenes rigs are
  recalibrated between sessions; paddock found three distinct front-camera
  rotations across 60 `val` scenes, and an adapter caching the first scene's
  calibration silently mis-warped roughly a quarter of its samples. Nothing
  failed — the error showed only as a drifting running mean. If you cache
  calibration, clear it per episode.
- **Check the frames.** Paddock's Bench2Drive notes record a one-line quality
  setting that rendered the sky black at midday and produced "a full set of
  results at one site before anyone looked at a frame".

## Reporting

**Separate what you verified from what you argued, and state the strength of
each claim.** Paddock's roadmap and review docs use an explicit ladder — proof /
strong / moderate / argued / unverified — and it is the reason its numbers are
worth anything. Use it.
Every result in the experiment registry carries one of these five words, plus
flags for what kind of number it is (`adapted`, `blind`, `smoke`, …); see
[experiments.md](experiments.md#strength). Older documents here use other words
(*reproduced*, *verified by reading*, *not supported*); the registry translates
them into this ladder.

Sample size is the recurring lesson: paddock has had conclusions overturned in
*both directions* by running more scenarios. Before claiming an improvement, ask
what effect size the benchmark you used can actually resolve.
