# Teaching Drive-JEPA to evade

The record of the first end-to-end training experiment in this repo: what was
built, what each run measured, and the six defects the measurements exposed.
Numbers here are measurements unless the text says they are an argument — the
standard [conventions.md](conventions.md) holds both repos to.

> **Correction (2026-10-12): every run on this page was camera-blind.** The trainer's
> inputs (`swerve_data.py`) and the paddock adapter that drove every collection both
> placed the cameras relative to nuScenes' LIDAR_TOP, yawed −90°. Drive-JEPA reads the
> image at points in ego coordinates, so none landed in the front image: a black
> frame gave the identical plan. Stage A, joint v1–v4, v4fix, l4fix, l5 and s1–s3
> were trained, and their episodes collected, without the image. The numbers below
> are what those runs measured; what they say about teaching Drive-JEPA to *see* a
> hazard and evade does not stand. Fixed here (`camera_extrinsics`) and in paddock
> #62, and `preflight.py` now refuses a model a blanked camera does not move.
>
> **And batched (2026-10-14).** Every run here trained at `--batch 8`. The vendor's
> spatial cross-attention picks each camera's queries from batch element 0's mask
> for the whole batch, so with the image reaching the model, every tick after the
> first would read the image along its neighbour's plans. That did not matter while
> the model was blind; it would from the first seeing run. Every forward pass is now
> one tick (`TICKS_PER_PASS` in `swerve_data.py`), and `--batch` refuses anything else.

**Status (2026-10-08): not finished; the capability is still absent.** Eleven
runs. The only one not worse than the released model is s1, the scorer retrained
alone: 3.671 against 3.568 over the full neuroNCAP set, +0.10 with a pair-level
standard error of 0.08, so not a proven win. Every run that retrains the plan
generator drives faster and loses the braking that earns the released model its
5.0 on stationary and side. [After the fix](#after-the-fix-l4fix-to-s3) has the
runs since the pose bug and what they taught.

**v1 to v4 trained their scorers on targets graded from the wrong place.** Found
on 2026-10-06: the trainer graded each tick as if the car were where the
episode's first-looked-up tick had it, 14–16 m off, which flipped a fifth of the
stationary verdicts and a third of the frontal ones. Defect 6 has the detail; the
trainer now refuses to start until a preflight shows its grading reproduces the
labels.

---

## The question

Paddock's neuroNCAP sweep put Drive-JEPA's frontal split at 2.42 against 4.23 on
stationary: it drives into oncoming traffic. An earlier probe found why — across
all 32 of its trajectory proposals, the widest reaches **0.70 m sideways on a
frontal tick**, which is not an evasion of anything. The model cannot propose a
swerve, so no amount of better selection will produce one.

Navtrain cannot teach it either: 20 hours of driving contains **1.5 seconds** of
emergency braking and no evasions. The data does not exist, so it has to be made.

## What was built

Five nuScenes train scenes reconstructed with NeuRAD (7.5 h of GPU each,
measured), actors inserted on collision courses, and the released model driven
through them while an expert labels every tick offline.

```
reconstruct  ->  author scenarios  ->  collect (the model drives)  ->  label  ->  train  ->  measure
```

The expert is in [expert/draw.py](../src/gallops/expert/draw.py). It **solves**
the evasion rather than picking from a lattice: the offset comes from geometry,
the timing from a lateral-acceleration and curvature budget. Two rules keep a
drawn target honest.

- **Draw from a prediction, verify against the truth.** The offset is sized
  against a constant-velocity extrapolation of the actor — roughly what two
  frames would show — then graded against the actor's real future. The target is
  always something the model had the information to produce, and a target that
  would crash is never taught.
- **Everything goes through the same grader as the model's own 32.** It
  reproduces paddock's `check.json` exactly: 38,080 of 38,080 candidate verdicts.

Three data sets, each 150 episodes over the same 15 situations (5 scenes × 3
hazard types × 10 seeds):

| data set | who drove | ticks | car out of the camera's view |
|---|---|---|---|
| round 0 | the released model | 9,495 | 5% |
| GT | the human's logged path; the model's plans saved, not driven | 5,426 | 25% |
| DAgger round 1 | s2 | 9,263 | 40% |

## What each run measured

neuroNCAP scene 0103, two runs per scenario. Every one is recorded
`valid: false` — three of twenty pairs under a checkpoint override is not a
neuroNCAP score, and `paddock report` excludes it.

Seeds 0 / 1. Every override run was matched to its checkpoint through the path
digest in its `run.json`.

| | frontal | stationary | side | mean |
|---|---|---|---|---|
| released (n=10) | 2.05–2.26, hits 10/10 | 5.0 ×10 | 5.0 ×10 | 4.05 |
| Stage A — refiner only | 1.90 / 1.90 | 1.30 / 0.89 | 0.81 / 0.56 | 1.22 |
| v1 — refiner and scorer | **5.00 / 5.00** | 0.00 / 0.00 | 0.00 / 5.00 | 2.50 |
| v2 — brake-first expert | 1.60 / 1.60 | 5.00 / 5.00 | 5.00 / 0.07 | 3.05 |
| v3 — progress capped at the expert's | not run | | | |
| v4 — progress peaks at the expert's | 2.15 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 | **4.53** |
| v4fix — v4, pose fixed | 2.07 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 | 4.51 |
| l4fix — + brake-to-stop expert | 1.41 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 | 4.40 |
| l5 — + lane-centre, drivable expert | 3.05 / 3.05 | 2.51 / 2.44 | 2.40 / 2.45 | 2.65 |
| s1 — scorer only, ranking loss, + GT | 2.02 / 2.23 | 5.00 / 5.00 | 5.00 / 5.00 | 4.04 |
| s2 — refiner only (swerves ×4), then scorer | 2.19 / 2.19 | 5.00 / 1.15 | 1.07 / 1.04 | 2.11 |
| s3 — DAgger round 1, from s2 | 0.56 / 0.80 | 1.04 / 0.83 | 0.73 / 0.86 | **0.80** |

Two corrections to earlier versions of this table. **v1 trained its scorer**: its
scorer weights differ from Stage A's (frozen) by up to 0.032, so the "refiner
only" in its sidecar was the stale note, and its mean is 2.50, not 2.56. **v3 was
never run on 0103**: no override run's digest matches its checkpoint. Measured
offline, its capped target tied braking with full speed and the model sped up 26%
where the expert braked 38%; v4 replaced it the same day.

**v4 is the best result and it is not the capability we wanted.** It drives like
the released model with harder braking: 0.11–0.21 m of lateral motion, slowing
from 7.7 to about 2 m/s. On held-out swerve ticks it moves 0.07 m where the
expert asks for 4.59. The frontal gain is lower impact speed, plus one seed of
two avoided entirely — not an evasion.

Two provenance notes on the v4 row. **The v4 checkpoint trained on
`labelled_round0_v2`**, not on the label set later named `v4`: training ended
2026-10-02 13:35 IST and the v3 and v4 labels were written at 23:04 that day and
on 2026-10-06. Its sidecar recorded neither, and said "stage A — refiner only"
for a joint run; it now records both, correctly. And **all four scorers carry
defect 6.**

## The six defects, and what found each

The experiments were worth more for these than for the scores. Each was found by
a different instrument, and three were invisible to every summary statistic.

**1. Teaching reactions to an off-camera hazard.** *Found by plotting the
training set.* Drive-JEPA reads one camera, half-FOV 32.6°, and the crossing
actor sits at a median 43.5° off axis. 400 of 9,495 ticks taught an evasion of
something outside the frame — a mapping from "empty road" to "swerve left", which
is worse than no training because it fires on ordinary driving. Every tick now
records the hazard's bearing and whether it is in frame; reactive targets that
cannot see their hazard are dropped, `continue` is kept at any bearing.

**2. A target that began at t = 0 and never came back.** *Found by watching the
closed-loop video.* The drawn swerve started its lateral move in the first
instant with the shortest ramp the tyres allow, and held the offset to the
horizon — "flat after". At 10 Hz that re-commands "be fully displaced, as hard as
possible, now" on every tick, and the model threw the wheel over at first sight
and sat at an angle. The plan now holds its lane, then moves, then recovers.

**3. An undrivable recovery.** *Found by probing a state round 0 never visits.*
`follow()` returns points **on** the logged path, so a plan for a car three
metres off it began with a three-metre jump half a second later: **24.5 m/s²**,
six times this module's own comfort limit. That is the answer to "how do I get
back in my lane", it is physically impossible, and it was taught at every tick
after an evasion. Round 0 hid it completely, because the released model never
leaves its lane. `rejoin()` eases the correction instead: 1.7 m/s², back in lane
by 4 s.

**4. Swerving where braking was available.** *Found by measurement.* The ladder
only reached braking when no swerve was clear, so it never braked for a parked
car. Braking alone was clear on **197 of 197** sampled stationary swerve ticks
and 10 of 10 side ones, every one taught as a swerve — while on 0103 the released
model scores 5.0 ten times out of ten by simply stopping. The split is geometric,
not by scenario name: braking removes the ego's own contribution to the closing
speed, so it resolves a parked or crossing car and not an oncoming one. Measured
closing speeds do not overlap near the −3.0 m/s threshold (oncoming −8.7 median,
crossing 0.1, parked 0.0).

**5. The scorer was trained to want the opposite of the expert.** *Found by
measuring behaviour after the data was already right.* `score_target` measured
progress against free-running speed, so a clear swerve scored 1.00 and a clear
brake 0.75 — every tick, all 32 proposals. The scorer picks what gets driven, so
changing the expert from swerve-first to brake-first moved 1,200 targets and
changed the behaviour **not at all**. Capping progress at the expert's distance
was not enough either: brake and full-speed both reached the cap and tied. The
term now peaks at the expert's distance and falls away on both sides, and slowing
on brake ticks went from −26% (accelerating) to +47%, against an expert asking
38% and a released model doing 50%.

The proposals were never the problem, which is worth recording because it was the
first guess: a third of the 32 travel under 75% of free-running and the shortest
is a near-dead stop. The brakes were always on offer; the scorer was declining
them.

**6. Every target graded from the wrong place.** *Found by a check that could not
be true.* To train the scorer, the trainer grades all 32 proposals on every tick —
does this plan hit the car, leave the road, leave the lane — and those verdicts are
its soft targets, and the anchor term's filter. Grading needs the car's path, the
map, and **where the ego is at that tick**. `episode_context` returned all three,
and `Referee` cached the result once per episode, so every tick was graded with
the ego wherever the first-looked-up tick had it. The cache came from `reach.py`
on 2026-09-30 and moved into the trainer with the scorer a day later, so it is in
every run that trained the scorer: v1 to v4.

Re-grading round 0's dumped proposals both ways measured what it cost:

| | mean pose error | hit/clear verdict wrong |
|---|---|---|
| stationary | 15.6 m | **19.5%** of 3,776 candidates |
| frontal | 13.8 m | **36.6%** of 3,008 |
| side | 15.6 m | 1.6% of 3,808 |

The lane term was measured from the wrong place on nearly every tick. Progress is
the one term that needs no pose, and it was the one part of the target that was
right — which fits what v4 learned: to travel as far as the expert, and nothing
about the car. On the 46 held-out stationary ticks where the released model is
creeping into the parked car, v4 picks a plan that crashes on all 46, travels
further than the released model (7.3 m against 5.8; the expert asks 2.5), and
ranks the best clear stop 14th of 32 where the released model ranks it 11th — a
clear stop is among the 32 on 96% of those ticks.

Every check that existed passed, and that is the lesson. The grader reproduces
paddock's 38,080 verdicts exactly; `label.py` grades at the right pose, and
re-grading reproduces its labels exactly. Each tested a part that was right. What
nothing did was compare **what the trainer computes** with **what the labeller
recorded**, and a creep check that copied the same cache is what finally exposed
it: it reported the released model's pick clear on ticks where it had crashed.

`episode_context` now caches only what an episode shares and returns the tick's own
pose, and [train/preflight.py](../src/gallops/train/preflight.py) runs before every
training step: the tree is committed and its tests pass under the training
interpreter; the trainer's own grading reproduces the labels exactly — expert
verdict, driven hit time, all 32 candidate verdicts — on ticks sampled late-first
across episodes; and the starting weights, fed the trainer's rebuilt inputs,
reproduce what they drove. The test suite holds a referee with the old cache and
asserts the preflight rejects it.

Its first real run refused to train. The refusal came from how the limit was
set, an absolute 5 cm from one scene's late ticks, where the gap scales with
speed: the refused smoke and the one that passed compared the same 12 ticks at
the same 0.1395 m median, and the second passed at 0.89% once the limit became a
share of travel. What the refusal prompted is what found a real defect. Measured
over 506 round-0 ticks, the rebuilt inputs drift from what the model saw by an
amount that depends on where in the episode a tick sits: the model's one look
into the past is the front frame 0.5 s back, and for an episode's first half
second that was one of the engine's priming frames, which the logger never
saves. Nearest-stamp selection silently handed over the episode's first frame
instead, and the rebuilt pick landed a median 0.4–0.5 m (p90 2.4 m) from the one
driven, against 0.10 m once the real frame exists. The gate as it then stood could
not have seen it: it was a median, and its sample started a tenth of the way into
each episode (1 of l4fix's 48 sampled ticks sat in positions 0–4).

About 7% of every round's ticks were trained on a past the model never had, and
the loss falls mostly on brake targets: on l4fix's train ticks, 269 of 1,423
brake (18.9%), 115 of 932 swerve (12.3%) and 74 of 4,119 continue (1.8%); on
held-out 0053, 147 of 560 brake (26%). They are now dropped before anything
counts them, recorded by kind in `notes.unrebuildable`, and a set with more than
10% of them is refused. Until then the dataset skipped them silently, so l4fix's
sidecar claims 1,423 brake ticks where about 1,154 trained, and validation losses
before and after 38690b9 are over different held-out sets. Saving the priming
frames at collection would get them back. Past that, the remaining gap is a
median 0.29% of four seconds' travel, most likely the JPEG round trip. The
fidelity check now gates the median against that and the p90 at 10%, bounds the
ticks it cannot rebuild, and samples each episode's earliest tick.

## After the fix: l4fix to s3

Each row changes one thing from the one above, for the reason given. "Held out"
is [train/safety_check.py](../src/gallops/train/safety_check.py) on scene 0053,
never trained on: the share of approach ticks (stationary, expert brakes) where
the model's pick is safe. The released model scores 89%.

| run | what changed, and why | on 0103 | held out |
|---|---|---|---|
| v4fix | v4's recipe on correctly graded targets | same as v4: slows to ~2 m/s, no swerve | 88% |
| l4fix | expert holds speed, then brakes firmly to a stop; ticks with no recorded past skipped | like v4, ends 2–3.4 m/s | 55% |
| l5 | expert follows the lane centre (it copied the human's path) and every plan is drivable | never below ~3 m/s; hits all 6 | 32% |
| s1 | scorer alone, refiner frozen, + ranking loss + 150 GT drives: l5 had safe candidates and picked wrong, and joint training damaged the candidates | same as released: slows to ~1.7 m/s | 8% |
| s2 | refiner alone (scorer frozen, swerve ticks ×4), then the scorer on its candidates: frontal needs swerves the released candidates never contain | holds ~5.8 m/s, swerves 2–3 m; passes the parked car once, otherwise hits | 40% |
| s3 | DAgger round 1: s2 drove 150 episodes, the expert labelled them; both stages retrained from s2 on all three data sets: s2 never came back after a swerve | stays on its route, holds ~6.4 m/s, neither swerves nor brakes; hits all 6 | 29% |

**s1 over the full set**, paired seed for seed with the released sweep of 5
September: 3.671 against 3.568, 83 collisions against 91 (10 avoided, 2 new);
stationary +0.16, side +0.17, frontal −0.08 per pair. +0.103 with a pair-level
standard error of 0.079. Most of the side gain is side-0108 seeds 1 and 8, the
same two episodes the momentum variant flipped: the crossing car misses by a
hair. The released sweep ran on older harness code and was not re-run.

What the runs show, measured unless marked:

- The released model's stationary and side crashes are choices: a safe stop is
  among its own 32 plans on 93–96% of the moments before it creeps into the car.
- Every run that retrains the refiner (l5, s2, s3) drives faster than the expert
  and the released model, and loses the braking that scores 5.0 on 0103.
- The refiner can learn to propose swerves (on frontal 0075 a safe candidate
  exists on 87% of approach ticks after s3a, 68% before), and round 1 taught it to
  stay near its route. The scorer cannot then pick among them on an unseen scene:
  held-out top-choice accuracy 23% for s1 on the released candidates, 7% for s2b
  and 4% for s3b on the retrained ones. Retraining the scorer on s3a's candidates
  made that worse than keeping s2's (5.8%).
- The expert and the grader see only the inserted car. Measured on 1,200 sampled
  ticks it rarely matters to the labels (expert plans hit a parked car on 0–3.5%,
  all least-impact; 0–0.3% of safe-labelled candidates), so the labels were not
  redone; the safety check's any-car column watches for it.
- *Argued:* the expert holds speed and brakes late, so a model that cannot judge
  "late" learns the first half. The scorer cannot see the car on a new scene, so
  it has no reason to switch to braking.

## What is still wrong

**The drawn targets are not yet convincing to the person who has to trust them.**
This is the owner's standing judgement after reviewing them case by case against
the camera, and it is recorded here because it is the live objection, not a
resolved one. Four rounds of changes came out of that review — the delay before a
swerve begins, the recovery back to the lane, braking for what is not closing on
us, and the brake's own delay — and the targets still do not read as what a driver
would do. The specific complaints that remain open are the occluded-hazard
reactions below and a reaction that is triggered by geometry rather than by
sight. **Nothing here should be treated as a settled expert.**

**The scorer's lane term is absolute where its progress term is relative.** A
clear swerve is docked for leaving the lane even when the expert's own plan
leaves it, so on frontal ticks the expert's correct answer never scores full
marks. The preference the scorer is taught is **+0.594 for braking and +0.072 for
swerving** — eight to one, for a model we are trying to teach to swerve. The fix
is one line, by analogy with progress, and has not been trained on yet.

**The expert reacts to hazards it cannot see.** Not off-camera this time —
occluded. On side-0070 the inserted car is behind a tree line for the first three
of its braking ticks; the box projects fully on-screen, so the bearing test passes
all 45. The render is correct and the expert is wrong. A geometric test cannot
see occlusion; the clean fix is a second render per scene without the inserted
actor, diffed per tick, which would also catch anything else the reconstruction
hides. Not built.

**Reaction timing is bounded by the horizon, not by perception.** The expert first
stops saying `continue` at a time-to-contact of 3.85 s (frontal), which is the
edge of a 4 s planning horizon — it cannot see further, so anything inside is
something its plan must avoid. Distances that look absurd are not: 60–70 m at
13–17 m/s of closing speed is under four seconds. But reacting on geometry rather
than on sight is why it looks early, and gating the reaction on real visibility
would move it later **and** let it stay gentle, instead of trading one for the
other. The brake ladder was measured four ways:

| ordering | holds speed | peak decel | slows to |
|---|---|---|---|
| no delay at all | 0.0 s | 3.3 m/s² | 59% |
| gentlest, then as late as that allows | 1.0 s | 3.3 m/s² | 59% |
| **latest first, delays to 2.0 s** | **2.0 s** | **4.0 m/s²** | **30%** |
| latest first, delays to 3.5 s | 2.0 s | 4.6 m/s² | 30% |

The third is in use. The fourth buys no extra waiting and costs emergency
braking; nuScenes driving almost never passes 4 m/s².

## The loop is too slow to iterate on

A full round is about six and a half hours:

| stage | wall clock | measured where |
|---|---|---|
| collect, 150 episodes | **4 h 48 m** | round 0, 02:09 to 06:57, ~1.9 min an episode |
| label | 25 m | |
| train, one epoch | 40 m | 25 m once the frame cache is warm |
| evaluate — reach, 0103 smoke, videos | 35 m | |

That is far too long to answer "did this change help", and it is the reason four
expert revisions in one day each cost a relabel and a retrain without ever
reaching a closed-loop check. Three things would shorten it, in increasing order
of effort.

**Separate checking behaviour from collecting data.** Only a DAgger round needs
the 4 h 48 m; a behaviour check does not. The 0103 smoke plus `reach` is about
35 minutes and already answers "does it brake, does it swerve, does it still
drive" — it just needs to be the default first question rather than the thing
done at the end. The expert revisions above were judged on relabelled targets
when the quicker and more honest check was to drive them.

**Stop each episode when it stops teaching anything.** This is the largest lever
and it comes from where the signal actually sits. Episodes are 70 ticks, 7
seconds, and what the expert teaches across them is not uniform:

| tenth of the episode | continue | brake | swerve | least-impact |
|---|---|---|---|---|
| 0–10% | 13% | **73%** | 15% | 0% |
| 10–20% | 33% | 49% | 18% | 0% |
| 20–30% | 44% | 37% | 19% | 0% |
| 40–50% | 56% | 16% | **23%** | 5% |
| 60–70% | 61% | 8% | 3% | 27% |
| 80–90% | 62% | 5% | **0%** | 33% |
| 90–100% | 63% | 2% | **0%** | 35% |

The scenarios are authored so the actor is already on a collision course at t = 0,
so **the expert reacts from the very first tick** — there is no gentle opening to
get through. The waste is the tail. Swerve targets fall to 3% by 60% of the way in
and are zero through the last third, while least-impact climbs to 35%: the car has
already lost, and a policy that handles the decision never reaches those states at
all. Cutting each episode at about 60% keeps every swerve target and roughly 95%
of the brakes, and takes collection to **2 h 50 m**.

One caveat before dropping the tail: neuroNCAP scores impact speed, so
least-impact behaviour does earn partial credit when a crash is genuinely
unavoidable, and that is 1,381 of 9,495 ticks. Truncating is right by the
on-policy argument and may cost a little of that credit. Worth watching the
frontal number when it lands rather than assuming either way.

**Collect fewer seeds.** 3 instead of 10 keeps all 15 scenario-scenes and their
variety. For a check aimed only at the frontal gap, 3 seeds over the 5 frontal
pairs is 15 episodes. A round does not have to be the same size every time, and
the manifest already records episodes and ticks per round so a small round is
honestly labelled as one.

**Run the scenes in parallel.** Collection is one renderer per scene, strictly
serial, and a renderer is ~8 GB with the scene load measured at about a minute.
Five at once is ~40 GB plus the model node — affordable when the card is free,
which it often is not on a shared box.

Stacked, on round 0's measured pace:

| | wall clock |
|---|---|
| full episodes, 10 seeds, serial scenes | 4 h 48 m — what was done |
| truncated at 60% | 2 h 50 m |
| + 3 seeds | 50 m |
| + scenes in parallel | ~15 m |

Three independent changes, not one risky one. Nothing here is built. It is
written down because the iteration cost, not the GPU cost, is what actually
limited this experiment.

**And then let the rounds extend the horizon.** Truncation makes a round cheap;
it does not make the tail unnecessary forever. As the model starts surviving the
decision it reaches states around tick 40 that the current model never gets to,
and those exist only if it drives again. That is what a round is for, and it is
the argument for many small rounds over one large one.

## Instruments

Built because the obvious measurement was the wrong one more than once.

- [train/reach.py](../src/gallops/train/reach.py) — lateral reach of the 32 **and**
  how much the plan slows, both against the expert's own geometry. Comparing
  against `expert_offset_m` is wrong: it is 0 for `continue` by construction,
  while a correct `continue` plan on curving scene 0053 sweeps 1.93 m. That
  baseline produced one wrong diagnosis before it was caught. Its `escape`
  column carried defect 6 until the fix; the reach and slowing columns are
  ego-frame geometry and never depended on the pose.
- [train/preflight.py](../src/gallops/train/preflight.py) — runs inside the
  trainer before the first step and refuses to train on any failure: committed
  tree and passing tests, the trainer's grading reproducing the labels exactly,
  and the starting weights reproducing what they drove. Its report goes into the
  checkpoint's sidecar.
- [train/safety_check.py](../src/gallops/train/safety_check.py) — replays the
  creep and approach ticks of one scene through the released model and any
  finetunes, and says whether each pick is safe and why not, including hits on
  any other car in the scene. Its guard refuses to report below 99.5% agreement
  with the labels: an earlier copy graded against the logged human path while the
  labels used the lane centre, and only a training scene exposed it.
- [train/export.py](../src/gallops/train/export.py) — `model.pt` to the vendor's
  checkpoint shape, so paddock's adapter needs no change to evaluate a finetune.
- `ops/dj_video/compare.py` — two runs side by side, each panel carrying its own
  score, which is the only form in which "it swerved the wrong way" is obvious.
- `plot_expert_cases.py` — the camera the model read beside the same tick from
  above. Three of the six defects above were found in a picture.

## Reproducing

```bash
bash ops/dj_train/run_joint.sh                        # train, detached, resumable; passes --dumps
bash ops/dj_train/evaluate.sh <run> <step>            # reach, smoke on 0103, videos
python -m gallops.expert.rounds show --root <runs>    # which rounds exist
```

The trainer requires `--dumps`: the candidate dumps the labels were made from,
collected by the weights training starts from. The preflight re-grades them
through the trainer and stops the run if they disagree with the labels.

DAgger trains on every round at once, and each round was driven by a different
checkpoint, so each data set comes as its own `--source LABELLED DUMPS DROVE`:

    --source labelled_round0_v5 collect_dumps            start   # the released model drove
    --source labelled_gt        collect_dumps_gt         start   # the human's path; the released model's plans
    --source labelled_round1_s2 collect_dumps_round1-s2  s2b     # s2 drove

Every source is re-graded against its own dumps. Fidelity -- the starting weights
reproduce what they drove -- runs only on `start` sources; a round 1 dump was made
by s2, and comparing the released weights with it would fail for the right reason.

The stages since s1 train one part at a time:

    --train-only scorer --rank-weight 1.0                    # s1, and the second stage of s2, s3
    --train-only refiner --oversample swerve=4               # the first stage of s2, s3
    --init <checkpoint>/model.pt                             # start from a previous run, after the preflight
    --exclude-episodes frontal-0064                          # its inserted car never renders

    python -m gallops.train.safety_check --labelled <dir> --drivable <dir> \
        --scene 0053 [--shape frontal --kind swerve] --checkpoint NAME=<model.pt> --out check.json
Episodes are keyed by their directory, not their name: round 1 replays round 0's
scenarios and seeds, so `stationary-0070/run_3` and its timestamps exist once per round.

The held-out scene is **0053**, not 0058: 0058 has 54 swerve ticks of 1,673,
exactly one of them frontal, so its validation loss measures ordinary driving.

## What would come next

0. **Retrain under the fix** — done: v4fix, l4fix and every run after.
1. **Round 1 of the on-policy loop** — done, under s2 (s3). It taught recovery and
   cost the swerve and the braking.
2. **Can the model see the car?** Drive-JEPA's scorer has a collision head that
   inference never computes; the scorer-cache work tests it and sub-score heads
   held out scene by scene; it is written up in docs/scorer-cache.md in the PR after this one.
3. **Teach earlier, gentler braking**, the way the released model already drives,
   so the refiner stops learning "hold speed".
4. **Occlusion-aware visibility** and **the lane term**, both still open from the
   list below.
5. **Before any claim:** a NAVSIM check on any checkpoint kept, and the released
   baseline re-run on today's harness.

The earlier list, kept for the record:

In the order the evidence supports, not the order of effort.

0. **Retrain under the fix before judging anything above.** Every comparison in
   this document between checkpoints, and every expert or loss change judged by
   one, was made through scorers trained on partly wrong targets. v4's recipe with
   only the fix changed says what defect 6 cost; the same with the current expert
   says where the work actually stands.
1. **A faster check.** Make the 35-minute smoke the first question after any
   expert or loss change, before a relabel. Everything below is cheaper to judge
   once this is the habit.
2. **The lane term**, one line, no new data, retrains in 40 minutes.
3. **Occlusion-aware visibility**, a second render pass per scene. This is also
   the change most likely to make the drawn targets convincing, because it moves
   the reaction from "geometry says contact is coming" to "the car is visible and
   something must be done".
4. **Round 1 of the on-policy loop** — but under a checkpoint that actually
   swerves, and at 3 seeds rather than 10. v4 does not swerve, so a round
   collected under it revisits round 0's distribution with harder braking. 18
   episodes were collected under v4 and stopped for exactly this reason; they are
   kept, with a note, rather than folded in.
