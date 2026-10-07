# Teaching Drive-JEPA to evade

The record of the first end-to-end training experiment in this repo: what was
built, what each run measured, and the six defects the measurements exposed.
Numbers here are measurements unless the text says they are an argument — the
standard [conventions.md](conventions.md) holds both repos to.

**Status: not finished, and the expert is not agreed.** The best checkpoint beats
the released model on a three-pair subset of neuroNCAP, and it does so by braking
harder rather than by evading — the capability this was built to add is still
absent. The drawn targets have not yet convinced the person who has to trust
them, and the iteration loop at six and a half hours a round is too slow to
settle that. Both are open, and both are below.

**And every scorer here trained on targets graded from the wrong place.** Found on
2026-10-06, after the runs below: the trainer graded each tick as if the car were
where the episode's first-looked-up tick had it, 14–16 m off, which flipped a
fifth of the stationary verdicts and a third of the frontal ones. The rows below
are true measurements of a trainer that was partly wrong. Defect 6 has the
detail; the trainer now refuses to start until a preflight shows its grading
reproduces the labels.

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

Round 0: 150 episodes, 9,495 labelled ticks.

## What each run measured

neuroNCAP scene 0103, two runs per scenario. Every one is recorded
`valid: false` — three of twenty pairs under a checkpoint override is not a
neuroNCAP score, and `paddock report` excludes it.

| | frontal | stationary | side | mean |
|---|---|---|---|---|
| released (n=10) | 2.05–2.26, hits 10/10 | 5.0 ×10 | 5.0 ×10 | 4.05 |
| v1 — refiner only | **5.00, 5.00** | 0.00 | 0.00 / 5.00 | 2.56 |
| v2/v3 — brake-first expert | 1.602 | 5.00 | 2.537 | 3.05 |
| v4 — scorer scored against the expert | 2.153 / 5.000 | 5.0, 5.0 | 5.0, 5.0 | **4.53** |

**v4 is the best result and it is not the capability we wanted.** It drives like
the released model with harder braking: 0.11–0.21 m of lateral motion, slowing
from 7.7 to about 2 m/s. On held-out swerve ticks it moves 0.07 m where the
expert asks for 4.59. The frontal gain is lower impact speed, plus one seed of
two avoided entirely — not an evasion.

Two provenance notes on these rows. **The v4 checkpoint trained on
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

Its first real run refused to train, and the refusal was right. The rebuilt
inputs drift from what the model saw by an amount that depends on where in the
episode a tick sits, measured over 506 round-0 ticks: the model's one look into
the past is the front frame 0.5 s back, and for an episode's first half second
that was one of the engine's priming frames, which the logger never saves.
Nearest-stamp selection silently handed over the episode's first frame instead,
and the rebuilt pick landed a median 0.4–0.5 m (p90 2.4 m) from the one driven,
against 0.10 m once the real frame exists. About 7% of every round's ticks were
trained on a past the model never had. They are now skipped as unrebuildable;
saving the priming frames at collection would get them back. Past that, the
remaining gap is a median 0.29% of four seconds' travel, most likely the JPEG
round trip, which is what the fidelity limit is now set against.

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

    --source labelled_round0_v5 collect_dumps          start   # the released model drove
    --source labelled_round1_l5 collect_dumps_round1-l5 l5      # l5 drove

Every source is re-graded against its own dumps. Fidelity -- the starting weights
reproduce what they drove -- runs only on `start` sources; a round 1 dump was made
by l5, and comparing the released weights with it would fail for the right reason.
Episodes are keyed by their directory, not their name: round 1 replays round 0's
scenarios and seeds, so `stationary-0070/run_3` and its timestamps exist once per round.

The held-out scene is **0053**, not 0058: 0058 has 54 swerve ticks of 1,673,
exactly one of them frontal, so its validation loss measures ordinary driving.

## What would come next

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
