# H15 — predictions, written before anything trains

**Commit this file before anything trains.** Its commit date is the evidence that the predictions
came first; a prediction written after the result is not one.

- **Written:** 5 Nov 2026, by Sid (reviewed by Naren)
- **Hypothesis:** Carrying the planner's output embeddings across moments makes the picker steadier
  and earlier.

## Arms

All three use the released Drive-JEPA frozen (encoder, generator, scorer), the same saved outputs
(H05.r1's cache: per moment, the scorer input 32 × 256, the six logits per proposal, ego 11, the 32
proposals), and the same data order.

- **(a) carried.** The memory steps once per moment along each chain and is handed to the closest
  proposal of the next moment (match limit below). Memory length 1, 2 or 4 moments, chosen on the
  selection set; random warm-up 1–16 moments in training.
- **(b) wiped.** The same layer, same size and training, with its state emptied every moment.
- **(c) released.** The released final score, no memory.

**Momentum is off for all three arms offline and on for all three in the gate.** Offline, the pick
is the argmax of the (nudged) final score; in the gate, paddock's momentum stage acts on that score.

## What will be measured

**Report set:** the 214 vendor val logs. Intervals are paired, log-clustered bootstrap over those
logs, 2,000 draws, seed 0, 95%. For every metric, the bootstrap is of the **3-seed mean**, and each
seed's value is shown beside it.

| # | Role | Benchmark / subset | Metric | Baseline | Predicted | Counts as support if | Counts as refuted if |
|---|---|---|---|---|---|---|---|
| P1 | **primary** | offline, 214 val logs, every graded moment (n = 15,320) | mean true PDM grade of the pick (vendor `get_sub_score` final score) | (b) | (a) − (b) > 0; magnitude to be stated after the val logs are graded, before training | (a) − (b) > 0 and its CI excludes 0 | the CI includes 0, or (a) − (b) < 0 |
| P2a | secondary | the same logs, every back-to-back pair (n = 18,798) | flip rate: pick-to-pick distance > 1 m (mean waypoint distance over the 3.5 s two plans share, ego-motion aligned) | (b) | (a) better than (b) | reported with its CI; not required to exclude 0 | — |
| P2b | secondary | the same logs, pairs where both moments are graded (n = 12,768) | two-frame extended comfort of the picks | (b) | (a) better than (b) | reported with its CI; not required to exclude 0 | — |
| P3 | gate | paddock's regression gate, every benchmark, quick tier then closed tier | each benchmark's own score | released (`{paddock_site}/queue/baselines/drivejepa.toml`) | no change | no REGRESSION on any benchmark | any REGRESSION |
| P4 | gate | neuroNCAP closed tier, side pairs 0103 and 0110 | score; when braking starts; speed at impact | (b) | (a) above (b) | (a) > (b) and (a) brakes no later | (a) ≤ (b), or (a) brakes later |

**Why P1 has no number yet.** No true grades exist: the 32 proposals at each moment have not been
graded. The only spread measured so far is between the released model's own predicted scores
(0.003 between the top two, momentum off), which is not the scale of the true grade. The magnitude
will be written here, and committed, after the 214 val logs are graded and before any training step.

## Fixed before training

- **Match limit 2 m.** A new proposal takes the memory of the closest old proposal (old waypoints
  2–8 moved into the new frame against new waypoints 1–7, mean distance); beyond 2 m it starts empty.
  On 40 val logs the p90 of that closest distance is 1.02 m.
- **Lengths** 1, 2, 4 moments; warm-up 1–16. "No limit" is not tried: val chains have a median of 7
  moments, and only 16% of graded val moments have 16 earlier ones.
- **Seeds** 0, 1, 2 for every arm. Identical data order across arms.
- **Grading:** the vendor's training-time scorer (`get_sub_score`) on the 32 cached proposals, train
  metric cache, without the anchors load. Rate measured on 200 moments first; 4 cores.
- **Navtest re-grade check:** the released pick of run `navsim/drivejepa/20261023T202240-de2f00`,
  re-graded on 200 navtest tokens; the sub-scores both sides share must agree within a tolerance
  set from the 200-token timing run, and that tolerance is written here before training.

## Controls

- **(b) separates memory from capacity:** same layer, same size, same training, nothing carried.
- **(c) is the released model.** With momentum off offline, (c) is exactly the released pick that a
  zero nudge reproduces (checked in the preflight: 100% of picks).

## Decision rules (plan, :217-218)

- (a) beats (b), and (b) ≈ (c): memory helps; go on to the gate.
- (a) ≈ (b), and both beat (c): capacity or training; rethink.
- All equal: stop, and write it up here.
- Any arm worse than (c): stop, or fix.
- P1 passes and P2 fails: supported on grade; comfort not shown.
- P1 fails: not supported, whatever P2 shows.

## Selection

The memory length and the epoch are chosen on **100 logs drawn from the 978 train logs** (numpy
`default_rng(0).choice(..., 100, replace=False)` over the sorted log names). The list is
[selection_logs.txt](selection_logs.txt) (sha256 `423335a0…6e1a11`; 9,731 moments, 7,007 graded).
Training uses the other 878 logs. Everything is reported on the 214 val logs, and navtest is never
used for choosing.

## Not blind

Known before writing:
- the slot-stability measurement on 40 of the 214 val logs: released picks more than 1 m apart on
  45% of back-to-back moments; closest path to the old pick 0.34 m; top-two gap 0.003. These are
  predicted scores with momentum off (paddock-runs `ops/memory_h15/slot_stability_40.json`);
- H05's probe and control results;
- the released checkpoint was selected on these 214 val logs (best epoch by `val/score_epoch`,
  recorded in the checkpoint's callbacks), although its training used only the 978 train logs.
  So (c) has a small selection advantage on the report logs; (a) against (b) is unaffected.

No true grade, and no output of any memory, has been seen.
