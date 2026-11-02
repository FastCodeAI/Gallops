# S7 — predictions, written before anything is run

Written 2026-10-08, after reading S6's REPORT.md and scripts and listing the stored files, before any S7 computation.
The three predictions are Sid's, copied from the brief. The definitions under each are mine; they are fixed here so
that no choice is made after seeing a number.

Data: S6 dumps `{runs}/drivejepa_retrain/08_scorer_diagnosis/dump/{unseen,seen}/shard_*.npz` (pass 9: `menu_9`, `slot_9`,
`logit_9`, `grade_9`, `maxspeed_9`, `human`, `states_main`), manifests `08_scorer_diagnosis/set_{unseen,seen}.parquet`,
S3's stored catalogue grades `{runs}/drivejepa_retrain/04_training_set/s1/cache/<clip>/<example>/anchors_scores_final.npy`,
catalogue `04_training_set/vocab_10240.npy`, and for H2 the trial's stored menus `07_closed_loop_test/menus/d016_<scene>/replan_*.npz`
(read only).

## P-H1 (highway: the menu is built wrong)

**Prediction.** On ≥ 60 % of the no-good-plan highway instants the limiting check is comfort or time buffer AND the best
menu plan's max speed exceeds the best catalogue curve's by > 15 %. Otherwise (spread across checks, or menu slower than
catalogue) the overshoot story is wrong.

Definitions.
- Highway instant = manifest `slice == "highway"`. No-good-plan = highest pass-9 menu grade < 0.95. Good-plan = ≥ 0.95.
  UNSEEN and SEEN are kept separate. **The verdict is read on UNSEEN**; SEEN is reported next to it.
- There are 8 checks plus the total, not 9 (S6 2.3): collision, drivable area, traffic light, driving direction (the four
  gates), progress, time buffer (time-to-collision), lane keeping, comfort (the four weighted checks).
- Best menu plan = highest total grade; among ties, the one whose end point is closest to the human's (S6's tie rule).
- Limiting check = the check that, when set to 1 with the other seven stored values left as they are, raises the
  recomputed total the most. Total = product of the four gates × weighted mean of the four weighted checks, weights read
  from the calculator's config; the formula is first checked against the stored totals and the largest error reported.
  Ties for the largest rise (within 1e-6) are reported as their own row and count as "comfort or time buffer" only if
  every tied check is comfort or time buffer. If no single check raises the total (two gates at zero), the row is
  "more than one gate failed". Known limit, stated now: stored progress is already zeroed by a failed gate, so removing a
  gate alone understates what that gate costs.
- Speeds are taken from the 8 poses (0.5 s apart, first step from the car's position): step speed = step length / 0.5 s;
  max speed = largest of the 8; end speed = the last. Same rule for menu plans, catalogue curves (cut to 8 poses by
  `[:, 4::5]`) and the human path.
- Best catalogue curve = highest stored grade among curves graded ≥ 0.95; among ties, closest end point to the human's.
  Secondary, reported beside it: the median max speed over all curves graded ≥ 0.95 at that instant.
- An instant satisfies P-H1 if (limiting check is comfort or time buffer) AND (best menu plan max speed > 1.15 × best
  catalogue curve max speed, primary definition). Instants with no catalogue curve ≥ 0.95 cannot satisfy it and stay in
  the denominator; the share over instants that have one is reported too.
- H1.4: car speed v0 = manifest `speed`. Deceleration asked in step k = (v(k−1) − v(k)) / 0.5 s. "First 1 s" = the larger
  of steps 1 and 2, floored at 0. The first step alone is reported too (the trial's `step1_decel`, `07_closed_loop_test/s5_menus.py`).
- Intervals: clip-level bootstrap, 95 %, 1000 resamples, seed 20261007 (as S6).

Not blind: I know from S6 that 56.5 % of UNSEEN highway instants have a menu plan ≥ 0.95 against 93.6 % for the catalogue,
that highway menu plans sit 3.1 m from their nearest catalogue curve, and that some menu slots are long fast plans
(S6 flag 17). I have not seen any per-check value of a best plan, nor any plan, curve or human speed.

## P-H2 (pull-out: the scorer prefers staying) — exam clips, inspection only, not an evaluation

**Prediction.** Median raw gap (stay minus leave) > 0.02 on both scenes, i.e. a real preference, not a tie. If instead
median raw gap < 0.02 and the nudge flips > 50 % of replans, the tie-break alone is the cause.

Definitions.
- Scenes d2715023 and b44a4a44, adapter 0.1.6, replans at or after handover (`t_after_ho ≥ 0`).
- Raw score = `S_plain` (the scorer's output before the momentum step); nudged = `pdm_score` (after it,
  (14 × raw + 2 × comfort) / 16); `chosen` is the car's pick.
- "Leave" = the plan's end position is ≥ 2 m to the side (|y| ≥ 2) or ≥ 5 m forward (x ≥ 5) of the car at that replan.
  Otherwise "stay". This differs from S6's and the trial's rule (end ≥ 2 m away in any direction).
- Gap = best stay score − best leave score. Replans with no stay plan or no leave plan are counted and left out.
- Flip = the raw top plan is a leave plan and the nudged top plan is a stay plan.
- Nothing is written under `07_closed_loop_test/`, and these two scenes are used for nothing else.

**Not blind, and largely so.** S6 flag 16 already reports, with its own rule, pick minus best pull-out plan at a median
0.551 (d2715023) and 0.072 (b44a4a44), and the momentum step changing 0 of 34 picks on each. So the direction of P-H2 is
known before this file is hashed. What is new in H2 is the different leave rule, raw against nudged separately, and the
per-replan tie and flip shares.

## P-H3 (everywhere: the lesson, not the features)

**Prediction.** The ranking head's UNSEEN regret < 0.09 (scorer 0.115) and rank agreement > 0.50 (scorer 0.39). If regret
stays > 0.10, the lesson was not the limit and the features are the ceiling.

Definitions.
- Input: `slot_9` (256 numbers per plan), standardised with the training clips' mean and spread. Target: `grade_9[..., -1]`.
  SEEN for training, UNSEEN for the final evaluation only. No other source.
- Split: SEEN clips 80 % train / 20 % validation, by clip, seed 20261008. All seeds 20261008 (bootstrap 20261007, as S6).
- Ranking heads: (a) linear, (b) 256 → 64 → 1 with one ReLU. Loss: for every pair of plans of one instant whose grades
  differ by ≥ 0.05, max(0, 0.1 − (output of the better − output of the worse)), averaged over pairs.
  Adam, learning rate 1e-3, 256 instants per batch, at most 60 passes, stop when validation mean regret has not improved
  for 8 passes, keep the best pass.
- The one regularisation choice: weight decay from {0, 1e-4, 1e-3, 1e-2}, by validation mean regret.
- "The ranking head" of P-H3 = whichever of (a), (b) has the lower validation mean regret. Chosen before UNSEEN is read.
  The other one is reported too.
- Baselines, same features and split: (i) the scorer's own total output `logit_9[..., -1]`; (ii) ridge regression on the
  grade, strength from {0.1, 1, 10, 100} by validation regret; (iii) the same architecture as the chosen ranking head
  trained with the scorer's loss form, `BCEWithLogitsLoss` on the grade as a soft target
  (`05_small_overfit_check/train_patch/drive_jepa_agent_s35.py:248`, loss object at `:99`), same optimiser, stopping rule and weight
  decay grid.
- Measures as S6 (`s6_analyze.py`): pick = highest output; regret mean and 90th percentile, share of picks graded ≥ 0.9,
  true-best rate, rank agreement = per-instant Spearman with average ranks, **median** (the scorer's 0.39 is a median;
  the mean is reported too), top-2 margin table. Slices all / highway / close-pass / ordinary.
- Check before reading anything else: baseline (i) must reproduce S6's 0.115 and 0.39 on UNSEEN.
- Momentum (H3.5): S6's offline stand-in (`s6_analyze.py:34-46`), score = sigmoid(output) for (i), (iii) and the ranking
  heads, predicted grade clipped to 0–1 for (ii). Stated now: a ranking head's output has no fixed scale, so the size of
  the nudge against it is arbitrary; this row is descriptive.
- Verdict on the raw pick, UNSEEN, all instants, point estimates: consistent if mean regret < 0.09 AND median rank
  agreement > 0.50. "Features are the ceiling" if mean regret > 0.10. Anything else: neither.

Not blind: S6 table 4.7 (a linear probe on the same 256 numbers separates grade ≥ 0.9 at 0.80 on UNSEEN against the
scorer's 0.76; comfort 0.95, progress 0.85). No ranking measure of any head other than the scorer's has been computed.
