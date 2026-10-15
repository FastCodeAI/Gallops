# S8 — predictions, written before anything is run

Written 2026-10-08 after S7's REPORT.md, before any S8 computation and before opening any conversion code for L1.2.
The two predictions are Sid's, copied from the brief. The definitions are mine and are fixed here.
L3 has no prediction.

Data: S6 dumps (`{runs}/drivejepa_retrain/08_scorer_diagnosis/dump/unseen/`, pass 9: `menu_9`, `grade_9`, `states_main`,
`human`), S7's `h1_instants.parquet` and `h3_unseen_per_instant.parquet`, S3's stored catalogue grades and
`04_training_set/vocab_10240.npy`. L3: S7's `h2_exam_inspection_*.csv` and, read-only, the trial folder.

## P-L1

**Prediction.** The swing has structure (lag-1 autocorrelation < −0.5, i.e. alternating) AND the catalogue curves go
through a different conversion path than menu plans. If instead the swing is unstructured and the paths are identical, the
jaggedness is learned, not mechanical.

Definitions.
- Instants: UNSEEN highway with no menu plan graded ≥ 0.95 (1,443 in S7). Plan = S7's best menu plan of the instant
  (`h1_instants.parquet`, `best_slot`). Step speeds = S7's: 8 half-second steps, step length / 0.5 s, first step from the
  car's position.
- Lag-1 autocorrelation of one plan = Σ(v(k) − m)(v(k+1) − m) over k = 1..7, divided by Σ(v(k) − m)² over k = 1..8,
  m = the plan's mean step speed. Plans whose step speeds spread under 0.5 m/s are left out and counted (no swing to
  describe). For 8 unrelated values this number averages about −0.1; perfect alternation gives about −0.9; a steady
  speed-up or slow-down gives a positive value.
- **"Structure" half holds if the median over instants is < −0.5.** Also reported: share of plans < −0.5, the same for the
  best catalogue curve and the human path, and for plans from other groups.
- Other structure measures, reported, not part of the verdict: share of sign changes between consecutive speed changes
  (alternation = 100 %, noise = about 67 %); mean of the odd steps minus mean of the even steps; share of the plan's total
  absolute deviation from its mean that falls in steps 1–2 (even spread = 25 %); Spearman correlation across instants
  between swing (largest minus smallest step speed) and the car's speed, and swing as a share of mean speed.
- **"Different path" half holds if any step between the stored pose array and the calculator's graded states differs for
  catalogue curves and menu plans** (pose count, spacing, resampling, controller, frame), established from code with
  path:line. A difference in where the 8 poses come from (catalogue: every fifth of 40 stored poses; menu: emitted as 8)
  is reported but does not count as a different conversion path if everything after the 8 poses is the same code.
- L1.4 contrast: the same numbers on all UNSEEN ordinary instants (best menu plan), split by good plan / no good plan.
- L1.1 sample: 20 + 20 UNSEEN highway instants drawn with seed 20261009 from the no-good-plan and good-plan groups.
  "40-pose version the calculator graded" = the simulated states stored by S6 (`states_main`), moved into the car's
  frame; if the calculator's interpolated 40 poses (before its vehicle simulator) can be reproduced from code, both are drawn.

Not blind. From S6 2.3 I already know that the calculator wrapper takes 8 poses, that the vendored code interpolates to 40
poses at 0.1 s and runs a vehicle simulator, and that S6 cut the catalogue to the same 8 poses by `[:, 4::5]`. I expect
the paths to be the same at grading time; I have not read how S3 graded the catalogue (`s3_grade.py`) or how the generator
emits its 8 poses. From S7 (3.1.6) I know the mean profile of these plans: faster around 2 s, slower around 3 s, largest
step at step 4 on 32 %. I have not computed any autocorrelation.

## P-L2

**Prediction.** For the ranking head, picks graded 0 are < 8 % of examples but contribute > 50 % of total regret. If they
contribute < 25 %, regret is spread and the features are the ceiling.

Definitions.
- UNSEEN, all 11,842 instants, raw pick (no momentum nudge). Ranking head = S7's chosen head (256 → 64 → 1), picks from
  `h3_unseen_per_instant.parquet`; scorer = its own output; random = expected value over the 32 plans (each plan weight 1/32).
- Regret = best grade on the menu − grade of the pick. "Graded exactly 0" = total grade ≤ 1e-9.
- Share of total regret from picks graded 0 = sum of regret over those instants / sum of regret over all instants.
- Which gate zeroed a pick: every gate (collision, drivable area, traffic light, driving direction) whose stored value is
  0 is counted; picks with more than one, and picks graded 0 with no gate at 0, get their own rows.
- Bands: regret = 0 (≤ 1e-9), (0, 0.1], (0.1, 0.5], > 0.5. Percentiles 50 / 90 / 99. Slices all / highway / close-pass /
  ordinary. Intervals: clip-level bootstrap, 95 %, 1000 resamples, seed 20261007.
- Verdict, ranking head, all instants, point estimates: consistent if share of picks graded 0 < 8 % AND their share of
  total regret > 50 %. "Regret is spread" if the share of total regret < 25 %. Otherwise neither.

Not blind: S7 gave the ranking head's mean regret 0.110 and 90th percentile 0.36 (scorer 0.29), and S6 gave a 90th
percentile of 1.0 for the slot-0 baseline. No band share, no count of picks graded 0 and no gate breakdown has been computed.

## L3 (no prediction) — exam clips, inspection only, not an evaluation

Raw score = `S_plain`; leave / stay rule as S7 H2. Bands fixed now for the question "unsafe or merely worse": best leave
score < 0.2 = scored as failing; 0.2–0.6 = in between; ≥ 0.6 = merely worse. Already known from S7: medians 0.094
(d2715023) and 0.801 (b44a4a44).
