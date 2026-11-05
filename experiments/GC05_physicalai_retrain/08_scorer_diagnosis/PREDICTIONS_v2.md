# S6 scorer diagnosis — predictions, version 2 (supersedes PREDICTIONS.md)

Written 2026-10-07 after Sid's reply to the Part 0 questions and before any of a, b, c was run. PREDICTIONS.md (version 1, sha256
8791c38f36df15ed2852b9efbcac174b826023b071d6bc3d3d24d646c9d51c1b, hashed 2026-10-07T22:33:23+05:30) is kept unchanged next to this file.
Version 1 had already been hashed when the change was requested, so the change is made as a new file, not as an edit.
Changes from version 1, on Sid's instruction: P-a1 and P-c1 dropped (the scorer cannot score a catalogue curve); P-c1' added.
Everything else is carried over word for word. This file is hashed and not edited afterwards.

Sets: UNSEEN = graded examples of val_full (clips with a map, never trained on). SEEN = clip-stratified sample of graded training
examples, same size. "Separation" = AUROC (0.5 = coin flip, 1.0 = perfect). Rank agreement = Spearman between the scorer's 32 scores and
the 32 true grades of one example. Intervals are clip-level bootstrap, 95 %.

## What I had already seen when this was written (disclosure)

From S4's stored pass-9 readout on UNSEEN (`06_first_training_run/run/readouts/val_full_pass09.parquet`, read during Part 0): strict match 37.9 %,
mean rank agreement 0.232, pick with grade ≥ 0.9 on 48.8 %, pick within 0.05 of best 61.8 %, mean regret 0.116, a plan with grade ≥ 0.95
on the menu on 65.9 %. The same five numbers for pass 1 and shipped. So the UNSEEN half of P-c1' and all of P-x2 are **not blind**.
I have not looked at any SEEN number (the training-probe readouts are unread), nor at any per-check or probe number.

## Predictions

- **P-a2** (menu ruler, lenient, true grades from c). If the ruler is the story, V3 under "pick has grade ≥ 0.9" jumps above 50 %.
  Otherwise it stays below 35 %.
  Reasoning: V3 asks for a pick within 1.5 m of the human; a pick can be far from the human and still grade well.
- **P-a3**. Lenient numbers for pass 1 ≥ pass 9 (the drop with passes is real, not a ruler artifact).
  Reasoning: if pass 1 is better only under the strict ruler, the drop is tie-breaking noise.
- **P-b1**. If the scorer is starved of streets, per-check results on SEEN are clearly better than UNSEEN (gap > 0.10 in separation)
  on the map-dependent checks AND a fresh probe trained on SEEN features also fails on UNSEEN.
  Reasoning: too few distinct mapped streets lets the scorer memorise the ones it saw.
- **P-b2**. If grades are not readable from a photo, the probe is weak even on SEEN for the map-dependent checks (separation < 0.70),
  while map-free checks (collision, progress) are readable (> 0.80).
  Reasoning: lane and direction rules live in the map, which the model never sees.
- **P-b3**. If the training loss is the issue, the probe on UNSEEN beats the scorer's own head on the same check by > 0.10 (the
  information is there; the scorer did not learn to use it).
  Reasoning: a simple readout doing better than the trained head points at how the head was trained.
- **P-c1'** (replaces P-c1). If the scorer was trained on true menu grades, rank agreement on the menu for pass 9 on SEEN is > 0.5 and
  on UNSEEN is < 0.3 (memorised); else both < 0.3 (can't learn).
  Reasoning: the scorer is trained on the calculator's grades of its own menu, so a gap between seen and unseen clips is memorising.
- **P-c2**. True grades of menu plans are lower than the nearest-catalogue grades S4 used, especially on highway (speed overshoot
  ~1.7×); the "right plan on the menu" rate drops below 93 %.
  Reasoning: the generator's plans overshoot in speed and the nearest catalogue curve does not.
  Note written with the prediction: S4 used true calculator grades, not nearest-catalogue grades (`s4_train.py:112,120`), and its 93 %
  is a distance rate, not a grade rate (`s4_report.py:44`). The first half is therefore measured as "true grade of each menu plan
  minus the stored grade of its nearest catalogue curve"; the second half as "any plan with true grade ≥ 0.95" against 93 %.

## Added predictions (mine, same rules; unchanged from version 1)

- **P-x1**. The five per-check outputs that carry zero training weight separate no better than the total output does on the same check
  (difference in separation < 0.05), for pass 1 and pass 9.
  Reasoning: with weight 0 nothing pushes them; any signal they carry is inherited from the shipped weights.
- **P-x2** (not blind, see disclosure). On UNSEEN, pass 9, menu ruler, the share of picks with true grade ≥ 0.9 is at least 15 points
  above the strict match rate (37.9 %), and the random-pick baseline is within 10 points of the scorer's lenient number.
  Reasoning: ties at the top are common on ordinary examples.
- **P-x3**. Recomputing "pick = true best" on UNSEEN from S4's stored pass-9 readout reproduces 37.9 % exactly (same file), and a fresh
  CPU forward pass lands within ±3 points.
  Reasoning: S4's parity check found the same pick on 8 of 8 between CPU and GPU.
