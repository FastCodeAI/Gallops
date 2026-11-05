# S6 scorer diagnosis — predictions, version 4 (adds P-a5 to version 3)

Written 2026-10-08 on Sid's instruction, before any a, b or c table is written to the report. PREDICTIONS_v3.md (sha256
8fcdbd7107a2d507b46589d8b5be0d31fcfa33df8d55a8be8258ee483b984b52), PREDICTIONS_v2.md and PREDICTIONS.md stay in force word for word and
are not repeated here. This file adds one measurement to part a and one prediction, and is hashed and not edited afterwards.

## Added measurement (part a): top-2 margin

For each example and each scorer (shipped, pass 1, pass 9, each on its own menu, raw score; UNSEEN and SEEN):
- score gap = score of the top-scored plan minus score of the second (scores are the sigmoid outputs, 0–1);
- true-grade gap = absolute difference in EPDMS grade between those same two plans.

Reported per slice (all, highway, close-pass, ordinary): median score gap, share of examples with score gap < 0.01, and, among the
examples with score gap < 0.01, the share where the two plans differ in true grade by > 0.1. Intervals by clip-level bootstrap as
elsewhere.

## Added prediction

- **P-a5** (Sid's). On pass 9, UNSEEN, all graded examples: (i) more than 70 % of examples have a top-2 score gap < 0.01, and
  (ii) in more than 30 % of those, the two plans differ in true grade by more than 0.1.
  Both halves must hold for P-a5 to hold; each half is reported separately.

## What I had already seen when this was written (disclosure)

- **Half (i) is NOT blind.** On the first 1,024 UNSEEN examples (34 clips) I computed the top-2 score gap: for pass 9 the median was
  0.0023 and 85.3 % of examples were below 0.01 (`repro_detail_1024.csv`, REPORT.md 2.5). Sid set the 70 % threshold after I reported
  that number.
- **Half (ii) is blind in the direct sense:** the true-grade gap between the top two plans has not been computed on any example.
  Indirectly related numbers I have seen on those 1,024 examples from a code test of `s6_analyze.py` (`table_a_c_PARTIAL.csv`,
  `table_menu_PARTIAL.csv`): pass-9 pick = true best 39.9 %, pick graded ≥ 0.9 53.8 %, mean regret 0.108, about 11.5 of 32 plans tied
  for the best grade and about 13.3 graded ≥ 0.9 per example, median rank agreement 0.35.
- 6,400 UNSEEN examples were dumped when this was written; nothing beyond the first 1,024 has been tabulated.
