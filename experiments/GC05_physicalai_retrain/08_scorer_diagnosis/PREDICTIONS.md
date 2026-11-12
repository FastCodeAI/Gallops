# S6 scorer diagnosis — predictions, written before checks a, b, c

Written 2026-10-07, after the Part 0 code reading and before any of a, b, c was run. This file is hashed and not edited afterwards;
corrections go in REPORT.md as dated notes.

Sets: UNSEEN = graded examples of val_full (clips with a map, never trained on). SEEN = clip-stratified sample of graded training
examples, same size. "Separation" = AUROC (0.5 = coin flip, 1.0 = perfect). Intervals are clip-level bootstrap, 95 %.

## Predictions as given in the brief (unchanged)

- **P-a1** (catalogue ruler, lenient). If the strict ruler is the main story, the share of picks with grade ≥ 0.9 on UNSEEN, pass 9,
  is > 70 % and mean regret < 0.05. Otherwise < 50 % and regret > 0.10.
  Reasoning: many curves tie near the top, so an exact-match ruler can call a good pick a miss.
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
- **P-c1**. If the scorer practised on the wrong curves, rank agreement on the menu is lower than on matched catalogue curves by > 0.15.
  Reasoning: a scorer trained on catalogue curves and used on generator plans would rank its training curves better.
- **P-c2**. True grades of menu plans are lower than the nearest-catalogue grades S4 used, especially on highway (speed overshoot
  ~1.7×); the "right plan on the menu" rate drops below 93 %.
  Reasoning: the generator's plans overshoot in speed and the nearest catalogue curve does not.

## What the Part 0 code reading already says about these (known before running; recorded here, not a change to the above)

Each line is a code fact with its location; REPORT.md Part 0 carries the detail.

1. The scorer has no way to score a catalogue curve. It reads one feature per menu slot from the generator's own trunk and maps it to
   6 numbers (`score_module/scorer.py:41-42`; slots are `init_feature`, `drive_jepa_model.py:33,62`). It never takes a path as input.
   So "the scorer's top catalogue curve" (P-a1) and "rank agreement on matched catalogue curves" (P-c1) are **not measurable as
   written**. Expected outcome for P-a1 and P-c1: not measurable; reported as such, not as consistent or inconsistent.
2. The scorer is trained on the generator's 32 menu plans with true EPDMS calculator grades computed during training
   (`drive_jepa_agent_s35.py:368-370, 389-397`), not on catalogue curves. The catalogue only supplies extra targets for the generator
   (`:306-354`).
3. S4 graded the 32 menu plans with the true EPDMS calculator, not with the nearest catalogue curve (`s4_train.py:112,120`). So the
   comparison in P-c2 has no "nearest-catalogue grades S4 used" side. S4's 93 % is "a plan within 1.5 m of the human is on the menu,
   close-pass examples" (`s4_report.py:44`), a distance, not a grade.
4. The scorer has 6 outputs, not 9 (`scorer.py:13`); they line up with the last 6 calculator columns (driving direction, progress,
   time-to-collision, lane keeping, comfort, total; `drive_jepa_agent_s35.py:245-248`). Only the total is trained: the weight on the
   other five is 0 (`drive_jepa_config.py:51-52`). There is no scorer output for collision, drivable area or traffic light.

## Added predictions (mine, same rules)

- **P-x1**. The five per-check outputs that carry zero training weight separate no better than the total output does on the same check
  (difference in separation < 0.05), for pass 1 and pass 9.
  Reasoning: with weight 0 nothing pushes them; any signal they carry is inherited from the shipped weights.
- **P-x2**. On UNSEEN, pass 9, menu ruler, the share of picks with true grade ≥ 0.9 is at least 15 points above the strict match rate
  (37.9 %), and the random-pick baseline is within 10 points of the scorer's lenient number.
  Reasoning: S4's own readout already stores the grade of the pick and of the best plan; ties at the top are common on ordinary examples.
- **P-x3**. Recomputing "pick = true best" on UNSEEN from S4's stored pass-9 readout reproduces 37.9 % exactly (same file), and a fresh
  CPU forward pass lands within ±3 points.
  Reasoning: S4's parity check found the same pick on 8 of 8 between CPU and GPU.
