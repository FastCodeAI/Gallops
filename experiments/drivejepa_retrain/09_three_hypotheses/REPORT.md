# S7 — Three hypotheses, three cheap checks

**STATUS 2026-10-08 23:30 IST: H1, H2 and H3 are complete. CPU only, no Drive-JEPA forward pass, nothing written outside
`09_three_hypotheses/` (local and NAS).** Paths are relative to `paddock-sid/Alpasim/analysis_3.0/Retrain-data/` unless they
start with `/`. All model numbers are pass 9 on its own menu, from S6's stored dumps.

## 1. The three numbers that matter

**H1:** on highway instants with no menu plan graded ≥ 0.95, comfort or time buffer is the limiting check and the best menu
plan's max speed is more than 15 % above the best catalogue curve's on **54.2 % (46.0–61.4)** of UNSEEN instants, below the
60 % line; and the speed excess is not a plan that drives too fast but a plan whose half-second steps are uneven (step speeds
spread over a median 18 m/s within one plan, against 1 m/s for the catalogue curve), with the same end speed as the
catalogue curve and the human. **H2 (exam clips, inspection only):** the scorer's raw preference for staying is a median
**0.84** on d2715023 and **0.12** on b44a4a44, never under 0.02 at any of the 68 replans, and the momentum nudge flips no
replan. **H3:** a ranking head on the same frozen 256 features orders the menu much better (rank agreement **0.57**
(0.55–0.59) against the scorer's 0.39) but does not pick better: UNSEEN regret **0.110 (0.099–0.123)** against 0.115.

## 2. Predictions

`PREDICTIONS.md`: sha256 `881eea0e4c73e690407b6626fe08beb0afb3421d3ca626f00480e1372e382b7b`, hashed
2026-10-08T23:07:50+05:30 (`predictions.sha256`; copy on the NAS), before any S7 computation. First S7 run started 23:09:34
(`state_log.csv`). The file states what was already known from S6 for each prediction.

| prediction | what it said | measured (UNSEEN, 95 % clip-bootstrap interval) | verdict |
|---|---|---|---|
| P-H1 | on ≥ 60 % of no-good-plan highway instants the limiting check is comfort or time buffer AND the best menu plan's max speed is > 15 % above the best catalogue curve's | both: 54.2 % (46.0–61.4). Limiting check alone: 62.2 % (54.3–68.9), all of it comfort; time buffer is the limiting check on 0.7 %. Speed alone: 64.4 % (56.3–70.8). SEEN: 57.4 % (50.9–63.4) | **inconsistent as written**, narrowly: the point estimate is under 60 % and the interval reaches 61.4 %. Not the "otherwise" branch either: the limiting check is not spread (comfort 62 %, lane keeping 18 %, a tie of the two 10 %) and the menu plan is slower than the catalogue curve on only 9 %. The post-hoc look (3.1.6) says "overshoot" is the wrong word for what the speed condition caught |
| P-H2 | median raw gap (stay minus leave) > 0.02 on both scenes; the alternative was a median under 0.02 with the nudge flipping > 50 % of replans | d2715023: 0.841 (0.539 to 0.963 over 34 replans). b44a4a44: 0.115 (0.040 to 0.423). Replans with raw gap < 0.02: 0 of 34 and 0 of 34. Flips by the nudge: 0 and 0. No interval: two scenes, inspection only | **consistent**. Not blind (flag 3) |
| P-H3 | ranking head UNSEEN regret < 0.09 and rank agreement > 0.50; if regret stays > 0.10 the features are the ceiling | regret 0.110 (0.099–0.123); rank agreement, median 0.569 (0.549–0.588) | **inconsistent**: the rank-agreement half holds, the regret half does not. Regret is above 0.10 at the point estimate, so by the rule written beforehand this is the "features are the ceiling" branch; the interval's lower end (0.099) touches the line. Regret minus the scorer's, paired: −0.005 (−0.011 to +0.001) |

## 3. Tables

### 3.1 H1 — highway instants, pass-9 menu (`s7_h1.py`; per instant `h1_instants.parquet`)

Definitions (fixed in `PREDICTIONS.md`): best menu plan = highest grade, closest end point to the human among ties.
Limiting check = the one check that, set to 1 with the others left as stored, raises the recomputed grade most. Grade =
product of the four gates × (5 × progress + 5 × time buffer + 2 × lane keeping + 2 × comfort) / 14; this formula
reproduces every stored grade of both sets to 0.000000 (`h1_tables.log`; weights from the calculator's config). There are 8
checks plus the total, not 9. Speeds come from the 8 poses: step speed = step length / 0.5 s, max speed = the largest of
the 8, end speed = the last. Best catalogue curve = highest stored grade ≥ 0.95, closest end point to the human among ties.

**3.1.1 Counts (H1.1).**

| set | highway instants | clips | no menu plan ≥ 0.95 | share | of those, a catalogue curve ≥ 0.95 exists |
|---|---|---|---|---|---|
| UNSEEN | 3,315 | 110 | 1,443 (93 clips) | 43.5 % (37.2–50.8) | 86.0 % |
| SEEN | 3,475 | 205 | 1,445 (177 clips) | 41.6 % (37.2–46.2) | 84.3 % |

**3.1.2 Limiting check of the best menu plan (H1.2).** Share of instants in the group.

| limiting check | UNSEEN, no good plan (1,443) | SEEN, no good plan (1,445) | UNSEEN, good plan (1,872) | SEEN, good plan (2,030) |
|---|---|---|---|---|
| comfort | 61.5 % (53.6–68.2) | 63.9 % (57.8–69.2) | 0 | 0 |
| lane keeping | 18.3 % (12.8–25.4) | 21.0 % (16.5–26.1) | 0 | 0 |
| tie, lane keeping and comfort (both failed) | 10.1 % (6.2–13.9) | 7.1 % (5.2–9.2) | 0 | 0 |
| progress | 5.9 % (3.7–8.4) | 4.2 % (2.8–5.5) | 23.7 % (18.4–29.8) | 26.8 % (22.6–31.4) |
| time buffer | 0.7 % (0.1–1.5) | 1.2 % (0.4–2.4) | 0 | 0 |
| driving direction (gate) | 0.9 % | 1.4 % | 0 | 0 |
| drivable area (gate) | 0.7 % | 0.1 % | 0 | 0 |
| collision (gate) | 0.1 % | 0 | 0 | 0 |
| more than one gate failed | 1.7 % | 1.1 % | 0 | 0 |
| none, the grade is 1 | 0 | 0 | 76.3 % (70.2–81.6) | 73.2 % (68.6–77.4) |
| best plan's grade, mean | 0.805 | 0.820 | 0.995 | 0.995 |

Comfort and lane keeping are pass/fail with weight 2 of 14, so a plan failing one of them grades 0.857 at best: 0.143 is
the rise in every comfort, lane-keeping and tie row. On no-good-plan instants the best plan fails comfort on 75.1 % and
lane keeping on 30.8 % (UNSEEN; `h1_posthoc_summary.csv`).

**3.1.3 Speed (H1.3).** Ratios per instant; "catalogue" columns use the instants that have a catalogue curve ≥ 0.95.

| measure | UNSEEN, no good plan | SEEN, no good plan | UNSEEN, good plan | SEEN, good plan |
|---|---|---|---|---|
| max speed, menu / catalogue: mean | 1.84 (1.67–2.00) | 1.55 (1.47–1.62) | 1.06 (1.05–1.07) | 1.06 (1.05–1.08) |
| max speed, menu / catalogue: median (25th, 90th pct) | 1.35 (1.15, 3.27) | 1.33 (1.15, 2.11) | 1.04 (1.00, 1.15) | 1.05 (1.00, 1.15) |
| end speed, menu / catalogue: mean; median | 1.00 (0.96–1.03); 1.01 | 1.03 (1.00–1.05); 1.02 | 0.94 (0.93–0.96); 0.95 | 0.94 (0.93–0.95); 0.96 |
| end speed, menu / human: mean; median (25th, 90th pct) | 1.01 (0.97–1.04); 1.02 (0.87, 1.32) | 1.05 (1.02–1.09); 1.03 (0.91, 1.33) | 1.01 (0.98–1.06); 1.00 (0.92, 1.14) | 0.99 (0.98–1.00); 0.99 (0.91, 1.12) |
| max speed, menu / human: median | 1.33 | 1.30 | 1.04 | 1.04 |
| menu max speed above the catalogue's | 89.1 % (85.7–91.9) | 88.9 % (85.9–91.7) | 72.6 % (69.3–75.5) | 76.1 % (73.4–78.9) |
| menu max speed below the catalogue's | 10.9 % | 11.1 % | 27.4 % | 23.9 % |
| menu max speed > 1.15 × catalogue's | 74.9 % (67.8–80.3) | 74.5 % (68.7–79.5) | 10.0 % (7.1–13.0) | 11.2 % (9.2–13.2) |
| menu max speed > 1.15 × median of all catalogue curves ≥ 0.95 (secondary) | 78.6 % | 80.1 % | 29.4 % | 27.1 % |
| mean speeds, m/s: car now / human end / catalogue end / menu end | 26.9 / 27.1 / 28.2 / 27.2 | 26.7 / 26.9 / 28.4 / 28.2 | 18.2 / 17.9 / 18.7 / 17.5 | 19.2 / 19.0 / 19.6 / 18.5 |
| mean max speed, m/s: menu (8 poses) / menu (calculator's simulation) / catalogue | 52.4 / 31.9 / 28.5 | 43.7 / 31.8 / 28.8 | 20.1 / 19.2 / 19.0 | 21.3 / 20.2 / 20.1 |

**3.1.4 Handover (H1.4).** Deceleration asked in the first 1 s, m/s², from the car's speed and the first two steps.

| measure | UNSEEN, no good plan | SEEN, no good plan | UNSEEN, good plan | SEEN, good plan |
|---|---|---|---|---|
| best menu plan: median; mean; 90th pct | 3.9; 12.5 (9.0–16.8); 20.7 | 3.6; 7.2 (5.9–8.6); 8.8 | 1.0; 2.9 (2.1–4.1); 5.0 | 1.2; 3.0 (2.3–4.0); 4.6 |
| best catalogue curve: median; mean; 90th pct | 1.1; 2.0 (1.7–2.3); 5.4 | 0.3; 1.3 (1.2–1.5); 4.4 | 1.3; 2.1 (1.8–2.4); 5.3 | 0.5; 1.6 (1.4–1.8); 5.0 |
| menu minus catalogue, mean | +11.5 (7.6–15.8) | +6.7 (5.2–8.1) | +0.8 (−0.0 to +2.1) | +1.5 (0.7–2.5) |
| best menu plan asks > 3 m/s²; > 6 m/s² | 61.7 %; 29.0 % | 57.4 %; 23.6 % | 20.3 %; 7.4 % | 20.9 %; 5.9 % |
| best catalogue curve asks > 3 m/s²; > 6 m/s² | 25.9 %; 8.1 % | 17.0 %; 4.4 % | 28.8 %; 7.6 % | 20.7 %; 5.7 % |
| first step alone (the trial's measure), menu: median; > 6 m/s² | 3.2; 21.2 % | 3.2; 19.0 % | 0.8; 4.1 % | 1.0; 2.8 % |

**3.1.5 Contrast (H1.5): the difference is the finding.** Where a good menu plan exists, its best plan fails no pass/fail
check, its max speed is within 6 % of the catalogue curve's, and its first-second deceleration matches the catalogue
curve's (1.0 against 1.3 m/s² median). Where none exists, the best plan fails comfort or lane keeping on 90 %, its largest
step speed is a third above the catalogue curve's at the median and nearly double on average, and it asks 3.9 m/s² in the
first second against 1.1. End speeds are the same in both groups (ratio 1.00 and 0.94 to the catalogue, 1.01 and 1.01 to
the human). SEEN and UNSEEN agree on every row.

**3.1.6 Added after the tables above were read (not part of P-H1; `s7_h1_posthoc.py`).** The 8-pose max speed of the best
menu plan (52 m/s mean) is far above both its end speed (27 m/s) and the max speed the calculator simulates for the same
plan (32 m/s). The reason is uneven spacing of the 8 poses, not sustained speed.

| measure | UNSEEN, no good plan | SEEN, no good plan | UNSEEN, good plan | SEEN, good plan |
|---|---|---|---|---|
| spread of the 8 step speeds within the plan, median, m/s: menu / catalogue / human | 18.3 / 1.1 / 0.6 | 17.0 / 0.9 / 0.8 | 3.6 / 1.3 / 1.0 | 3.8 / 1.1 / 1.1 |
| plans with a spread over 5 m/s: menu / catalogue / human | 81.6 % (74.7–86.7) / 3.0 % / 4.6 % | 81.7 % (76.8–85.8) / 5.0 % / 5.3 % | 26.2 % / 4.3 % / 7.6 % | 30.7 % / 5.0 % / 7.2 % |
| of best plans that fail comfort, spread over 5 m/s | 97.5 % | 97.1 % | — | — |
| menu step speed minus the car's speed, median, at 2.0 s / 3.0 s | +2.5 / −5.1 m/s | +2.9 / −3.7 m/s | +0.5 / −0.1 | +0.4 / −0.0 |
| P-H1's speed condition with the calculator's simulated max speed instead | 35.3 % (29.2–40.8) | 36.1 % (30.8–40.5) | 8.4 % | 6.7 % |
| P-H1's joint condition with the simulated max speed | 29.5 % (23.4–35.0) | 33.1 % (27.7–37.6) | 0 | 0 |
| simulated max speed / catalogue max speed, median | 1.12 | 1.13 | 1.01 | 1.01 |

Reading: on these instants the best plan on the menu surges around 2 s and falls back around 3 s, ending at the human's
speed. That is what fails comfort. It is a plan that is unevenly spaced, more than a plan that is too fast.

### 3.2 H2 — EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION (`s7_h2.py`)

Scenes d2715023 and b44a4a44, adapter 0.1.6, read-only on `07_closed_loop_test/menus/d016_<scene>/replan_*.npz`; 34 replans each,
0.5 s apart from handover. Raw = `S_plain`, nudged = `pdm_score` = (14 × raw + 2 × comfort) / 16
(`07_closed_loop_test/s5_menus.py:49-50`). Rule: **leave** = the plan ends ≥ 2 m to the side or ≥ 5 m forward of the car at that
replan; otherwise **stay**. Gap = best stay score − best leave score. Per replan: `h2_exam_inspection_replans.csv`; all 32
scores with tags (H2.1): `h2_exam_inspection_scores32.csv`; summary: `h2_exam_inspection_summary.csv`.

| exam clips, inspection only, not an evaluation | d2715023 | b44a4a44 |
|---|---|---|
| replans (all have both kinds of plan) | 34 | 34 |
| leave plans among the 32, median (range) | 11 (9–13) | 18.5 (15–21) |
| raw score, best stay plan, median | 0.945 | 0.918 |
| raw score, best leave plan, median | 0.094 | 0.801 |
| raw gap, median (smallest to largest) | 0.841 (0.539 to 0.963) | 0.115 (0.040 to 0.423) |
| nudged gap, median (smallest to largest) | 0.861 (0.597 to 0.968) | 0.225 (0.093 to 0.495) |
| raw gap, first 5 s / after 5 s, median | 0.786 / 0.875 | 0.068 / 0.123 |
| replans with raw gap < 0.02 | 0 of 34 | 0 of 34 |
| rank of the best leave plan, raw, median (best to worst) | 20 (16 to 23) | 13 (6 to 18) |
| rank of the best leave plan, nudged, median | 20 | 13.5 |
| raw top plan is a leave plan | 0 of 34 | 0 of 34 |
| nudge flipped a would-be leave pick to stay | 0 of 34 | 0 of 34 |
| car's pick is a leave plan | 0 of 34 | 0 of 34 |
| car's pick, end position forward, median (range) | 0.31 m (−0.03 to 1.36) | 0.02 m (−1.61 to 0.61) |
| comfort term of the nudge, mean over leave plans / over stay plans, averaged over replans | 0.00 / 0.71 | 0.01 / 0.85 |

The preference is real on both scenes and at every replan; it is not a tie. The nudge pushes the same way (leave plans get
a comfort term of 0 next to a stationary previous plan, which widens the gap, by about 0.1 on b44a4a44) but it decides nothing: the
raw top plan is already a stay plan 68 times out of 68. On b44a4a44 the gap is at its smallest, 0.04–0.07, between 2.5 and
5.5 s after handover and never closes.

### 3.3 H3 — a ranking lesson on the same 256 features (`s7_h3.py`)

Set-up as pre-registered: SEEN split by clip into 9,476 training instants (582 clips) and 2,366 validation instants (145
clips), seed 20261008; 2,289,175 training pairs with grades ≥ 0.05 apart. Choices made on SEEN validation regret and written
to `h3_selection.json` at 23:22:52, before UNSEEN was opened at 23:23:07 (`h3.log:73`): the ranking head is the
256 → 64 → 1 network (validation regret 0.079 against 0.098 for linear), weight decay 0.01; scorer's-loss head weight decay
0.001; ridge strength 100. Baseline (iii) reproduces `05_small_overfit_check/train_patch/drive_jepa_agent_s35.py:248`,
`final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])`, loss object `nn.BCEWithLogitsLoss()`
at `:99`. Check: the scorer row reproduces S6 (regret 0.1149, rank agreement 0.386 against S6's 0.115 and 0.39).

**Table 3.3.1 — UNSEEN, all 11,842 instants, 401 clips, raw pick.** Brackets: 95 % interval, clip-level bootstrap.

| head on the 256 features | regret, mean | regret, 90th pct | pick graded ≥ 0.9, % | pick = true best, % | rank agreement, median | regret minus the scorer's |
|---|---|---|---|---|---|---|
| (i) the scorer's own output | 0.115 (0.105–0.125) | 0.29 (0.23–0.40) | 49.5 (45.8–52.7) | 38.3 (35.6–41.3) | 0.39 (0.35–0.42) | — |
| (ii) ridge regression on the grade | 0.119 (0.109–0.129) | 0.28 (0.23–0.37) | 45.5 (42.1–48.7) | 34.3 (31.9–36.9) | 0.31 (0.28–0.34) | +0.004 (−0.003 to +0.011) |
| (iii) scorer's loss form, 256 → 64 → 1 | 0.108 (0.098–0.118) | 0.24 (0.20–0.34) | 49.3 (45.7–52.7) | 38.9 (36.4–41.7) | 0.44 (0.42–0.47) | −0.007 (−0.013 to −0.002) |
| **ranking head, 256 → 64 → 1** | **0.110 (0.099–0.123)** | 0.36 (0.21–0.50) | 51.3 (47.6–54.5) | 47.3 (44.6–50.1) | **0.57 (0.55–0.59)** | −0.005 (−0.011 to +0.001) |
| ranking head, linear (not chosen) | 0.127 (0.115–0.140) | 0.51 (0.37–0.73) | 49.8 (46.3–53.0) | 46.2 (43.6–48.9) | 0.54 (0.52–0.55) | +0.012 (+0.005 to +0.020) |
| random pick (expected) | 0.210 (0.197–0.224) | — | 38.6 | 33.2 | — | — |

**Table 3.3.2 — UNSEEN by slice, raw pick.** Regret mean; pick graded ≥ 0.9; true best; rank agreement median.

| head | highway (3,315) | close-pass (1,563) | ordinary (6,964) |
|---|---|---|---|
| (i) scorer | 0.149 (0.128–0.171); 35.6 %; 11.9 %; 0.06 | 0.150 (0.119–0.182); 33.5 %; 43.5 %; 0.43 | 0.091 (0.079–0.103); 59.7 %; 49.7 %; 0.51 |
| (ii) ridge | 0.156 (0.134–0.182); 34.4 %; 16.5 %; 0.23 | 0.131 (0.110–0.154); 30.5 %; 39.3 %; 0.35 | 0.098 (0.087–0.110); 54.2 %; 41.6 %; 0.34 |
| (iii) scorer's loss form | 0.134 (0.113–0.160); 35.7 %; 20.2 %; 0.41 | 0.133 (0.109–0.162); 33.4 %; 38.4 %; 0.40 | 0.089 (0.077–0.101); 59.3 %; 48.0 %; 0.47 |
| ranking head | 0.134 (0.111–0.160); 38.1 %; 24.3 %; 0.51 | 0.146 (0.117–0.178); 35.1 %; 50.4 %; 0.55 | 0.091 (0.078–0.104); 61.1 %; 57.6 %; 0.60 |
| ranking head, linear | 0.152 (0.127–0.179); 36.2 %; 23.4 %; 0.48 | 0.149 (0.124–0.176); 35.1 %; 50.4 %; 0.54 | 0.110 (0.094–0.126); 59.6 %; 56.1 %; 0.57 |
| random pick | 0.208; 26.9 %; 15.8 % | 0.267; 26.2 %; 35.8 % | 0.199; 47.0 %; 40.9 % |

90th-percentile regret by slice (scorer / ranking head): highway 0.42 / 0.43, close-pass 0.62 / 0.66, ordinary 0.19 / 0.18.

**Table 3.3.3 — with the momentum nudge, as in the car (H3.5), UNSEEN, all.** S6's offline stand-in for the previous plan;
applied on 95.9 % of instants.

| head | regret, mean | pick graded ≥ 0.9, % | pick = true best, % | pick changed by the nudge |
|---|---|---|---|---|
| (i) scorer | 0.113 (0.103–0.124) | 49.8 | 38.5 | 32.5 % |
| (ii) ridge | 0.116 (0.106–0.127) | 47.3 | 34.5 | 41.1 % |
| (iii) scorer's loss form | 0.105 (0.095–0.115) | 49.8 | 38.6 | 32.5 % |
| ranking head | 0.113 (0.102–0.125) | 51.2 | 46.4 | 30.7 % |
| ranking head, linear | 0.127 (0.115–0.140) | 49.9 | 44.0 | 33.0 % |

With the nudge the ranking head and the scorer have the same regret (0.113). On highway: scorer 0.148, ranking head 0.136.

**Table 3.3.4 — top-2 margin, UNSEEN, all.** Score gap on the 0–1 scale used for the nudge (sigmoid of the output; the
clipped predicted grade for ridge).

| head | score gap between the top two, median | gap < 0.01, % | of those: true grades differ by > 0.1, % | true grades of the top two differ by > 0.1, all instants, % | top plan is the worse one by > 0.1, % |
|---|---|---|---|---|---|
| (i) scorer | 0.0025 | 84.0 | 11.5 | 13.9 | 6.8 |
| (ii) ridge | 0.0073 | 58.8 | 12.1 | 15.3 | 7.7 |
| (iii) scorer's loss form | 0.0033 | 76.7 | 10.0 | 12.8 | 6.2 |
| ranking head | 0.0014 | 99.1 | 11.6 | 11.7 | 5.6 |
| ranking head, linear | 0.0023 | 95.5 | 13.4 | 13.9 | 6.7 |

Highway, share of instants where the top plan is the worse of the top two by > 0.1: scorer 11.4 %, ranking head 8.0 %.
The ranking head's score-gap columns say little (flag 8); the last two columns do not depend on scale.

**Table 3.3.5 — SEEN validation clips (145 clips, 2,366 instants; the heads did not train on them, the scorer did).**

| head | regret, mean | pick = true best, % | rank agreement, median |
|---|---|---|---|
| (i) scorer | 0.072 (0.060–0.085) | 46.6 | 0.54 |
| (ii) ridge | 0.095 (0.081–0.110) | 37.0 | 0.33 |
| (iii) scorer's loss form | 0.077 (0.064–0.091) | 44.7 | 0.50 |
| ranking head | 0.079 (0.064–0.094) | 52.4 | 0.61 |
| ranking head, linear | 0.098 (0.080–0.116) | 50.0 | 0.56 |

Reading. The ranking lesson does what a ranking lesson should: order (0.39 → 0.57), true-best rate (38 → 47 %), and
highway order most of all (0.06 → 0.51). It does not lower regret: three different lessons on these features land at
0.108–0.115 on UNSEEN, against 0.077–0.079 on held-out SEEN clips. The ranking head finds the best plan more often and,
when wrong, is wrong by more (90th-percentile regret 0.36 against 0.29). What limits regret on UNSEEN is shared by all the
heads, so it is not the lesson; that this is the features rather than the 9,476 training instants is not shown (flag 9).

## 4. Flags

1. **Run order.** H1's first run computed and saved every per-instant value, then stopped in the table step on a bug of mine
   (`x.clip` is a pandas method, not the clip column; `h1_attempt1_table_bug.log`). H2 and H3 started at 23:21:48 and
   23:21:53; H1's tables were rebuilt from the saved `h1_instants.parquet` at 23:22:07 (`--tables-only`) with the column
   access fixed and no definition changed. So H1's tables were finished after H2 and H3 had started. Nothing from H2 or H3
   had been read when the fix was made.
2. **A wait loop of mine held the queue for 7 minutes** (23:14:53 to 23:21:48): it looked for a running `s7_h1.py` and
   matched its own command line. Nothing ran in that time.
3. **P-H2 was not blind.** S6 flag 16 already showed the pick ahead of the best pull-out plan by a median 0.551 and 0.072
   and the nudge changing 0 of 34 picks on each scene. Stated in `PREDICTIONS.md` before hashing.
4. **H2's leave rule is not S6's.** S6 and the trial count a plan as pulling out if it ends ≥ 2 m away in any direction;
   H2's rule (≥ 2 m sideways or ≥ 5 m forward) moves plans that creep 2–5 m forward to "stay": leave plans are 11 and 18.5
   of 32 here against 21 and 24 there, and the best leave plan scores 0.09 and 0.80 against 0.38 and 0.85. The verdict is
   the same under both rules (S6's smallest gaps were 0.123 and 0.041, both above 0.02). The car's pick itself ends within
   1.4 m of where it stands at every replan.
5. **H1's "max speed" measured unevenness, not speed** (3.1.6, post hoc). With the calculator's simulated max speed the
   speed condition holds on 35 % of no-good-plan instants and P-H1's joint condition on 29.5 %. The verdict in section 2
   uses the definition that was pre-registered; this is the more useful number. The 3.1.6 table was chosen after seeing 3.1.3.
6. **The brief says 9 checks; there are 8 plus the total** (S6 2.3). "Time buffer" (time-to-collision) is the limiting
   check on under 1.5 % of no-good-plan instants: P-H1's "comfort or time buffer" is in practice comfort alone.
7. **The limiting-check rule understates the gates.** Stored progress is already zeroed when a gate fails, so setting one
   gate to 1 does not restore the progress it cost. Gates are the limiting check on 3.4 % (UNSEEN) with this rule.
   Stated in `PREDICTIONS.md`.
8. **A ranking head's output has no fixed scale**, so its score gaps (99 % under 0.01) and the size of the momentum nudge
   against it are arbitrary. Table 3.3.3's ranking-head rows are descriptive. Stated in `PREDICTIONS.md`.
9. **H3's heads saw 9,476 graded instants; the scorer was trained on 21,806, with the whole network.** "The features are
   the ceiling" is the branch the pre-registered rule selects, but H3 cannot separate the features from the amount of
   graded data. The seen-to-unseen gap of the fresh heads (0.079 → 0.110) is about the size of the scorer's (0.072 → 0.115).
10. **Ridge strength made no difference** (validation regret 0.0950, 0.0950, 0.0950, 0.0949 over the grid): the grid was too
    weak for 303,232 training plans. Not re-tuned.
11. **196 of 9,476 training instants have no pair of plans 0.05 apart** and contribute nothing to the ranking loss.
12. **No catalogue curve graded ≥ 0.95 on 14.0 % (UNSEEN) and 15.7 % (SEEN) of no-good-plan highway instants.** They stay
    in P-H1's denominator; over instants that have one, the joint share is 63.0 % (55.2–69.7) UNSEEN and 68.1 % SEEN. As in
    S6, the catalogue side is a floor (S3's pre-filter left some curves ungraded).
13. **Best catalogue curve is tie-broken toward the human**, which pulls its speed toward the human's. The secondary
    definition (median over all curves ≥ 0.95) gives the same picture: joint share 55.8 % (47.4–62.9).
14. **Start gate.** The brief's gate (load < 32, free swap > 15 GB) and S6's later RAM gate (available RAM > 40 GB) were
    both applied; both held at every start (`state_log.csv`). No long run: the whole of S7 took under 5 minutes of CPU.
15. **Before S7 started, a directory search of mine walked `{nas}` for about 3 minutes** (read-only, to locate the S6
    folders) and was stopped. It wrote nothing.
16. **`07_closed_loop_test/` was not written to**: no file under it is newer than 23:00 on 2026-10-08.
17. **Disk state.** Start (23:07): root disk 45 GB free of 1.8 TB (98 % full), NAS 5.37 TB free. End (23:27): root 44 GB
    free, NAS 5.37 TB free. S7's folder is 3.0 MB locally; the 1 GB change on the root disk is not S7's. Root is still at
    98 %, as in S6 flag 23. Nothing was deleted.

## 5. Files written (`09_three_hypotheses/`, copied to `{runs}/drivejepa_retrain/09_three_hypotheses/`)

| file | sha256 |
|---|---|
| `PREDICTIONS.md` | `881eea0e4c73e690407b6626fe08beb0afb3421d3ca626f00480e1372e382b7b` |
| `predictions.sha256` | `f4e15f8a424b5a5afc8b85755ee3a8d09d517a609a46efa57cfe2fbee0409397` |
| `s7_common.py` | `f1cd7103d2c73858b00e2e730c68ed534c88cb6c4d90789bc9aedc049a4eebb0` |
| `s7_h1.py` | `5a278d4b86ab05ad36f3538d97e487d6ebd5225ebff44c4ac9503a105209814d` |
| `s7_h1_posthoc.py` | `abc665c479b95e4073f6b40f9601fe9e8f8b48dfb05cc07f4e778c21ace7ff26` |
| `s7_h2.py` (as run: `e99040b9…95568577`; one line number in its docstring was corrected afterwards, no code change) | `23536eba5898f453751a27c8145f23360e48859fd31fb5b08f893f949038132e` |
| `s7_h3.py` | `1cffd8034dea9765a0d25733eb96cc1b1a4f7c9fcedb1d0245cb1b8bdbfceb9f` |
| `run_s7_py.sh` | `af909bf96e26511ffc9ae6ca467695f3fa5a5567d18d13602e224b1a2193556d` |
| `run_s7_rest.sh` | `746303bfd1af1aefc2f0d317b2d2a12b433437f1811da444544bd2981dadb720` |
| `h1_instants.parquet` | `d1268e1d7e12744d812c31d17cf2692223433a6dab7beac206a4b25337cdc92d` |
| `h1_1_counts.csv` | `9ee82893703ef07964c222adab146bc70a6c9a9bfcce1650de26a3ce6f2121fa` |
| `h1_2_limiting_check.csv` | `d9bf1fc3421c5f99aa880e50b44196f048911d3f58577edf46bcb0b52d07a030` |
| `h1_3_speed.csv` | `36cf40c16d6367abc888e227c22c6430bacb66daf5398c7519efc56ddb644d51` |
| `h1_4_handover_decel.csv` | `c89dd3bd2355b87aeadd6b4a3888e54a1feb8d8c44fdce5fef3da91d1e1f7e6a` |
| `h1_prediction.csv` | `18108dd75ac0bcc460f1d599d11576d86b0d2430acbb122096be71f5a05d944a` |
| `h1_posthoc_summary.csv` | `90f6d7e5029d1001996e49da907e6c5f198ead0a9b6b7db8edf943089d26db33` |
| `h1_posthoc_step_profile.csv` | `b9e0e9b2ffc3f7e9159535571de17454ce74996960a2edd3f7c3db7766220dbb` |
| `h1.log`, `h1_attempt1_table_bug.log` (same content) | `c89a451f36966fbd36fb2fdbe47bdbbcbba05a8be92ac7abc4827124c6c5d387` |
| `h1_tables.log` | `7b882320cf07366856c70177654cbcd5809d02b563b6565735d5bba331d3a81c` |
| `h1_posthoc.log` | `eff787d56fab8b3e3d7e1b62c1c860046293ecfacb392d3860e3eb48405c1871` |
| `h2_exam_inspection_replans.csv` | `d165ac5b3f8f95dd0b6b39d9f3c4386651cc6f84f4e86db2d10cc76b2da3aaff` |
| `h2_exam_inspection_scores32.csv` | `3438e2f3e69b76366f309b984521849b578837a65e9e88c34f06f61b087fe062` |
| `h2_exam_inspection_summary.csv` | `f605a7ffe1233654eb53c75da084cfc42d441cd065514fdba6d8841138b67914` |
| `h2.log` | `8acc633d0f8d9e736472307be3c03ab374b2a6539b5706ee325fbda014c7d6f3` |
| `h3_selection.json` | `be76cbfb134a4548f8b42a9a2a3bde47a96dd6db21f49506852a30191244ce59` |
| `h3_training_runs.csv` | `c4397729fe49ed776cdb88b41593fcbce00a0108b8f0739b60e5312489c7b370` |
| `h3_models.pt` | `11c83e98bb56bb0b55db9f56f4957908cc511d8f51f72fd4ed1ea1e3e6c6fab6` |
| `h3_table.csv` | `009cf743e78d704bb70995b154d4ea7436a9f0d14f276a962ed9d238122d88f4` |
| `h3_top2_margin.csv` | `be118f914b87f192d919e626c24fba46f59e1bad57c64b1c11b1939b128b34a5` |
| `h3_unseen_per_instant.parquet` | `09fba3b00253e58468a5a56ee6e6c62614514446b0c98a31270a98f20ff058fb` |
| `h3.log` | `13da4d52b2804b693c5fbbbc1498ed5ac97b57789077a1219d03b2142e0c8e4f` |

Also `state_log.csv` (load, swap, RAM and disk at every start and end) and `files.sha256` (the list above). This report
is not in the list.

**CPU hours per part** (from `/usr/bin/time` in each log): H1 0.004 (8.5 CPU-s for the instants, 5.9 for the tables, 1.1
post hoc; 5.3 min of wall time, nearly all reading 6,790 small files from the NAS). H2 under 0.001 (0.8 CPU-s). H3 0.068
(245 CPU-s: 12 head trainings in 40 s, the rest the momentum stand-in and the bootstrap). **Total: about 0.07 CPU hour.**
The brief expected hours for H3; the heads are small and each training converged in 12–39 passes.

**GPU job the numbers make most informative: the generator retrain with filtered targets (H1)** — on 43.5 % of highway
instants no menu plan is good while a smooth catalogue curve is on 86 % of them, and H3 says a better lesson on the same
features reorders plans without lowering regret; the filter should be on even spacing (comfort), not on speed.
