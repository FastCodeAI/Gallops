# S8 — Three quick looks before any GPU work

**STATUS 2026-10-08 23:55 IST: L1, L2 and L3 are complete. CPU only, no Drive-JEPA forward pass, nothing written outside
`10_three_quick_looks/` (local and NAS).** Paths are relative to `paddock-sid/Alpasim/` unless they start with `/`.
`RD` = `analysis_3.0/Retrain-data`, `NAV` = `adapters/drivejepa/src/navsim_v2/navsim`,
`AGENT` = `NAV/agents/drive_jepa_perception_based`. All model numbers are pass 9 on its own menu, UNSEEN, from S6's dumps.

## 1. Numbers and verdicts

`PREDICTIONS.md`: sha256 `f78e796b522905226aca3eb6bea5836f16bf223b79b7fa890202e1cfbd6d044d`, hashed
2026-10-08T23:40:43+05:30 (`predictions.sha256`; copy on the NAS), before any S8 computation and before the conversion
code was opened. First run started 23:43:48 (`state_log.csv`).

| prediction | what it said | measured (95 % clip-bootstrap interval) | verdict |
|---|---|---|---|
| P-L1 | the swing is alternating (lag-1 autocorrelation < −0.5) AND catalogue curves go through a different conversion path than menu plans; if the swing is unstructured and the paths identical, the jaggedness is learned, not mechanical | lag-1 autocorrelation, median over the 1,443 no-good-plan highway instants: **−0.09 (−0.13 to −0.05)**; 8.7 % of plans are below −0.5. Unrelated values would give about −0.1. Conversion path: **the same code** for menu plans and catalogue curves from the 8 poses on (2.2) | **inconsistent on both halves.** This is the brief's other branch: learned, not mechanical. "Unstructured" needs one qualifier: the swing is not alternating, but it is not pure noise either. It is a fixed shape per menu slot that appears above about 20 m/s (2.3, post hoc) |
| P-L2 | for the ranking head, picks graded 0 are < 8 % of examples but contribute > 50 % of total regret; if < 25 %, regret is spread | picks graded 0: **9.5 % (8.1–11.2)**; their share of total regret: **64.8 % (60.4–68.5)** | **inconsistent as written, on the first half only** (9.5 % is not under 8 %; the interval's lower end is 8.1 %). The substance holds: regret is concentrated, not spread. Not the "< 25 %" branch |

L3 (no prediction; exam clips, inspection only): the scorer scores leaving as **failing** on d2715023 (best leave plan
0.09, median) and as **merely worse** on b44a4a44 (0.80 against 0.92 for staying). Two different situations.

## 2. L1 — what the jagged highway plans look like

### 2.1 Sample and plots (L1.1; `s8_l1.py`, seed 20261009)

20 UNSEEN highway instants with no menu plan graded ≥ 0.95 and 20 with one, drawn at random. For the best menu plan of
each, three versions and two references:

- **raw**: the 8 waypoints as the generator emits them (0.5 s apart, car frame);
- **calculator input**: the car's position put in front, then straight-line interpolation to 41 poses 0.1 s apart;
- **graded**: the 41 states the calculator's vehicle simulator produced from that input (what every check is computed on);
- the best catalogue curve (the 8 poses that were graded, and its 40 stored poses as a thin line) and the human path.

Per-instant table: `l1_sample_instants.csv` (grade, checks, swing, lag-1, the 8 step speeds, plot file). Every pose of every
version: `l1_sample_plans.csv` (5,792 rows). Plots index in section 5.

What the 20 no-good-plan plots show: comfort fails on 14 of 20; the 8 raw waypoints lie on a plausible road line but are
bunched and stretched along it. Example `plots/nogood_05_4cb05d65_14500.png` (car at 38 m/s): step speeds
33, 39, 41, **65**, **14**, 17, 26, 39 m/s. The simulator cannot follow that: its speed goes 38 → 45 → 27 → 36 m/s, and the
graded path ends away from the raw end point. The catalogue curve and the human are flat at 36–40 m/s. In the 20 good-plan
plots the raw steps stay within a few m/s and the simulated path sits on the waypoints.

### 2.2 Generator output to calculator input, step by step (L1.2)

| step | what happens | where |
|---|---|---|
| 1. What the generator emits | 32 plans × 8 poses (x, y, heading), 0.5 s apart to 4 s, in the car's frame | `AGENT/drive_jepa_config.py:27, 36, 62` |
| 2. How the 8 poses are produced | each of the 32 × 8 = 256 poses has its own learned starting feature, plus the ego-status encoding; one small network turns each feature into one (x, y, heading), independently of the other poses. Nothing ties pose k to pose k+1, and there is no speed or spacing output | `AGENT/drive_jepa_model.py:33, 61-62, 66, 69`; `AGENT/traj_refiner.py:24`; network `AGENT/bevformer/transformer_decoder.py:65-78` |
| 3. What the generator is trained against | mean absolute distance per pose to the human path, best of 32 plans, plus the same to up to 4 catalogue curves graded ≥ 0.95, plus a term pushing plans apart. Catalogue curves enter as 8 poses (`[:, 4::5]` of 40). No term on spacing or smoothness | `RD/05_small_overfit_check/train_patch/drive_jepa_agent_s35.py:111, 306-350` (human `:313`, catalogue `:331`, spread `:348`, `:252-261`) |
| 4. Menu plans handed to the calculator | the 32 × 8 poses as emitted, no change: training `:357`, `:179-199` → `get_sub_score_s35` (`:27`, call at `:52`); S6's dump the same call | `drive_jepa_agent_s35.py` as cited; `RD/08_scorer_diagnosis/s6_dump.py:109` |
| 5. Catalogue curves handed to the calculator | stored as 40 poses 0.1 s apart; **cut to 8 poses 0.5 s apart** (`[:, 4::5]`), then the same call | `RD/04_training_set/s3_grade.py:9`, `:63` |
| 6. Wrapper | `score(mc, poses)` takes `(N, 8, 3)` in the car's frame | `RD/04_training_set/s3_scorer.py:29-31` |
| 7. 8 poses → 41 poses | each plan is treated as 8 poses at 0.5 s (`NAV/common/dataclasses.py:259-265`); the car's rear-axle pose is put in front at t = 0; **straight-line interpolation** to 0, 0.1, …, 4.0 s | `AGENT/score_module/compute_navsim_score.py:18, 42`; `NAV/evaluate/pdm_score.py:86-95, 118, 124` |
| 8. Frame | car frame → map frame by the rear-axle position and heading, before the interpolation | `NAV/evaluate/pdm_score.py:101-115` |
| 9. Controller | the 41 poses are followed by a tracking controller driving a bicycle model of the car, 0.1 s steps; the resulting 41 states are what is graded | `compute_navsim_score.py:19, 49`; `NAV/planning/simulation/planner/pdm_planner/simulation_v2/pdm_simulator.py:29-30, 61-70` |

**Is the same path used for catalogue curves? Yes, from step 6 on it is the same code, same pose count, same spacing, same
interpolation, same controller, same frame.** The one difference is step 5 against step 4: a catalogue curve arrives as 40
stored poses and is thinned to every fifth; a menu plan is born as 8. That is a difference in where the 8 poses come from,
not in conversion, and by the rule fixed in `PREDICTIONS.md` it does not count. Two things follow from the table that are
worth knowing anyway:

- The catalogue's 0.1 s detail is thrown away before grading, and the catalogue curves are smooth by construction (cut from
  real driving), so their 8 poses are evenly spaced whatever the generator does.
- The straight-line interpolation in step 7 turns uneven spacing straight into speed jumps: a waypoint 5 m too far is
  10 m/s for half a second. The controller in step 9 then smooths it (2.3: an 18 m/s swing in the input becomes 7 m/s in the
  graded states), at the price of 7 m/s² and a path that no longer ends where the plan ends.

### 2.3 Is the swing a pattern? (L1.3, L1.4; `l1_structure.csv`, `l1_step_profile.csv`, `plots/structure_overview.png`)

Best menu plan of each instant, 8 step speeds. Lag-1 autocorrelation: about −0.9 = alternating, about −0.1 = unrelated
values, positive = one steady drift.

| measure | highway, no good plan (1,443) | highway, good plan (1,872) | ordinary, all (6,964) | ordinary, no good plan (1,840) |
|---|---|---|---|---|
| lag-1 autocorrelation, median | **−0.09 (−0.13 to −0.05)** | +0.10 (0.05–0.16) | +0.41 (0.39–0.43) | +0.39 (0.36–0.43) |
| plans with lag-1 below −0.5 | 8.7 % | 5.9 % | 2.0 % | 1.8 % |
| plans with lag-1 above 0 | 39.5 % | 58.7 % | 82.3 % | 83.0 % |
| consecutive speed changes that flip sign (alternating = 100 %, unrelated ≈ 67 %) | 62.5 % | 55.4 % | 47.6 % | 48.4 % |
| odd steps minus even steps, m/s: mean; median size | −1.2; 3.4 | 0.0; 0.6 | −0.1; 0.2 | −0.1; 0.3 |
| share of the swing in the first second (even = 25 %) | 14.9 % | 21.1 % | 33.1 % | 32.4 % |
| swing (fastest minus slowest step), median, m/s | 18.3 | 3.6 | 1.7 | 1.7 |
| swing as a share of the plan's mean speed, median | 0.65 | 0.20 | 0.27 | 0.30 |
| swing against car speed across instants (Spearman) | +0.73 | +0.53 | +0.11 | +0.14 |
| swing, median, car at 0–10 / 10–20 / 20–30 / over 30 m/s | 5.2 / 3.6 / 17.8 / 49.3 | 3.1 / 2.8 / 5.2 / 5.0 | 1.7 / 1.6 / — / — | 1.7 / 1.8 / — / — |
| plans left out (spread under 0.5 m/s) | 0 | 14 | 732 | 159 |
| same plan after the calculator's simulator: speed swing; largest acceleration, median | 6.9 m/s; 7.2 m/s² | 2.2 m/s; 1.6 m/s² | 1.4 m/s; 0.8 m/s² | 1.4 m/s; 0.9 m/s² |
| simulated end point to raw end point, median | 1.5 m | 0.7 m | 0.4 m | 0.4 m |
| reference, best catalogue curve: lag-1; swing | +0.63; 1.1 m/s | +0.63; 1.3 m/s | not computed | not computed |
| reference, human: lag-1; swing | +0.62; 0.6 m/s | +0.62; 1.0 m/s | +0.62; 1.3 m/s | +0.62; 1.3 m/s |

Answers to the three questions in the brief:

- **Odd/even alternating: no.** Lag-1 is −0.09, where unrelated values give −0.1 and alternation −0.9; sign flips 62.5 %
  against 67 % for unrelated values.
- **Growing with speed: yes, steeply, and with a threshold.** Under 20 m/s the swing is 4–5 m/s; at 20–30 m/s it is 18;
  above 30 m/s it is 49. But speed alone does not decide it: at the same car speed, instants that do have a good plan swing
  5 m/s.
- **Concentrated in the first second: no.** 15 % of the swing is in the first two steps (25 % would be even). The typical
  shape is in the middle: median step speed minus the plan's mean is −1.7, +1.5, +0.8, +1.2, −0.5, **−4.3**, **−3.1**, +0.1
  m/s over the 8 steps, with the fastest step at 2.0 s on 32 % of plans.

**Contrast (L1.4).** Ordinary-clip plans are smooth whether or not the menu has a good plan: lag-1 +0.41 (a steady drift,
like the human's +0.62), swing 1.7 m/s, 0.8 m/s² after simulation. So the jaggedness is a highway, high-speed thing, and on
ordinary clips a missing good plan has some other cause.

**Added after these tables were read (post hoc; `s8_l1_posthoc.py`, `l1_posthoc_slot_signature.csv`,
`l1_posthoc_slot_profiles.csv`).** The sample showed slot 8 repeating one shape (step 4 very fast, step 5 very slow). Over
all 32 plans of every highway instant:

| measure, all 32 plans | highway, car ≥ 20 m/s (1,850) | highway, car < 20 m/s (1,465) | ordinary, car ≥ 10 m/s (1,960) |
|---|---|---|---|
| swing as a share of mean speed, median | 0.89 | 0.24 | 0.18 |
| slots whose average shape swings by more than 30 % of mean speed | 29 of 32 | 5 of 32 | 5 of 32 |
| a single plan's shape against its own slot's average shape, correlation, median | 0.62 | 0.72 | 0.84 |
| the same against the all-slot average shape | 0.40 | 0.41 | 0.30 |

Above 20 m/s nearly every slot has its own fixed spacing error. Five slots (1, 4, 13, 14, 19) repeat theirs almost exactly
(correlation 0.96–0.99) with one step at 2.2–2.5 times the plan's mean speed. No slot passes comfort on more than 25 % of
these instants. The best plan on no-good-plan instants comes from few slots (29, 8, 20, 7 supply 46 %; the top 8 supply
65 %), and 29 and 20 are the two slots with the smallest shape. So "unstructured" is not quite right: the error is learned
per slot and switches on at highway speed.

## 3. L2 — where the regret lives (`s8_l2.py`; `l2_regret.csv`, `l2_gates.csv`, `l2_zero_picks_best_available.csv`)

UNSEEN, raw pick. Random = expected value over the 32 plans. Brackets: 95 % interval.

**Table 3.1 — all 11,842 instants.**

| measure | scorer | ranking head | random pick |
|---|---|---|---|
| regret, mean | 0.115 | 0.110 | 0.210 |
| regret = 0 | 38.3 % (35.6–41.3) | 47.3 % (44.6–50.1) | 33.2 % |
| regret in (0, 0.1] | 36.7 % (34.3–39.1) | 30.8 % (28.4–33.2) | 26.1 % |
| regret in (0.1, 0.5] | 17.7 % (16.1–19.3) | 13.5 % (12.2–15.0) | 23.7 % |
| regret > 0.5 | 7.3 % (6.4–8.4) | 8.4 % (7.1–9.8) | 16.9 % |
| 50th / 90th / 99th percentile | 0.02 / 0.29 / 1.00 | 0.00 / 0.36 / 1.00 | 0.05 / 0.97 / 1.00 |
| **picks graded exactly 0** | 8.1 % (6.9–9.5) | **9.5 % (8.1–11.2)** | 17.5 % (15.8–19.4) |
| **share of total regret from picks graded 0** | 51.6 % (47.3–55.9) | **64.8 % (60.4–68.5)** | 70.0 % (67.3–72.7) |
| share of total regret from regret in (0, 0.1] / (0.1, 0.5] / > 0.5 | 12.7 / 31.3 / 56.0 % | 8.5 / 23.8 / 67.7 % | 5.3 / 21.5 / 73.2 % |
| regret, mean, over instants where the pick is not graded 0 | 0.061 (0.056–0.066) | 0.043 (0.039–0.047) | 0.077 |

**Table 3.2 — by slice.** Picks graded 0; their share of total regret; regret = 0; regret > 0.5; mean regret without the
picks graded 0.

| pick | highway (3,315) | close-pass (1,563) | ordinary (6,964) |
|---|---|---|---|
| scorer | 7.9 %; 39.2 % (30.4–47.2); 11.9 %; 8.4 %; 0.098 | 13.2 %; 66.0 % (58.2–72.6); 43.5 %; 11.1 %; 0.059 | 7.1 %; 55.9 % (49.9–61.5); 49.7 %; 6.0 %; 0.043 |
| ranking head | 10.7 %; 61.2 % (51.5–68.6); 24.3 %; 9.7 %; 0.058 | 13.4 %; 68.2 % (61.0–74.7); 50.4 %; 11.6 %; 0.053 | 8.1 %; 66.1 % (60.7–70.9); 57.6 %; 7.1 %; 0.033 |
| random pick | 13.4 %; 52.9 %; 15.8 %; 14.5 %; 0.113 | 26.4 %; 80.1 %; 35.8 %; 23.4 %; 0.072 | 17.5 %; 75.5 %; 40.9 %; 16.6 %; 0.059 |

90th percentile of regret, scorer / ranking head: highway 0.42 / 0.43, close-pass 0.62 / 0.66, ordinary 0.19 / 0.18. The
99th percentile is 1.0 everywhere.

**Table 3.3 — which gate zeroed the picks graded 0.** Share of those picks; a pick can fail more than one gate. Every pick
graded 0 has at least one gate at 0; the traffic-light gate never is.

| pick, slice | picks graded 0 | drivable area | driving direction | collision | more than one gate |
|---|---|---|---|---|---|
| scorer, all | 962 | 59.1 % | 26.5 % | 26.3 % | 12.0 % |
| ranking head, all | 1,129 | 60.4 % | 35.5 % | 23.4 % | 19.0 % |
| ranking head, highway | 354 | 78.5 % | 47.5 % | 8.8 % | 34.5 % |
| ranking head, close-pass | 210 | 44.3 % | 27.1 % | 35.7 % | 6.2 % |
| ranking head, ordinary | 565 | 55.0 % | 31.2 % | 28.0 % | 14.2 % |
| scorer, highway | 262 | 78.6 % | 29.0 % | 14.1 % | 21.8 % |
| random pick, all | 2,072 (expected) | 56.1 % | 34.9 % | 36.8 % | 26.6 % |

**Table 3.4 — on instants where the pick is graded 0, what was the best available grade?**

| pick, slice | best available: mean; median | a plan ≥ 0.95 was on the menu | a plan ≥ 0.9 | best available is 0 too (no regret) | plans graded 0 on that menu, of 32 |
|---|---|---|---|---|---|
| scorer, all | 0.73; 0.86 | 45.8 % | 48.5 % | 19.1 % | 23.1 |
| ranking head, all | 0.75; 0.86 | 44.6 % | 47.8 % | 16.3 % | 23.2 |
| ranking head, highway | 0.77; 0.86 | 35.6 % | 39.8 % | 10.7 % | 21.9 |
| ranking head, close-pass | 0.74; 0.86 | 36.2 % | 36.7 % | 12.9 % | 23.7 |
| ranking head, ordinary | 0.74; 0.99 | 53.5 % | 57.0 % | 21.1 % | 23.8 |
| random pick, all | 0.84; 1.00 | 58.3 % | 60.6 % | 8.9 % | 18.8 |

Context (`l2_menu_context.csv`): 17.5 % of all menu plans are graded 0; 48.1 % of menus hold at least one such plan and
1.6 % hold nothing else. The scorer's and the ranking head's picks are both graded 0 on 6.3 % of instants; the two heads
pick the same slot on 12.6 %.

Reading. Regret is a few big losses, not many small ones. Two thirds of the ranking head's regret comes from the one
instant in ten where it picks a plan that fails a gate, usually leaving the drivable area, on menus where about 23 of the 32
plans fail one and a good plan sat next to them half the time. Take those instants out and the ranking head is clearly
ahead (0.043 against 0.061); it loses that lead by picking gate failures more often than the scorer does (9.5 % against
8.1 %). That is the mechanism behind S7's "better order, same regret": a pairwise lesson counts a 0.95-against-0.90
mistake and a 0.95-against-0 mistake almost alike. S6 found the gates readable from these same features (linear probe
0.88–0.89), so S7's "the features are the ceiling" is not supported by where the regret sits.

## 4. L3 — pull-out scores in absolute terms — EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION (`s8_l3.py`)

Scenes d2715023 and b44a4a44, 34 replans each, 0.5 s apart from handover. Raw score = the scorer's output before the
momentum step, on a 0–1 scale trained toward the grade. Leave / stay rule as S7 H2. Per replan:
`l3_exam_inspection_replans.csv`; summary `l3_exam_inspection_summary.csv`. The trial folder holds no shipped-model scores
for these replans; the last two columns are the shipped weights from the 0.1.4b baseline run of the same two scenes
(`analysis_3.0/Re-ranker/baseline_014b/menus/b014b_<scene>/`), **a different run, so not the same replans** (same scenes,
same replan times, car nearly stationary in both).

| exam clips, inspection only, not an evaluation | d2715023, 0.1.6 (trial) | b44a4a44, 0.1.6 (trial) | d2715023, shipped (0.1.4b run) | b44a4a44, shipped (0.1.4b run) |
|---|---|---|---|---|
| leave plans among the 32, median | 11 | 18.5 | 4.5 | 18 |
| best stay plan, raw score: median (range) | 0.945 (0.885–0.983) | 0.918 (0.860–0.940) | 0.909 (0.837–0.958) | 0.676 (0.648–0.800) |
| best leave plan, raw score: median (range) | **0.094** (0.020–0.388) | **0.801** (0.515–0.831) | 0.024 (0.010–0.113) | 0.333 (0.156–0.491) |
| best leave, first 5 s / after 5 s, median | 0.139 / 0.070 | 0.820 / 0.796 | 0.056 / 0.016 | 0.384 / 0.285 |
| replans with best leave < 0.2 (scored as failing) | 31 | 0 | 34 | 1 |
| replans with best leave 0.2–0.6 | 3 (the first three) | 1 | 0 | 33 |
| replans with best leave ≥ 0.6 (merely worse) | 0 | 33 | 0 | 0 |
| median score over all leave plans | 0.051 | 0.694 | 0.019 | 0.209 |
| leave plans scored ≥ 0.6, median count | 0 | 16 | 0 | 0 |
| best leave plan ends, median: forward; sideways | 6.1 m; 1.9 m | 5.5 m; 0.6 m | 7.7 m; 3.0 m | 5.8 m; 0.9 m |
| gap, best stay minus best leave, median | 0.841 | 0.115 | 0.883 | 0.344 |

Answer. On **d2715023** the scorer scores leaving as unsafe: the best leave plan is under 0.2 at 31 of 34 replans and every
leave plan sits near 0.05, the level of a failed gate. On **b44a4a44** it scores leaving as merely worse: 0.80 for the best
leave plan against 0.92 for staying, with 16 leave plans above 0.6. The retrain moved b44a4a44 from "in between" (0.33
under the shipped weights) to "merely worse"; it did not move d2715023. Whether the scorer is right that leaving fails a
gate on d2715023 cannot be told from scores: there are no grades for these clips.

## 5. Plots index (`10_three_quick_looks/plots/`, 45 files; hashes in `plots.sha256`)

- `structure_overview.png` — lag-1 histogram, step-speed shape, swing against car speed.
- `sheet_nogood_speed.png`, `sheet_nogood_topdown.png` — the 20 no-good-plan instants on one sheet each (speed per step; top-down).
- `sheet_good_speed.png`, `sheet_good_topdown.png` — the 20 good-plan instants.
- `nogood_01_…png` to `nogood_20_…png`, `good_01_…png` to `good_20_…png` — one instant each: left, top-down (raw waypoints
  as red circles on the calculator-input line and the graded curve; blue ticks = where the simulated car is at each 0.5 s);
  right, speed per step. File names carry the instant id; `l1_sample_instants.csv` lists them with their numbers.
- Top-down plots stretch the sideways axis (a few metres against 100+ m forward); angles are not to scale.

Worth opening first: `nogood_05_4cb05d65_14500.png`, `nogood_20_f650ef79_07500.png` (one step at 142 m/s),
`nogood_19_f5e30435_11500.png` (first step at 67 m/s, the rest even), then `sheet_good_speed.png` for the contrast.

## 6. Flags

1. **The two conversion paths do not differ.** Menu plans and catalogue curves are graded through the same code from 8
   poses on (2.2). The only difference is upstream: catalogue curves are 40 stored poses thinned to 8
   (`RD/04_training_set/s3_grade.py:9`), so their stored 0.1 s detail is never graded.
2. **What is graded is not what the generator emits.** The calculator joins the 8 waypoints with straight lines and lets a
   controller follow them. On no-good-plan highway instants the graded path ends a median 1.5 m from the plan's own end
   point and asks 7.2 m/s². This is the calculator working as designed (standard for this metric), not a bug.
3. **P-L1's path half was not blind**: S6 2.3 had already described the 8 → 40 interpolation and the catalogue cut. Stated
   in `PREDICTIONS.md`.
4. **Lag-1 autocorrelation over 8 values is a coarse measure.** −0.09 cannot be told apart from unrelated values (−0.1);
   that is the finding, but it also means it cannot see a fixed non-alternating shape. The per-slot table in 2.3 does, and
   it was added after the pre-registered tables were read.
5. **L1.2 covers grading only.** How the car's adapter hands the chosen plan to its controller was not examined.
6. **No catalogue reference on ordinary clips** (L1.4): the best catalogue curve was only available for highway instants
   from S7; the human path is the reference there.
7. **L1's first run stopped on its last figure** (a plotting bug of mine, `l1_attempt1_overview_figure_bug.log`); every
   table and the 44 other plots had been written. L2 and L3 then ran. L1 was rerun whole after the one-line fix and its
   tables are byte-identical to the first run (three files checked by hash).
8. **The ranking head picks plans graded 0 more often than the scorer** (9.5 % against 8.1 %; highway 10.7 % against
   7.9 %). S7 did not report this.
9. **16–19 % of picks graded 0 are on menus where the best plan is graded 0 too**; they cost no regret. 1.6 % of menus are
   all graded 0.
10. **Random-pick percentiles** are over all 32 plans of all instants with equal weight, not over simulated draws.
11. **L3's shipped-weights columns are from a different run and a different folder** than the brief named
    (`Re-ranker/baseline_014b/`, read-only). Compare levels, not replans. The trial's own report quotes the same run.
12. **L3 uses S7's leave rule**, which counts plans creeping 2–5 m forward as stay (S7 flag 4). With S6's rule (and the
    nudged score) the best pull-out plan on d2715023 scored 0.38 (median) rather than 0.09: the plans that score at all there
    are the short ones.
13. **Start gate**: load < 32, free swap > 15 GB and available RAM > 40 GB all held at every start (`state_log.csv`).
14. **Disk state.** Start (23:40): root 44 GB free of 1.8 TB (98 % full), NAS 5.37 TB free. End (23:48): the same. S8's
    folder is 9.5 MB. Nothing was deleted. `07_closed_loop_test/`, `Re-ranker/baseline_014b/`, `08_scorer_diagnosis/` and `09_three_hypotheses/`
    have no file newer than S8's start.

## 7. Files written (`RD/10_three_quick_looks/`, copied to `{runs}/drivejepa_retrain/10_three_quick_looks/`)

| file | sha256 |
|---|---|
| `PREDICTIONS.md` | `f78e796b522905226aca3eb6bea5836f16bf223b79b7fa890202e1cfbd6d044d` |
| `predictions.sha256` | `65ee07aa3096f6a26d4512543ec8411cd790b5f5ee05d3dea44258bed43b991a` |
| `s8_l1.py` | `654675f28c6592b0fab1af18331a28bc3a0410e9e82c55dfefe71bc1d493064a` |
| `s8_l1_posthoc.py` | `bd0d2f84588b8048cdb1941d60cd3d37dcbe5d3ee2a7df4394872937844e3755` |
| `s8_l2.py` | `e85b2f42ec2a092184d89b5d1036c851c0822c87c5d23fd2d610bf0f1d979001` |
| `s8_l3.py` | `2db991fb9e06fedfb1825087dc85c17ddf59ab83439c7922318748a34b8ca4af` |
| `run_s8_py.sh` | `18c37487447e97d25679d11865d64281b7748f127817733e2e2df1ac5b3a8303` |
| `run_s8_all.sh` | `6b2915b7481474ccc5f92fd2898f11f006dbbac77767f7dac42b0b90b7c5842c` |
| `l1_structure_per_instant.parquet` | `1d7a116299c0ace43b8fe2b2722501acec4c6451c1d6201a5e1f1afa6866f462` |
| `l1_structure.csv` | `50156633a988ad87ab19d5a3bf14123e9e9ff8eedf694d8267b61feb7af16276` |
| `l1_step_profile.csv` | `1bf94fb0802dba354da7ef0721ea3a7cc374f56289b29b580ad40a3f3b6ed505` |
| `l1_sample_instants.csv` | `48775f857fd177467e6689d1add8b1412adb64930fc54bd0548bd50c766f4295` |
| `l1_sample_plans.csv` | `4f303f81b59c6df6559927d26812fad41730771d5d6c1c8fea00628b27d41224` |
| `l1_posthoc_slot_signature.csv` | `2c88bdd94ca420cc5a3c94a25db02427f28cfa9e1eeff9fe423c112b0f234aa7` |
| `l1_posthoc_slot_profiles.csv` | `b55e6ece8aebe8b4815aac3bc696989857459d15caf6ddd512e4c82d5f83fff6` |
| `l1.log` | `ca244f30c4f42c207d0195652644e94f2e73bdeac8cd49656dd32a66179b5d6c` |
| `l1_attempt1_overview_figure_bug.log` | `4eafbd917a82d52e9aca44b9cdf31b90743b5e99b8b583d38043e43c993fe0da` |
| `l1_posthoc.log` | `fa5bb2083f564eaa87a6357f01ee2b82903c121996b15dada2eae7478b5ef501` |
| `l2_regret.csv` | `219909419df317949f3abcdecc9b29b953f1ecf20596967f1443a4c43d76dc35` |
| `l2_gates.csv` | `d6ffca8a155406c4ae101be3c0ce9b7dc2b1069f8ca9aeba5f7dff0672b497a4` |
| `l2_zero_picks_best_available.csv` | `edeb8f36b25e7b4cdfff5db279800e45097d73f7a57b2309acedd5e527a447ab` |
| `l2_menu_context.csv` | `1c6318e7193ecf593707b211aa50ed865b2b0652cb1d0e6e14c6875e0cf5274d` |
| `l2_per_instant.parquet` | `8d75cd6e5faa625c6bb887b2dc1cb12a87f24cf3fafece3aa0cd46d2c40b050b` |
| `l2.log` | `8dd7ea996aeacb7eead87a45be5452be959895f991ab7bae09a739be1487931f` |
| `l3_exam_inspection_replans.csv` | `8ca2d87374950b2d1fec1af019528bfbe85b696de5a63f4d28f247161a5579e5` |
| `l3_exam_inspection_summary.csv` | `447440bab8b3f1fc7c658bd51d9d0e35e1dc7dd5a92d7935767926807bd0b09c` |
| `l3.log` | `908a583672af5a14e859c30993e31c1468cb87d38c87318d0cfd0123fd7f93ef` |
| `plots.sha256` (the 45 plot files) | `e7485f685f520701e09f791f4867efa80fddad27d79f1d493f571db84b8dd349` |

Also `state_log.csv` and `files.sha256`. This report is not in the list.

**CPU**: L1 38 CPU-s (two runs of 19 s), post hoc under 1 s, L2 9 s, L3 under 1 s. **Total about 0.014 CPU hour.**

**Which fix L1 points to: generator retrain, not resampling** — the conversion is the same code for menu plans and catalogue
curves and the uneven spacing is already in the 8 raw waypoints, as a learned per-slot shape that appears above about
20 m/s; the generator's loss has no term on spacing and each waypoint is decoded on its own.
