# S9 — Two CPU builds: the two-stage scorer head (B) and pull-out partial grades (C); spec for A

**STATUS 2026-10-09 02:20 IST: B, C (C.1–C.4) and the A spec are complete. CPU only; the GPU was not used. Nothing was
written outside `11_scorer_head_and_pullout_grades/` (local and NAS).** Paths are relative to `paddock-sid/Alpasim/` unless they start with `/`.
`RD` = `analysis_3.0/Retrain-data`, `SM` = `adapters/drivejepa/src/navsim_v2/navsim/agents/drive_jepa_perception_based/score_module`.
All model numbers are pass 9 on its own menu.

## 1. Summary

- **B does not deliver.** On UNSEEN the two-stage head picks a plan graded 0 on 8.9 % of instants (7.6–10.5) with regret
  0.110 (0.100–0.121): the same as S7's ranking head (9.5 %, 0.110) and no better than the scorer (8.1 %, 0.115). On the
  SEEN held-out clips it looked like a fix (4.1 %, 0.075). The gate head reads each gate at 0.89–0.91 on UNSEEN, about
  S6's probes, but at the threshold chosen on SEEN it vetoes only 44 % of the plans that fail a gate.
- **The numbers that looked good were measured on clips the model trained on.** Every head (the scorer too) has regret
  0.072–0.080 on SEEN held-out clips and 0.102–0.115 on UNSEEN. The threshold curve on SEEN is flat (0.075–0.081 over the
  whole range), so the veto was never doing much; the seen-to-unseen gap is the whole story.
- **C: map-free checks run on all 283 pull-out clips** (9,053 examples, 32 plans each). Only comfort is map-free in the
  code; collision and time buffer read three map facts, set by hand here in two ways. A leaving plan passes collision on
  72.9 % (69.8–75.8) of pull-out menus. On the menus that matter, with the car standing still, it is 48.1 % (42.7–53.4).
- **C.4: adding the pull-out collision labels made the gate head more sure, not less, that leaving d2715023 collides**:
  gate P of the best leave plan 0.98 (median) against 0.89 before; under 0.5 at 0 of 34 replans. On b44a4a44 leave plans
  mostly pass the gate (P 0.29) and the two-stage head still picks a stay plan at 34 of 34 replans, because its ranker
  prefers staying. (Exam clips, inspection only.)
- **A is specified, not run** (`A_spec.md`). The brief's "table above" with the success numbers did not reach me; the
  numbers in the spec are my proposal.

## 2. Predictions

`PREDICTIONS.md`: sha256 `31d429334bde6fd8787b9aebf4bba3940d155b0ae11c44341d6516557a21d66b`, hashed
2026-10-09T00:04:16+05:30 (`predictions.sha256`; copy on the NAS). Before hashing I read code and listed files; nothing was
computed. First run 00:04 (`state_log.csv`).

| prediction | what it said | measured (95 % clip-bootstrap interval) | verdict |
|---|---|---|---|
| P-B | two-stage head on UNSEEN: picks graded 0 < 4 % and regret < 0.08 | picks graded 0: 8.9 % (7.6–10.5); regret 0.110 (0.100–0.121) | **inconsistent on both halves** |
| P-C, first half | a leaving plan passes collision on > 60 % of pull-out menus | 72.9 % (69.8–75.8) of all 9,053 menus (strict setting; lenient 74.5 %) | holds as written. With the car under 0.5 m/s: 48.1 % (42.7–53.4) |
| P-C, second half | after C.4, gate P of the best leave plan on d2715023 below 0.5 at a majority of replans | 0 of 34 replans (before C.4: 2 of 34); median P 0.978 (before: 0.895) | **fails; it moved the other way** |
| P-C | both halves | | **inconsistent** |

## 3. Part B — two-stage head on the frozen 256 features (`s9_b.py`, `s9_lib.py`)

Set-up as pre-registered. SEEN split = S7's (9,476 training instants / 582 clips, 2,366 validation instants / 145 clips;
scaling checked equal to S7's). Gate head 256 → 64 → 3 (kerb = drivable area, oncoming = driving direction, collision;
"fails" = stored gate below 1), weight decay 0.001 by validation loss; P = 1 − Π(1 − p). Ranker = S7's head with each pair's
loss multiplied by |grade gap| (weighted sum over weighted count) and only pairs with both plans at P ≤ t. Selection: veto
P > t, rank the rest, lowest P if all are vetoed. Choices written to `b_selection.json` at 00:08:27, UNSEEN opened 00:08:41.
Check: baselines reproduce S7/S8 (scorer 0.1149 and 8.12 %; S7 head 0.1101 and 9.53 %).

**Table 3.1 — gate head, AUROC, pooled over plans (B.2; `b_gate_auroc.csv`).**

| gate | plans failing, UNSEEN | gate head, SEEN held-out clips | gate head, UNSEEN | S6 linear probe, UNSEEN (table 4.7) |
|---|---|---|---|---|
| kerb (drivable area) | 9.8 % | 0.942 (0.926–0.955) | 0.893 (0.869–0.916) | 0.88 (0.86–0.91) |
| oncoming (driving direction) | 9.4 % | 0.946 (0.933–0.960) | 0.910 (0.890–0.928) | 0.89 (0.86–0.91) |
| collision | 6.4 % | 0.955 (0.937–0.967) | 0.888 (0.867–0.908) | 0.88 (0.85–0.90) |
| any of the three (P) | 19.3 % | 0.936 (0.926–0.946) | 0.861 (0.845–0.874) | — |
| any of the three, by the scorer's own output | 19.3 % | — | 0.833 (0.817–0.848) | — |

The small network adds 0.01–0.02 over S6's linear probe. Jointly it is 0.03 above what the scorer's own output already does.

**Table 3.2 — regret against t on the SEEN held-out clips (B.4; `b_t_curve.csv`).**

| t | regret | picks graded 0 | plans vetoed | instants with every plan vetoed | best plan vetoed | veto + S7's ranker, regret |
|---|---|---|---|---|---|---|
| 0.05 | 0.075 | 3.1 % | 46.0 % | 18.0 % | 48.1 % | 0.076 |
| 0.1 | 0.075 | 3.7 % | 34.9 % | 10.0 % | 34.2 % | 0.074 |
| 0.2 | 0.079 | 4.5 % | 26.0 % | 4.7 % | 23.6 % | 0.077 |
| 0.3 | 0.081 | 4.4 % | 21.4 % | 2.3 % | 17.0 % | 0.077 |
| 0.4 | 0.079 | 4.5 % | 18.1 % | 1.5 % | 13.4 % | 0.076 |
| 0.5 | 0.079 | 4.1 % | 15.4 % | 1.0 % | 9.8 % | 0.080 |
| **0.6** | **0.075** | 4.1 % | 13.0 % | 0.5 % | 8.0 % | 0.079 |
| 0.7 | 0.077 | 4.3 % | 10.7 % | 0.3 % | 6.0 % | 0.080 |
| 0.8 | 0.077 | 4.2 % | 8.2 % | 0.2 % | 3.8 % | 0.081 |
| 0.9 | 0.076 | 3.9 % | 5.3 % | 0.0 % | 2.1 % | 0.079 |
| 1.0 (no veto) | 0.077 | 4.1 % | 0 | 0 | 0 | 0.079 |

Chosen: t = 0.6 (lowest regret; 0.0751 against 0.0751 at 0.1 and 0.0754 at 0.05). The curve is flat: the spread over t
(0.075–0.081) is about the size of the run-to-run difference between rankers, and no veto at all gives 0.077. For
reference on the same clips: scorer 0.072, S7 head 0.079, gate head alone 0.123.

**Table 3.3 — UNSEEN, all 11,842 instants, 401 clips, at t = 0.6 (B.5; `b_table.csv`).**

| head, raw pick | regret, mean | regret, 90th pct | picks graded 0 | zeroed by kerb / oncoming / collision | true best | pick ≥ 0.9 | rank agreement, median |
|---|---|---|---|---|---|---|---|
| **two-stage head** | **0.110 (0.100–0.121)** | 0.30 | **8.9 % (7.6–10.5)** | 59 / 33 / 25 % | 41.3 % | 51.0 % | 0.49 |
| scorer's own output | 0.115 (0.105–0.125) | 0.29 | 8.1 % (6.9–9.5) | 59 / 27 / 26 % | 38.3 % | 49.5 % | 0.39 |
| S7 ranking head | 0.110 (0.099–0.123) | 0.36 | 9.5 % (8.1–11.2) | 60 / 36 / 23 % | 47.3 % | 51.3 % | 0.57 |
| veto (t = 0.6) + S7's ranker | 0.110 (0.098–0.122) | 0.36 | 9.4 % (7.9–11.0) | 60 / 34 / 24 % | 47.2 % | 51.1 % | 0.57 |
| weighted ranker, no veto | 0.109 (0.099–0.120) | 0.25 | 8.6 % (7.2–10.2) | 60 / 31 / 25 % | 39.7 % | 50.7 % | 0.50 |
| gate head alone (lowest P) | 0.136 (0.127–0.144) | 0.28 | 5.1 % (4.1–6.2) | 58 / 30 / 25 % | 19.7 % | 36.2 % | — |
| random pick (expected) | 0.210 | — | 17.5 % | | 33.2 % | 38.6 % | — |

Two-stage head minus S7's head, paired: regret −0.000 (−0.004 to +0.004).

**Table 3.4 — two-stage head by slice, and with the momentum nudge.** Regret; picks graded 0; true best; pick ≥ 0.9.

| | highway (3,315) | close-pass (1,563) | ordinary (6,964) |
|---|---|---|---|
| two-stage, raw | 0.132 (0.110–0.157); 9.6 %; 20.3 %; 37.8 % | 0.151 (0.122–0.185); 13.9 %; 43.7 %; 35.7 % | 0.090 (0.078–0.103); 7.5 %; 50.8 %; 60.8 % |
| two-stage, with the nudge | 0.135; 10.0 %; 19.9 %; 37.9 % | 0.154; 14.1 %; 43.3 %; 35.0 % | 0.090; 7.5 %; 49.8 %; 60.6 % |
| scorer, raw | 0.149; 7.9 %; 11.9 %; 35.6 % | 0.150; 13.2 %; 43.5 %; 33.5 % | 0.091; 7.1 %; 49.7 %; 59.7 % |
| S7 head, raw | 0.134; 10.7 %; 24.3 %; 38.1 % | 0.146; 13.4 %; 50.4 %; 35.1 % | 0.091; 8.1 %; 57.6 %; 61.1 % |

With the nudge, all instants: two-stage 0.111 and 9.1 % (pick changed on 30.7 %); scorer 0.113 and 8.1 %; S7 head 0.113 and
9.9 %. Gates that zeroed the two-stage picks: highway 80 / 44 / 9 %, close-pass 45 / 24 / 39 %, ordinary 52 / 30 / 30 %.

**Table 3.5 — what the veto does on UNSEEN at t = 0.6 (`b_veto_stats.csv`).**

| | all | highway | close-pass | ordinary |
|---|---|---|---|---|
| plans failing a gate | 19.3 % | 14.1 % | 30.5 % | 19.3 % |
| plans vetoed | 11.6 % | 6.8 % | 17.0 % | 12.7 % |
| failing plans vetoed | 44.0 % | 33.7 % | 42.7 % | 48.1 % |
| passing plans vetoed | 3.9 % | 2.4 % | 5.7 % | 4.3 % |
| plans graded 0 that are vetoed | 46.3 % | 34.5 % | 46.2 % | 50.6 % |
| of the plans left in, graded 0 | 10.6 % | 9.4 % | 17.1 % | 9.9 % |
| instants with the best plan vetoed | 9.2 % | 6.6 % | 11.2 % | 10.0 % |
| best grade left after the veto / on the menu, mean | 0.916 / 0.928 | 0.896 / 0.912 | 0.877 / 0.887 | 0.934 / 0.944 |

Reading.
- The veto halves the plans graded 0 (17.5 % → 10.6 % of what the ranker may pick from) and costs 0.012 of best grade. The
  ranker then picks from what is left about as badly as before: 8.9 % against 10.6 % available.
- The "gate head alone" row is the floor for this gate head: even the plan it trusts most is graded 0 on 5.1 % of
  instants (1.6 % of menus are all graded 0). So "< 4 %" was out of reach for any threshold with this head.
- The grade-gap weighting does what it should on the tail (90th-percentile regret 0.25 against 0.36 for S7's head) and
  gives back S7's gain in true-best rate (39.7 % against 47.3 %).
- Everything that separates the heads on SEEN held-out clips disappears on UNSEEN. Choosing t, stopping rules and weight
  decay on SEEN held-out clips, as the brief asked, measures the wrong thing: the features of those clips were trained on.

## 4. Part C — partial grades on the raw pull-out clips

### 4.1 Which checks run with boxes and egomotion and no map (C.1)

`RD/04_training_set/s3_scorer.py` has no map-free mode: `score()` passes the map objects straight into the vendored calculator
(`:31-32`). What each check reads is in the vendored code:

| check | needs the map? | where |
|---|---|---|
| comfort | **no.** Simulated states and the past human path only | `SM/train_pdm_scorer.py:769-822` |
| collision | **partly.** Overlap with boxes needs no map (`:419-433`). Whose fault it is does: a side contact with a moving actor counts only if the car is "in more than one lane" or "off the drivable area" at that step (`:435-440`, `:455-462`). Contacts with a stopped actor, or with an actor in front, count regardless (`:451-454`, `:460`) | `SM/train_pdm_scorer.py:404-482` |
| time buffer | **partly.** The projection and overlap need no map (`:581-633`). An actor ahead always counts; an actor beside counts only if the car is in more than one lane, off the drivable area, or in an intersection (`:647-673`; the intersection question is asked of the map object at `:668-670`) | `SM/train_pdm_scorer.py:566-684` |
| the three map facts | computed from map polygons in one place | `SM/train_pdm_scorer.py:333-402` |

So of the three checks in the brief, comfort runs map-free as is; collision and time buffer run once the three facts are
supplied. They were supplied two ways (both stored): **strict** = the car is in more than one lane at every step (side
contacts are its fault; primary, because a car leaving the kerb is outside a lane); **lenient** = none of the three. The
calculator's own methods were called in their own order (`:185-207`) without the map step; nothing vendored was edited
(`s9_c_dump.py`).

### 4.2 Timing, projection, run (C.2)

| | examples | wall | per example | CPU |
|---|---|---|---|---|
| timing run, 10 clips | 320 | 244 s | 0.76 s | 0.36 CPU h (4.07 CPU-s per example) |
| **projection for 283 clips, written 00:16:47 before the launch** (`c_projection.md`) | 9,053 | about 1.9 h | 0.76 s | about 10 CPU h |
| full run, 00:16–02:06, under the gate | 9,053 | 1.82 h | 0.72 s | 9.74 CPU h (3.87 CPU-s per example) |

6 threads + 2 checking workers + 2 loaders; own memory peak 7.9 GB; no pause (5-minute load never above 17.2, available
RAM never below 62 GB). The split forward equals the model's own forward exactly (0.0 on plans and scores, same pick).
The manifest has 283 pull-out clips, not the brief's 285 (S0 counted 285 in the raw set; 283 are in the training manifest).

### 4.3 Is the map-free scene right? (`c_alignment.csv`)

| check | value |
|---|---|
| human path rebuilt from egomotion against the stored training target, largest difference per example | median 0.4 mm, 99th pct 33 mm, max 113 mm |
| examples with the real vehicle box / with 1.5 s of past / with a full 4 s future | 100 % / 100 % / 100 % |
| boxes at the anchor, median | 33 |
| **the human path itself passes collision** | **86.7 % strict, 88.3 % lenient** |
| the human path passes time buffer / comfort | 78.9 % / 95.7 % |

The frames and clocks line up. But the human drove these clips without hitting anything, and the calculator says the
human path collides on 13 %. That is the noise floor of this collision label (box size and position errors next to parked
cars, and the calculator's controller not tracking the path exactly). It is a flag on everything below.

### 4.4 Output (C.3)

Per example and plan: `nc_strict`, `ttc_strict`, `nc_lenient`, `ttc_lenient`, `hc`, and
**map-free grade = collision gate × (5 × time buffer + 2 × comfort) / 7** (`mapfree_grade_strict`, `_lenient`), with the
32 plans, the 256 features and the scorer outputs. On the NAS: `{runs}/drivejepa_retrain/11_scorer_head_and_pullout_grades/pullout/`
`pullout_all.npz` (sha256 `5e113b3e…125a0`), `pullout_manifest.csv` (one row per example: clip, time, shard, row; sha256
`93751388…9169b`), `dump/shard_0000…0035.npz`, `files.sha256` (also `c_pullout_files.sha256` locally). 382 MB.

**Table 4.1 — per plan (`c_per_plan.csv`).**

| | all menus (9,053) | car under 0.5 m/s (3,483) | car at 0.5 m/s or more (5,570) |
|---|---|---|---|
| plans that leave (S7's rule) | 74.9 % | 51.2 % | 89.8 % |
| plans passing collision, strict / lenient | 67.9 / 69.5 % | 49.5 / 49.7 % | 79.5 / 81.9 % |
| leave plans passing collision, strict | 66.3 % | 29.9 % | 79.3 % |
| stay plans passing collision, strict | 72.9 % | 70.0 % | 81.2 % |
| plans passing time buffer, strict | 61.6 % | 41.0 % | 74.5 % |
| plans passing comfort | 86.5 % | 87.8 % | 85.6 % |
| map-free grade, mean, strict | 0.605 | 0.414 | 0.725 |

**Table 4.2 — share of pull-out menus on which a plan leaves and passes collision (`c_per_menu_summary.csv`).**

| subset | menus | strict | lenient | … and has map-free grade 1 | by clip: median; clips at 0 %; clips under 60 % |
|---|---|---|---|---|---|
| **all menus (the prediction)** | 9,053 | **72.9 % (69.8–75.8)** | 74.5 % (71.6–77.2) | 60.8 % | 78 %; 1.8 %; 36 % |
| car under 0.5 m/s | 3,483 | 48.1 % (42.7–53.4) | 48.5 % | 32.1 % | 60 %; 30 %; 49 % |
| before the clip's first move-off | 4,009 | 54.1 % (48.3–59.2) | 54.5 % | 40.2 % | 60 %; 30 %; 49 % |
| car under 0.5 m/s and the human leaves within 4 s | 596 | 66.6 % (61.0–72.1) | 66.8 % | 40.6 % | 100 %; 35 %; 38 % |
| car at 0.5 m/s or more | 5,570 | 88.4 % (86.2–90.5) | 90.8 % | 78.8 % | 94 %; 2.5 %; 5.3 % |

Per clip: `c_by_clip.csv`. Other facts on the same menus (`c_per_menu.parquet`): with the car standing still the pass-9
scorer's pick is a leave plan on 11.7 % of menus (14.4 % when the human is about to leave), and the human leaves within
4 s on 17.1 %; a leave plan exists on every menu.

Reading. The 72.9 % holds the prediction as written, but it is carried by menus where the car is already moving and
nearly every plan "leaves". Standing still, a collision-free leave plan is on the menu about half the time, and on 30 % of
clips never; even when the human is about to pull out it is two thirds. Part of the shortfall is the label's noise floor
(4.3), part is real: standing still, 70 % of leave plans touch a box against 30 % of stay plans.

### 4.5 C.4 — gate head retrained with the pull-out collision labels (`s9_b.py --c4`)

Pull-out examples added to the gate head's training: 7,230 examples (226 clips) to train, 1,823 (57 clips) to validation;
collision output only (strict label; 32.1 % of pull-out plans fail it), the other two outputs masked. Ranker retrained
with the same recipe on the new P; t kept at 0.6.

| | B | C.4 |
|---|---|---|
| gate AUROC on UNSEEN: kerb / oncoming / collision / any | 0.893 / 0.910 / 0.888 / 0.861 | 0.900 / 0.913 / 0.892 / 0.870 |
| collision AUROC on held-out pull-out clips (map-free label) | — | 0.910 (0.879–0.933) |
| gate P on UNSEEN plans, C.4 against B | | correlation 0.96; veto decision changed on 3.1 % of plans |
| failing plans vetoed / passing plans vetoed, UNSEEN | 44.0 % / 3.9 % | 47.9 % / 4.3 % |
| **two-stage head, UNSEEN: regret** | 0.110 (0.100–0.121) | 0.102 (0.093–0.112) |
| **picks graded 0** | 8.9 % (7.6–10.5) | 7.3 % (6.1–8.9) |
| regret 90th pct; true best; pick ≥ 0.9; rank agreement | 0.30; 41.3 %; 51.0 %; 0.49 | 0.23; 36.7 %; 50.0 %; 0.44 |
| by slice, regret: highway / close-pass / ordinary | 0.132 / 0.151 / 0.090 | 0.135 / 0.121 / 0.082 |
| by slice, picks graded 0 | 9.6 / 13.9 / 7.5 % | 9.2 / 9.8 / 5.9 % |
| with the momentum nudge: regret; picks graded 0 | 0.111; 9.1 % | 0.104; 7.6 % |
| C.4 gate head with **B's** ranker: regret; picks graded 0 | | 0.109 (0.099–0.120); 8.8 % |

C.4's two-stage head is the best row of S9 on UNSEEN (0.102, 7.3 %) and still misses both of P-B's lines. The gain is not
the pull-out labels: the new gate head with B's ranker gives 0.109 and 8.8 %, the same as B. The difference comes from the
retrained ranker (same recipe, different pair filter, different stopping pass), which is one draw; treat 0.102 against
0.110 as within the spread of this recipe, not as an effect (flag 6).

### 4.6 The two stuck trial scenes — EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION (`s9_trial_gate.py`)

d2715023 and b44a4a44, adapter 0.1.6, 34 replans each. The 256 features are not in the trial folder; they were recomputed
on CPU from the trial's logged frames with the same weights (read-only replay, method of `07_closed_loop_test/s5_menus.py`): picks
match the trial at 78 of 78 replans, plans to 0.0 m in the replay and within 0.9 mm in the feature pass. Best leave plan =
the leave plan with the highest raw scorer score (S7 H2). t = 0.6. Per replan: `c4_exam_inspection_replans.csv`; all 32
plans: `c4_exam_inspection_plans32.csv`.

| exam clips, inspection only, not an evaluation | d2715023, B head | d2715023, C.4 head | b44a4a44, B head | b44a4a44, C.4 head |
|---|---|---|---|---|
| gate P of the best leave plan, median (range) | 0.895 (0.43–0.98) | **0.978 (0.91–0.99)** | 0.159 (0.06–0.52) | 0.291 (0.12–0.72) |
| replans with that P below 0.5 | 2 of 34 | **0 of 34** | 33 of 34 | 29 of 34 |
| its three outputs, median: kerb / oncoming / collision | 0.14 / 0.01 / 0.88 | 0.14 / 0.01 / 0.97 | 0.04 / 0.00 / 0.10 | 0.05 / 0.00 / 0.24 |
| gate P of the best stay plan, median | 0.006 | 0.015 | 0.009 | 0.009 |
| lowest gate P among leave plans, median | 0.87 | 0.97 | 0.13 | 0.23 |
| leave plans vetoed, median (of 11 and 18.5) | 11 | 11 | 0 | 1.5 |
| new pick is a leave plan | 0 of 34 | 0 of 34 | 0 of 34 | 0 of 34 |
| new pick ends, forward, median | 0.05 m | 0.01 m | −0.04 m | −0.04 m |

- **d2715023**: the gate head says every leave plan collides (collision output 0.88, then 0.97 after C.4), from the first
  second: at handover the B head gave the best leave plan 0.45, the C.4 head 0.94. The pull-out labels taught it that
  leaving from a standstill usually touches a box (70 % of such plans do in the labels, 4.4), and it applies that here.
  This agrees with the scorer's own raw score for leaving on this scene (0.09, S8 L3). Whether leaving really collides on
  this clip cannot be told: there are no grades or boxes for exam clips in this work.
- **b44a4a44**: leave plans pass the gate (P 0.16–0.29, none vetoed at most replans) and the pick still stays at 34 of 34.
  Here the ranker, not the gate, prefers staying, the same picture as the scorer (leave 0.80 against stay 0.92, S8 L3).
- Neither build would have moved either car.

## 5. Part A

`A_spec.md`: generator-only continuation from pass 9, scorer and encoder frozen, added spacing term on all 32 plans
(mean |second difference of consecutive waypoint distances|), weights 0.03 / 0.1 / 0.3 on a 1-pass pilot, then 2 passes,
learning rate 2.5e-5, S4's tripwires plus a scorer-unchanged check. **Not run.** Anchoring measurement (CPU, UNSEEN
menus, `a_spacing_now.csv`): the term is 2.47 m on today's menus (13.3 m on highway at ≥ 20 m/s, 0.39 m on ordinary),
0.046 m on the human path and 0.027 m on catalogue curves; the generator's distance term is 0.74.

## 6. Flags

1. **The brief's success table for A did not arrive** ("success numbers as in the table above"; there is no table in the
   brief I received). `A_spec.md` section 7 holds my proposal and says so.
2. **Selecting on SEEN held-out clips is optimistic, for everything.** The trunk that makes the 256 features trained on
   those clips. Regret there is 0.072–0.080 for every head; on UNSEEN 0.102–0.115. P-B's choice of t was made on the SEEN
   curve as the brief asked; the curve is flat, so t = 0.6 is close to arbitrary. A threshold or a stopping rule that is
   meant to hold on new streets needs unseen clips set aside for it; UNSEEN was used once, at one t, as asked.
3. **The gate head cannot reach "< 4 %" at any threshold**: the plan it trusts most is graded 0 on 5.1 % of UNSEEN instants.
4. **Collision and time buffer are not map-free in the code** (4.1). What is stored is those checks with the three map
   facts fixed by hand, two ways. The strict and lenient results differ by under 2 points everywhere.
5. **The map-free collision label has a 13 % false-alarm floor**: the human path fails it on 13.3 % (strict) of pull-out
   examples. It was used as a training label in C.4 as it is.
6. **C.4's better UNSEEN numbers (0.102, 7.3 %) are one retrained ranker**, not the gate head: with B's ranker the C.4 gate
   head gives 0.109 and 8.8 %. No repeat with other seeds was run, so the size of the run-to-run spread is not measured;
   the validation regrets of the 11 B rankers (0.075–0.081) suggest about 0.005.
7. **The pull-out clips are training clips of pass 9** (human path only). Its menus and features on them are in-sample; the
   shares in 4.4 may be better than they would be on unseen pull-outs.
8. **283 pull-out clips, not 285.** `04_training_set/manifest.parquet` has 283 `B_raw` clips.
9. **P-C's first half holds only as written.** Over all menus 72.9 %; with the car standing still 48.1 %; 30 % of clips never
   have a collision-free leave plan while standing. I pre-registered the all-menus reading as the verdict and the
   standing-still one as the telling number.
10. **Trial scenes: read-only replay on CPU.** Camera frames of the two exam clips were written to a scratch folder on the
    NAS (850 MB) for 3 minutes and deleted; `c4_trial_features.npz` holds features, plans and scores only. The first delete
    failed on open NFS handles and was completed by hand a minute later. The replay's worker log is in `logs/`. No file
    under `07_closed_loop_test/`, the two run folders, `adapters/` or any earlier stage folder is newer than S9's start.
11. **Feature pass against the trial's stored plans: up to 0.9 mm and 1e-4 in score**, not 0 (torch 2.8 overlay here, the
    adapter's worker on its own torch). Accepted under the 1 mm rule set in the script; the replay itself matched to 0.0.
12. **The nudge row for the two-stage head** uses sigmoid(ranker output), whose scale is arbitrary (S7 flag 8).
13. **Correction to S8 REPORT 2.2, step 3**: the "term pushing plans apart" in the generator loss has weight 0
    (`drive_jepa_config.py:50`). Today nothing in the loss acts on the 31 plans that are not the closest one.
14. **S4's evaluation file shows the collision-corner and area losses at 0.0** on `val_full`
    (`RD/06_first_training_run/run/evals.csv`: `pred_ce_loss`, `pred_l1_loss`, `pred_area_loss`). S6 2.3 listed them as active scorer
    losses. Seen while writing the A spec; not investigated.
15. **Start gate**: 5-minute load < 32, free swap > 15 GB, available RAM > 40 GB held at every start; the long run never
    paused. **GPU: not used** (every process ran with `CUDA_VISIBLE_DEVICES=""`; `torch.cuda.is_available()` false in the log).
16. **Disk.** Start (00:00): root 44 GB free of 1.8 TB, NAS 5.37 TB. End (02:11): root **37 GB**, NAS 5.37 TB. S9 wrote
    5.8 MB locally and 382 MB on the NAS. A search of my home and the temp folders for files over 20 MB written since
    midnight found only a 256 MB update of the CLI itself; the other 7 GB is not S9's. Root is at 98 %.
17. One line in `s9_trial_features.py` (the delete, now `ignore_errors=True`) was changed after it ran; no result depends on it.

## 7. Files

Local `RD/11_scorer_head_and_pullout_grades/` (58 files; every sha256 in `files.sha256`), copied to `{runs}/drivejepa_retrain/11_scorer_head_and_pullout_grades/`.

| group | files |
|---|---|
| predictions | `PREDICTIONS.md` (`31d42933…1d66b`), `predictions.sha256` |
| part B | `s9_lib.py`, `s9_b.py`, `b_selection.json`, `b_t_curve.csv`, `b_gate_auroc.csv`, `b_table.csv`, `b_veto_stats.csv`, `b_training_runs.csv`, `b_models.pt`, `b_unseen_per_instant.parquet`, `b.log` |
| part C dump | `s9_c_dump.py`, `run_s9_dump.sh`, `c_projection.md`, `c_timing.log`, `c_dump.log`, `c_dump_launches.jsonl`; NAS `pullout/` (`pullout_all.npz` `5e113b3e…125a0`, `pullout_manifest.csv` `93751388…9169b`, 36 shards, `files.sha256`) |
| part C tables | `s9_c_analyze.py`, `c_alignment.csv`, `c_per_plan.csv`, `c_per_menu_summary.csv`, `c_per_menu.parquet`, `c_by_clip.csv`, `c_pullout_files.sha256`, `c_analyze.log`, and the same with `_timing` for the 10-clip run |
| part C.4 | `c4_selection.json`, `c4_t_curve.csv`, `c4_gate_auroc.csv`, `c4_table.csv`, `c4_veto_stats.csv`, `c4_training_runs.csv`, `c4_models.pt`, `c4_unseen_per_instant.parquet`, `c4.log` |
| trial scenes (exam, inspection only) | `s9_trial_inputs.py`, `s9_trial_features.py`, `run_s9_trial_features.sh`, `s9_trial_gate.py`, `c4_trial_replay.json`, `c4_trial_features.npz`, `c4_exam_inspection_replans.csv`, `c4_exam_inspection_plans32.csv`, `c4_exam_inspection_summary.csv`, three logs, `logs/trial_worker.log` |
| part A | `A_spec.md`, `s9_a_spacing_now.py`, `a_spacing_now.csv`, `a_spacing_now.log` |
| run control | `run_s9_py.sh`, `run_s9_after_dump.sh`, `state_log.csv`, `files.sha256` |

**CPU hours**: pull-out run 9.74; its timing run 0.36; part B 0.12; C.4 0.08; trial replay 0.11 and feature pass 0.09;
everything else under 0.01. **Total about 10.5 CPU hours**, all but 0.4 of it the pull-out run. Wall time 2 h 10 min.
