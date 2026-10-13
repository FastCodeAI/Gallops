# T03 - offline validation of three candidate fixes for the high-speed swerve (Drive-JEPA adapter 0.1.6)

Offline, CPU only. Scripts: `t03_common.py` (imports t02's reader/controller, adds the fork-based branch, Fix A/B constructions, a Sim with other MPC settings), `t03_testbed.py`, `t03_fixc.py`, `t03_report.py` (this file is generated). Interpreter `{runs}/alpasim_bulk/nurec-venv/bin/python`. Nothing under `alpasim/` or `adapters/` was edited. Measured facts only; no ranking.

## 0. What was run

- **Full set, no subsampling**: all 1170 replans of all 30 scenes x 6 variants = 7020 held-plan runs (66 s wall on 15 processes). 0 MPC solves returned a status other than solved.
- 1020 replans are at or after handover (34 per scene). The other 150 (5 per scene, -2.5 s .. -0.5 s) are **before handover**: the controller request on those ticks carries the 151-pose ground-truth trajectory, not the model's plan, so the model's plan never reached the controller. They are run as a hypothetical ('what if this plan had been handed to the controller at the recorded state') with the menu's chosen plan as baseline, tagged `pre_handover_hypothetical`, and counted separately everywhere.
- Procedure per replan: one simulator per scene is replayed teacher-forced from the session's first request (t02's `warm_tf`); at each replan tick the process is forked once per distinct plan, the child holds that plan for 1.0 s in one request (10 solves, 50 ms reporting) and returns the metrics; the parent continues the recorded replay. Checked: after branching, the parent still reproduces the recorded steering to 3e-16 rad; two branches of the same plan give identical numbers.
- Baseline plan after handover = the 9 poses in the recorded controller request (max difference from the regenerated menu's chosen proposal asserted < 1 mm on every replan).
- Tags: `car_speed_mps` = recorded speed at the tick; `pre_swerve` = no recorded |front steering angle| > 0.2 rad in the preceding 1.0 s and no off-road tick so far; groups by handover speed: highway (04749bb9, acd22430), bd07fb91, other >= 10 m/s (9 scenes), < 10 m/s (18 scenes).
- `peak_abs_lateral_from_plan_path_m` = perpendicular distance of the car from the held plan's own polyline (continued straight past its end; blank when the plan is shorter than 0.5 m); `peak_abs_lateral_rig_m` = t02's definition (y in the rig frame at the tick).

Variant constructions (path = polyline through the 9 poses, arc-length parametrised, as t02's `Path`):
- **A1** s(t) = v0 t + a t^2/2 with a = 2 (L - 4 v0)/16, L = the plan's 4 s arc length; if v0 + 4a < 0 the deceleration that stops exactly at L is used and the plan stays there. Heading = path tangent (3-point smoothed, as t02 B4e); where neighbouring poses are < 0.05 m apart the tangent is undefined and the path's re-sampled recorded heading is used.
- **A2** = A1 with a >= -4.0 m/s2 when car speed >= 10 m/s. **A3** = recorded plan unless a1 < -4.0 and speed >= 10; then poses at 0.5/1.0/1.5 s are re-spaced on the path with the constant acceleration that reaches the plan's 2 s arc length (heading re-sampled), poses 2.0-4.0 s untouched.
- **B1** drop proposals with a1 < -4.0 (speed >= 10) or a1 > +4.0 or mean 4 s speed > 1.5 v0 + 5; pick = argmax of the adapter's final score (`pdm_score`, the score after the momentum recalibration; argmax of it equals the recorded `chosen` on 1,170/1,170 replans) among the survivors. **B2** = A1 applied to the B1 pick.
- a1 = 2 (s(1 s) - v0), t02's ordering variable.

## 1. Baseline reproduction and the key ticks

Baseline at the two highway handover ticks: **0.525 / 0.401 rad** (t02: 0.525 / 0.401). Peak |front steering angle| within 1.0 s (rad):

| scene | tick_after_handover_s | car_speed_mps | pre_swerve | plan_a1_mps2 | baseline | A1 | A2 | A3 | B1 | B2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 0 | 23.563 | True | -7.109 | 0.525 | 0.005 | 0.005 | 0.017 | 0.019 | 0.003 |
| acd22430 | 0 | 23.429 | True | -5.196 | 0.401 | 0.001 | 0.001 | 0.005 | 0.289 | 0.002 |
| 04749bb9 | 0.5 | 22.185 | False | -13.312 | 0.685 | 0.415 | 0.41 | 0.51 | 0.429 | 0.393 |
| acd22430 | 0.5 | 22.498 | False | -12.032 | 0.612 | 0.216 | 0.216 | 0.531 | 0.292 | 0.198 |
| 04749bb9 | 3 | 21.912 | False | -19.271 | 0.784 | 0.46 | 0.407 | 0.781 | 0.43 | 0.36 |
| acd22430 | 3 | 9.09 | False | -5.103 | 0.27 | 0.257 | 0.257 | 0.27 | 0.27 | 0.257 |
| bd07fb91 | 6 | 14.295 | True | -10.482 | 0.783 | 0.084 | 0.084 | 0.191 | 0.108 | 0.108 |

## 2. Replans whose held plan steers > 0.3 rad within 1.0 s (post-handover; denominators in `n_replans`)

| group | state | n_replans | baseline | A1 | A2 | A3 | B1 | B2 |
|---|---|---|---|---|---|---|---|---|
| highway | pre-swerve | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| highway | disturbed | 66 | 36 | 33 | 33 | 34 | 31 | 31 |
| bd07fb91 | pre-swerve | 25 | 1 | 2 | 2 | 0 | 0 | 2 |
| bd07fb91 | disturbed | 9 | 5 | 5 | 5 | 5 | 5 | 5 |
| other>=10 | pre-swerve | 306 | 0 | 0 | 0 | 0 | 0 | 0 |
| <10 | pre-swerve | 508 | 1 | 2 | 2 | 1 | 1 | 2 |
| <10 | disturbed | 104 | 49 | 54 | 54 | 49 | 49 | 54 |

By car speed at the tick:

| speed | state | n_replans | baseline | A1 | A2 | A3 | B1 | B2 |
|---|---|---|---|---|---|---|---|---|
| car < 10 m/s | disturbed | 160 | 74 | 80 | 80 | 74 | 74 | 80 |
| car < 10 m/s | pre-swerve | 579 | 1 | 4 | 4 | 1 | 1 | 4 |
| car >= 10 m/s | disturbed | 19 | 16 | 12 | 12 | 14 | 11 | 10 |
| car >= 10 m/s | pre-swerve | 262 | 3 | 0 | 0 | 0 | 0 | 0 |

Same split, threshold 0.1 rad:

| speed | state | baseline | A1 | A2 | A3 | B1 | B2 |
|---|---|---|---|---|---|---|---|
| car < 10 m/s | disturbed | 147 | 147 | 147 | 147 | 147 | 147 |
| car < 10 m/s | pre-swerve | 36 | 43 | 43 | 36 | 36 | 43 |
| car >= 10 m/s | disturbed | 19 | 18 | 18 | 18 | 18 | 18 |
| car >= 10 m/s | pre-swerve | 3 | 0 | 0 | 1 | 2 | 1 |

**Added (not asked): the same holds continued to 2.0 s** (the longest hold whose 2 s MPC horizon stays inside the 4 s plan; first 1.0 s identical to the 1.0 s hold). Replans steering > 0.3 rad within 2.0 s:

| speed | state | n_replans | baseline | A1 | A2 | A3 | B1 | B2 |
|---|---|---|---|---|---|---|---|---|
| car < 10 m/s | disturbed | 160 | 83 | 86 | 86 | 83 | 83 | 86 |
| car < 10 m/s | pre-swerve | 579 | 6 | 10 | 10 | 6 | 6 | 10 |
| car >= 10 m/s | disturbed | 19 | 16 | 13 | 13 | 15 | 13 | 10 |
| car >= 10 m/s | pre-swerve | 262 | 3 | 0 | 0 | 0 | 0 | 0 |

Peak |steer| within 2.0 s at the key ticks (rad):

| scene | tick_after_handover_s | baseline | A1 | A2 | A3 | B1 | B2 |
|---|---|---|---|---|---|---|---|
| 04749bb9 | 0 | 0.525 | 0.005 | 0.005 | 0.21 | 0.019 | 0.003 |
| acd22430 | 0 | 0.401 | 0.001 | 0.001 | 0.127 | 0.289 | 0.003 |
| 04749bb9 | 0.5 | 0.685 | 0.415 | 0.41 | 0.51 | 0.429 | 0.393 |
| acd22430 | 0.5 | 0.741 | 0.216 | 0.216 | 0.782 | 0.646 | 0.198 |
| 04749bb9 | 3 | 0.785 | 0.46 | 0.407 | 0.785 | 0.726 | 0.482 |
| acd22430 | 3 | 0.27 | 0.257 | 0.257 | 0.27 | 0.27 | 0.257 |
| bd07fb91 | 6 | 0.783 | 0.109 | 0.109 | 0.191 | 0.158 | 0.134 |

Pre-handover hypothetical replans (not part of the counts above):

| speed | n_replans | baseline | A1 | A2 | A3 | B1 | B2 |
|---|---|---|---|---|---|---|---|
| car < 10 m/s | 90 | 1 | 1 | 1 | 1 | 1 | 1 |
| car >= 10 m/s | 60 | 9 | 0 | 0 | 0 | 0 | 0 |

## 3. Change relative to baseline, by car speed and state

`n_plan_or_pick_changed` = plan differs from the recorded one by > 1e-6 (A1/A2 re-time every plan; B1 only when the pick changes). Speed after 1 s minus baseline: positive = the car is faster after the 1.0 s hold than with the recorded plan (median over changed plans).

| variant | subset | n | n_plan_or_pick_changed | n_abs_change_gt_0p02 | n_worse_by_gt_0p05 | n_better_by_gt_0p05 | max_increase_rad | max_decrease_rad | max_peak_steer_rad | baseline_max_peak_steer_rad | median_speed_after_1s_minus_baseline_mps | max_speed_after_1s_minus_baseline_mps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | car < 10 m/s, pre-swerve | 579 | 579 | 12 | 6 | 0 | 0.2 | 0.02 | 0.374 | 0.302 | 0.074 | 2.1 |
| A1 | car < 10 m/s, disturbed | 160 | 160 | 40 | 17 | 2 | 0.099 | 0.127 | 0.63 | 0.657 | 0.096 | 1.476 |
| A1 | car >= 10 m/s, pre-swerve | 262 | 262 | 3 | 0 | 3 | 0.003 | 0.699 | 0.084 | 0.783 | 0.02 | 2.619 |
| A1 | car >= 10 m/s, disturbed | 19 | 19 | 14 | 1 | 11 | 0.153 | 0.502 | 0.521 | 0.784 | 0.533 | 2.873 |
| A2 | car < 10 m/s, pre-swerve | 579 | 579 | 12 | 6 | 0 | 0.2 | 0.02 | 0.374 | 0.302 | 0.074 | 2.1 |
| A2 | car < 10 m/s, disturbed | 160 | 160 | 40 | 17 | 2 | 0.099 | 0.127 | 0.63 | 0.657 | 0.096 | 1.476 |
| A2 | car >= 10 m/s, pre-swerve | 262 | 262 | 3 | 0 | 3 | 0.003 | 0.699 | 0.084 | 0.783 | 0.02 | 2.619 |
| A2 | car >= 10 m/s, disturbed | 19 | 19 | 14 | 1 | 11 | 0.153 | 0.502 | 0.521 | 0.784 | 1.987 | 3.47 |
| A3 | car < 10 m/s, pre-swerve | 579 | 0 | 0 | 0 | 0 | 0 | -0 | 0.302 | 0.302 |  | 0 |
| A3 | car < 10 m/s, disturbed | 160 | 0 | 0 | 0 | 0 | 0 | -0 | 0.657 | 0.657 |  | 0 |
| A3 | car >= 10 m/s, pre-swerve | 262 | 4 | 3 | 0 | 3 | 0 | 0.592 | 0.191 | 0.783 | -0.053 | 0.481 |
| A3 | car >= 10 m/s, disturbed | 19 | 11 | 9 | 0 | 8 | 0 | 0.436 | 0.781 | 0.784 | 0.429 | 0.789 |
| B1 | car < 10 m/s, pre-swerve | 579 | 0 | 0 | 0 | 0 | 0 | -0 | 0.302 | 0.302 |  | 0 |
| B1 | car < 10 m/s, disturbed | 160 | 0 | 0 | 0 | 0 | 0 | -0 | 0.657 | 0.657 |  | 0 |
| B1 | car >= 10 m/s, pre-swerve | 262 | 4 | 3 | 0 | 3 | 0.001 | 0.675 | 0.289 | 0.783 | 2.322 | 3.98 |
| B1 | car >= 10 m/s, disturbed | 19 | 12 | 12 | 0 | 9 | 0 | 0.524 | 0.517 | 0.784 | 5.161 | 8.546 |
| B2 | car < 10 m/s, pre-swerve | 579 | 579 | 12 | 6 | 0 | 0.2 | 0.02 | 0.374 | 0.302 | 0.074 | 2.1 |
| B2 | car < 10 m/s, disturbed | 160 | 160 | 40 | 17 | 2 | 0.099 | 0.127 | 0.63 | 0.657 | 0.096 | 1.476 |
| B2 | car >= 10 m/s, pre-swerve | 262 | 262 | 3 | 0 | 3 | 0.003 | 0.675 | 0.108 | 0.783 | 0.02 | 4.667 |
| B2 | car >= 10 m/s, disturbed | 19 | 19 | 15 | 1 | 12 | 0.153 | 0.639 | 0.506 | 0.784 | 4.029 | 8.79 |

Share of replans whose plan or pick changed: A1: 1020/1020 post-handover, 281/281 at car speed >= 10 m/s; A2: 1020/1020 post-handover, 281/281 at car speed >= 10 m/s; A3: 15/1020 post-handover, 15/281 at car speed >= 10 m/s; B1: 16/1020 post-handover, 16/281 at car speed >= 10 m/s; B2: 1020/1020 post-handover, 281/281 at car speed >= 10 m/s.

The 12 largest increases of peak steering under A1 (all at car speed < 2 m/s except one disturbed highway tick; lateral distance from the plan path stays in millimetres on the slow ones):

| scene | tick_after_handover_s | car_speed_mps | state | baseline | A1 | increase | baseline_lat_from_plan_m | A1_lat_from_plan_m | plan_arc_4s_m |
|---|---|---|---|---|---|---|---|---|---|
| bd07fb91 | 15.5 | 0.292 | pre-swerve | 0.174 | 0.374 | 0.2 | 0.005 | 0.009 | 4.28 |
| bd07fb91 | 16 | 0.191 | pre-swerve | 0.169 | 0.333 | 0.164 | 0.003 | 0.009 | 4.126 |
| 04749bb9 | 5 | 15.058 | disturbed | 0.103 | 0.257 | 0.153 | 0.986 | 2.36 | 65.919 |
| bd07fb91 | 15 | 0.37 | pre-swerve | 0.065 | 0.204 | 0.139 | 0.001 | 0.006 | 4.603 |
| bae463d3 | 7.5 | 1.768 | pre-swerve | 0.219 | 0.342 | 0.122 | 0.009 | 0.004 | 3.882 |
| acd22430 | 14 | 0.434 | disturbed | 0.146 | 0.246 | 0.099 | 0.012 | 0.008 | 3.037 |
| 791d68ce | 10.5 | 0.615 | disturbed | 0.42 | 0.519 | 0.099 | 0.011 | 0.011 | 4.715 |
| cdd9448c | 15 | 1.037 | disturbed | 0.236 | 0.332 | 0.096 | 0.006 | 0.01 | 7.341 |
| bae463d3 | 16 | 0.113 | disturbed | 0.48 | 0.575 | 0.096 | 0.003 | 0.002 | 3.192 |
| 791d68ce | 8.5 | 1.869 | disturbed | 0.326 | 0.404 | 0.078 | 0.012 | 0.01 | 2.7 |
| acd22430 | 14.5 | 0.421 | disturbed | 0.133 | 0.209 | 0.076 | 0.009 | 0.006 | 2.981 |
| 8023dce5 | 13 | 0.152 | pre-swerve | 0.087 | 0.162 | 0.075 | 0.001 | 0.001 | 1.419 |

## 4. Fix A: how much the plan is altered (position difference from the recorded chosen plan, m; post-handover replans)

| variant | group | n | n_changed | 0p5s_median_m | 0p5s_p95_m | 0p5s_max_m | 1s_median_m | 1s_p95_m | 1s_max_m | 2s_median_m | 2s_p95_m | 2s_max_m | 4s_median_m | 4s_p95_m | 4s_max_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | highway | 68 | 68 | 0.155 | 1.688 | 2.688 | 0.219 | 4.059 | 6.214 | 0.433 | 6.024 | 8.398 | 0 | 0 | 0 |
| A1 | bd07fb91 | 34 | 34 | 0.111 | 0.739 | 1.797 | 0.235 | 1.63 | 3.339 | 0.606 | 2.773 | 4.364 | 0 | 0 | 0 |
| A1 | other>=10 | 306 | 306 | 0.118 | 0.24 | 1.61 | 0.172 | 0.444 | 2.533 | 0.278 | 0.705 | 3.035 | 0 | 0 | 0 |
| A1 | <10 | 612 | 612 | 0.108 | 0.262 | 1.257 | 0.147 | 0.462 | 2.26 | 0.232 | 0.851 | 2.741 | 0 | 0 | 0 |
| A1 | all | 1020 | 1020 | 0.113 | 0.28 | 2.688 | 0.158 | 0.541 | 6.214 | 0.269 | 0.96 | 8.398 | 0 | 0 | 0 |
| A1 | car >= 10 m/s | 281 | 281 | 0.123 | 0.957 | 2.688 | 0.189 | 1.671 | 6.214 | 0.309 | 3.104 | 8.398 | 0 | 0 | 0 |
| A2 | highway | 68 | 68 | 0.155 | 1.688 | 3.036 | 0.219 | 4.152 | 7.582 | 0.433 | 6.842 | 13.64 | 0 | 0 | 20.262 |
| A2 | bd07fb91 | 34 | 34 | 0.111 | 0.739 | 1.797 | 0.235 | 1.63 | 3.339 | 0.606 | 2.773 | 4.364 | 0 | 0 | 0 |
| A2 | other>=10 | 306 | 306 | 0.118 | 0.24 | 1.61 | 0.172 | 0.444 | 2.533 | 0.278 | 0.705 | 3.035 | 0 | 0 | 0 |
| A2 | <10 | 612 | 612 | 0.108 | 0.262 | 1.257 | 0.147 | 0.462 | 2.26 | 0.232 | 0.851 | 2.741 | 0 | 0 | 0 |
| A2 | all | 1020 | 1020 | 0.113 | 0.28 | 3.036 | 0.158 | 0.541 | 7.582 | 0.269 | 0.96 | 13.64 | 0 | 0 | 20.262 |
| A2 | car >= 10 m/s | 281 | 281 | 0.123 | 0.957 | 3.036 | 0.189 | 1.671 | 7.582 | 0.309 | 3.104 | 13.64 | 0 | 0 | 20.262 |
| A3 | highway | 68 | 12 | 0 | 1.389 | 2.199 | 0 | 2.696 | 4.21 | 0 | 0 | 0 | 0 | 0 | 0 |
| A3 | bd07fb91 | 34 | 2 | 0 | 0.359 | 1.52 | 0 | 0.592 | 2.228 | 0 | 0 | 0 | 0 | 0 | 0 |
| A3 | other>=10 | 306 | 1 | 0 | 0 | 1.244 | 0 | 0 | 1.663 | 0 | 0 | 0 | 0 | 0 | 0 |
| A3 | <10 | 612 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A3 | all | 1020 | 15 | 0 | 0 | 2.199 | 0 | 0 | 4.21 | 0 | 0 | 0 | 0 | 0 | 0 |
| A3 | car >= 10 m/s | 281 | 15 | 0 | 0.307 | 2.199 | 0 | 1.144 | 4.21 | 0 | 0 | 0 | 0 | 0 | 0 |

- A1's constant acceleration is below -5 m/s2 on 1 of 1020 replans (1 of them at car speed >= 10 m/s) and below -4 m/s2 on 3; most negative -6.78 m/s2. A1 uses the stop-at-end-point rule on 25 replans.
- A2's cap bites on 3 replans; the plan then ends [5.43, 20.26, 4.93] m beyond the model's end point. A3 triggers on 15 replans.

Replans where A1 needs more than 4 m/s2 of constant deceleration (peak steer per variant, rad):

| scene | tick_after_handover_s | group | car_speed_mps | pre_swerve | a1_mps2 | plan_arc_4s_m | A1_accel_mps2 | A1_stops_before_4s | A2_cap_bites | A2_end_beyond_model_end_m | baseline | A1 | A2 | A3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 0.5 | highway | 22.185 | False | -13.312 | 51.31 | -4.679 | False | True | 5.431 | 0.685 | 0.415 | 0.41 | 0.51 |
| 04749bb9 | 3 | highway | 21.912 | False | -19.271 | 35.387 | -6.784 | True | True | 20.262 | 0.784 | 0.46 | 0.407 | 0.781 |
| 04749bb9 | 6 | highway | 16.117 | False | -12.631 | 27.537 | -4.716 | True | True | 4.93 | 0.784 | 0.39 | 0.358 | 0.564 |

## 5. Fix B1: menu filter

| speed | n | survive_min | survive_median | survive_mean | n_all_32_survive | n_le_1_survive | n_none | n_pick_changed | n_dropped_a1_low | n_dropped_a1_high | n_dropped_too_fast |
|---|---|---|---|---|---|---|---|---|---|---|---|
| car < 10 m/s | 739 | 25 | 30 | 30.161 | 200 | 0 | 0 | 0 | 0 | 1345 | 48 |
| car >= 10 m/s | 281 | 1 | 28 | 27.416 | 0 | 1 | 0 | 16 | 793 | 493 | 11 |

Pick changed on 16/1020 post-handover replans (16/281 at car speed >= 10 m/s; 10/150 of the pre-handover hypothetical ones). The filter never emptied the menu. Every replan whose pick changed:

| scene | tick_after_handover_s | phase | car_speed_mps | pre_swerve | n_survive_of_32 | original_pick | new_pick | new_pick_rank_final_score | new_pick_rank_plain_score | original_a1_mps2 | new_a1_mps2 | original_mean_speed_4s_mps | new_mean_speed_4s_mps | human_mean_speed_4s_mps | new_lateral_from_original_path_2s_m | new_lateral_from_original_path_4s_m | baseline | B1 | B2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | -2.5 | pre_handover_hypothetical | 24.46 | True | 11 | 26 | 18 | 3 | 3 | -6.009 | -3.022 | 20.311 | 21.872 | 23.691 | -0.006 | 0.078 | 0.405 | 0.003 | 0.001 |
| 04749bb9 | -2 | pre_handover_hypothetical | 24.123 | True | 10 | 26 | 18 | 3 | 3 | -6.014 | -3.145 | 19.793 | 21.071 | 23.652 | -0.013 | -0.097 | 0.445 | 0.009 | 0.001 |
| 04749bb9 | -1.5 | pre_handover_hypothetical | 23.735 | True | 12 | 26 | 18 | 8 | 8 | -6.044 | -3.365 | 19.218 | 20.446 | 23.638 | -0.013 | -0.183 | 0.481 | 0.236 | 0.002 |
| 04749bb9 | -0.5 | pre_handover_hypothetical | 23.47 | True | 11 | 26 | 18 | 4 | 4 | -5.45 | -2.728 | 20.09 | 20.817 | 23.71 | -0.009 | -0.038 | 0.423 | 0.019 | 0.002 |
| 04749bb9 | 0 | post_handover | 23.563 | True | 11 | 13 | 18 | 5 | 5 | -7.109 | -2.469 | 18.855 | 21.092 | 23.743 | -0.015 | 0.313 | 0.525 | 0.019 | 0.003 |
| 04749bb9 | 0.5 | post_handover | 22.185 | False | 6 | 24 | 5 | 15 | 15 | -13.312 | -3.714 | 12.828 | 16.32 | 23.758 | 7.826 | 5.495 | 0.685 | 0.429 | 0.393 |
| 04749bb9 | 3 | post_handover | 21.912 | False | 1 | 11 | 4 | 12 | 12 | -19.271 | -2.622 | 8.847 | 23.214 | 23.388 | -0.873 | -9.666 | 0.784 | 0.43 | 0.36 |
| 04749bb9 | 3.5 | post_handover | 18.385 | False | 19 | 11 | 6 | 3 | 3 | -11.401 | -1.818 | 11.459 | 22.251 | 23.248 | 2.881 | 15.622 | 0.533 | 0.289 | 0.274 |
| 04749bb9 | 4.5 | post_handover | 12.776 | False | 18 | 20 | 6 | 2 | 2 | 6.597 | 3.792 | 19.288 | 17.242 | 22.999 | 0.593 | 0.742 | 0.181 | 0.14 | 0.124 |
| 04749bb9 | 5.5 | post_handover | 16.847 | False | 26 | 10 | 27 | 2 | 2 | -4.822 | -2.807 | 13.416 | 13.928 | 22.85 | -0.642 | -4.989 | 0.527 | 0.041 | 0.028 |
| 04749bb9 | 6 | post_handover | 16.117 | False | 6 | 11 | 30 | 11 | 11 | -12.631 | -3.283 | 6.884 | 15.112 | 22.83 | 6.46 | 32.933 | 0.784 | 0.26 | 0.227 |
| 04749bb9 | 6.5 | post_handover | 12.524 | False | 20 | 11 | 16 | 2 | 2 | -9.568 | -3.199 | 5.21 | 10.974 | 22.811 | 5.132 | 20.625 | 0.495 | 0.468 | 0.461 |
| 16b83346 | 3 | post_handover | 10.361 | True | 28 | 10 | 12 | 2 | 2 | -6.244 | -3.064 | 7.435 | 8.273 | 6.672 | -0.008 | -0.816 | 0.002 | 0.003 | 0.003 |
| acd22430 | -2 | pre_handover_hypothetical | 23.406 | True | 12 | 26 | 18 | 8 | 8 | -5.789 | -3.044 | 19.377 | 20.862 | 23.336 | -0.023 | -0.05 | 0.471 | 0.206 | 0.005 |
| acd22430 | -1.5 | pre_handover_hypothetical | 23.291 | True | 12 | 26 | 18 | 8 | 8 | -5.666 | -3.13 | 19.438 | 21.189 | 23.321 | -0.029 | 0.131 | 0.413 | 0.095 | 0.003 |
| acd22430 | -1 | pre_handover_hypothetical | 23.372 | True | 11 | 26 | 18 | 5 | 5 | -5.587 | -2.97 | 20.127 | 21.412 | 23.323 | -0.024 | 0.054 | 0.426 | 0.097 | 0.003 |
| acd22430 | -0.5 | pre_handover_hypothetical | 23.367 | True | 12 | 26 | 28 | 2 | 2 | -5.022 | -3.727 | 20.437 | 20.426 | 23.325 | -0.003 | -0.15 | 0.341 | 0.208 | 0.002 |
| acd22430 | 0 | post_handover | 23.429 | True | 12 | 26 | 28 | 3 | 3 | -5.196 | -3.822 | 20.37 | 20.469 | 23.327 | -0.006 | -0.132 | 0.401 | 0.289 | 0.002 |
| acd22430 | 0.5 | post_handover | 22.498 | False | 2 | 13 | 29 | 7 | 7 | -12.032 | -3.955 | 15.137 | 23.274 | 23.325 | 0.265 | 32.035 | 0.612 | 0.292 | 0.198 |
| acd22430 | 1.5 | post_handover | 18.053 | False | 24 | 11 | 3 | 3 | 3 | -11.136 | -1.661 | 12.212 | 15.99 | 23.33 | -0.058 | 0.101 | 0.781 | 0.485 | 0.142 |
| acd22430 | 2 | post_handover | 16.299 | False | 25 | 11 | 13 | 3 | 3 | -9.304 | -2.497 | 9.8 | 15.863 | 23.317 | 2.589 | 12.262 | 0.587 | 0.456 | 0.421 |
| acd22430 | 2.5 | post_handover | 12.467 | False | 27 | 11 | 13 | 4 | 4 | -8.641 | -3.981 | 6.621 | 9.776 | 23.283 | 0.502 | -13.883 | 0.407 | 0.275 | 0.267 |
| bae463d3 | -2.5 | pre_handover_hypothetical | 0.039 | True | 25 | 19 | 16 | 2 | 2 | 4.262 | 1.417 | 2.42 | 1.813 | 1.159 | -0.01 | 0.006 | 0.011 | 0.003 | 0.002 |
| bd07fb91 | 6 | post_handover | 14.295 | True | 25 | 11 | 3 | 4 | 4 | -10.482 | -3.23 | 6.714 | 10.423 | 7.976 | 1.064 | -6.279 | 0.783 | 0.108 | 0.108 |
| bd07fb91 | 6.5 | post_handover | 11.499 | False | 28 | 11 | 15 | 3 | 3 | -8.777 | -3.665 | 4.655 | 7.167 | 6.807 | 0.404 | 0.938 | 0.563 | 0.513 | 0.506 |
| e121e37d | -2.5 | pre_handover_hypothetical | 13.065 | True | 29 | 11 | 24 | 2 | 2 | -9.541 | -3.885 | 7.879 | 8.239 | 12.984 | 0.001 | 0.02 | 0.648 | 0.003 | 0.003 |

## 6. Cost side: replans where the fix brakes later or less than the model asked

A2 rows = the cap bites (plan ends beyond the model's end point). B1 rows = the pick changed and the new pick's 4 s arc length is > 0.1 m longer than the original's. The run's per-tick metrics contain collision_*, offroad and min_distance_to_obstacle_m but **no traffic-light metric**, so 'red light followed' cannot be read from the recorded metrics. The recorded-run columns describe what happened in the recorded run (with the original plan and the swerve), within 4 s after the tick.

| fix | scene | tick_after_handover_s | phase | car_speed_mps | pre_swerve | original_plan_end_speed_mps | original_plan_a1_mps2 | original_const_accel_to_end_mps2 | fixed_plan_end_speed_mps | ends_beyond_model_end_m | recorded_collision_any_within_4s | recorded_collision_front_or_lateral_within_4s | recorded_offroad_within_4s | recorded_min_distance_to_obstacle_within_4s_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 04749bb9 | -2.5 | pre_handover_hypothetical | 24.46 | True | 16.825 | -6.009 | -2.074 | 22.685 | 6.241 | False | False | False | 4.715 |
| B1 | 04749bb9 | -2 | pre_handover_hypothetical | 24.123 | True | 16.391 | -6.014 | -2.165 | 21.501 | 5.111 | False | False | False | 6.837 |
| B1 | 04749bb9 | -1.5 | pre_handover_hypothetical | 23.735 | True | 15.756 | -6.044 | -2.259 | 20.141 | 4.909 | False | False | False | 12.681 |
| B1 | 04749bb9 | -0.5 | pre_handover_hypothetical | 23.47 | True | 18.208 | -5.45 | -1.69 | 20.095 | 2.906 | False | False | True | 18.815 |
| B1 | 04749bb9 | 0 | post_handover | 23.563 | True | 9.113 | -7.109 | -2.354 | 22.513 | 8.947 | False | False | True | 18.815 |
| A2 | 04749bb9 | 0.5 | post_handover | 22.185 | False | 18.381 | -13.312 | -4.679 | 7.185 | 5.431 | False | False | True | 19.075 |
| B1 | 04749bb9 | 0.5 | post_handover | 22.185 | False | 18.381 | -13.312 | -4.679 | 6.795 | 13.97 | False | False | True | 19.075 |
| A2 | 04749bb9 | 3 | post_handover | 21.912 | False | 5.784 | -19.271 | -6.784 | 6.912 | 20.262 | False | False | True | 15.376 |
| B1 | 04749bb9 | 3 | post_handover | 21.912 | False | 5.784 | -19.271 | -6.784 | 44.496 | 57.471 | False | False | True | 15.376 |
| B1 | 04749bb9 | 3.5 | post_handover | 18.385 | False | 9.109 | -11.401 | -3.463 | 35.546 | 43.171 | False | False | False | 15.376 |
| B1 | 04749bb9 | 5.5 | post_handover | 16.847 | False | 9.174 | -4.822 | -1.716 | 9.088 | 2.049 | False | False | False | 15.376 |
| A2 | 04749bb9 | 6 | post_handover | 16.117 | False | 3.655 | -12.631 | -4.716 | 1.117 | 4.93 | False | False | False | 17.939 |
| B1 | 04749bb9 | 6 | post_handover | 16.117 | False | 3.655 | -12.631 | -4.716 | 14.423 | 32.912 | False | False | False | 17.939 |
| B1 | 04749bb9 | 6.5 | post_handover | 12.524 | False | 2.601 | -9.568 | -3.763 | 11.829 | 23.059 | False | False | False | 25.02 |
| B1 | 16b83346 | 3 | post_handover | 10.361 | True | 7.929 | -6.244 | -1.463 | 7.763 | 3.352 | False | False | False | 1.477 |
| B1 | acd22430 | -2 | pre_handover_hypothetical | 23.406 | True | 17.159 | -5.789 | -2.015 | 22.146 | 5.941 | False | False | True | 1.768 |
| B1 | acd22430 | -1.5 | pre_handover_hypothetical | 23.291 | True | 17.976 | -5.666 | -1.927 | 23.437 | 7.003 | False | False | True | 1.768 |
| B1 | acd22430 | -1 | pre_handover_hypothetical | 23.372 | True | 18.975 | -5.587 | -1.622 | 19.673 | 5.139 | False | False | True | 1.768 |
| B1 | acd22430 | 0 | post_handover | 23.429 | True | 17.521 | -5.196 | -1.53 | 13.091 | 0.395 | False | False | True | 4.87 |
| B1 | acd22430 | 0.5 | post_handover | 22.498 | False | 2.632 | -12.032 | -3.681 | 12.742 | 32.549 | False | False | True | 6.823 |
| B1 | acd22430 | 1.5 | post_handover | 18.053 | False | 10.674 | -11.136 | -2.92 | 10.86 | 15.113 | False | False | False | 6.823 |
| B1 | acd22430 | 2 | post_handover | 16.299 | False | 5.091 | -9.304 | -3.249 | 9.676 | 24.253 | False | False | False | 6.823 |
| B1 | acd22430 | 2.5 | post_handover | 12.467 | False | 8.943 | -8.641 | -2.923 | 9.332 | 12.622 | False | False | False | 6.823 |
| B1 | bd07fb91 | 6 | post_handover | 14.295 | True | 5.704 | -10.482 | -3.804 | 8.924 | 14.833 | True | True | False | 0 |
| B1 | bd07fb91 | 6.5 | post_handover | 11.499 | False | 3.01 | -8.778 | -3.551 | 4.163 | 10.045 | True | True | False | 0 |
| B1 | e121e37d | -2.5 | pre_handover_hypothetical | 13.065 | True | 8.84 | -9.541 | -2.593 | 5.007 | 1.441 | False | False | False | 4.54 |

## 7. Stability check: 4.0 s holds

**Only the first 2.0 s of a 4 s hold are a fair test.** The plan has poses up to 4.0 s and the MPC looks 2.0 s ahead; `_tvp_fun` clamps reference times to the plan's last pose, so from 2.0 s into the hold the reference for the far horizon steps stands still at the end point. Every variant, including the re-timed ones, then gets a full -9 m/s2 brake command and a steering swing to 0.68-0.785 rad between 3 and 4 s (in the run this never happens: a new plan arrives every 0.5 s). The `first_2s` columns are the valid part; the whole-4 s columns are shown as asked.

| scene | tick_after_handover_s | variant | peak_abs_steer_first_2s_rad | peak_abs_lateral_from_plan_path_first_2s_m | min_u_long_first_2s_mps2 | peak_abs_steer_rad | peak_abs_lateral_from_plan_path_m | min_u_long_mps2 | plan_arc_4s_m | dist_travelled_4s_m | steer_1s_rad | speed_1s_mps | lat_1s_m | steer_2s_rad | speed_2s_mps | lat_2s_m | steer_3s_rad | speed_3s_mps | lat_3s_m | steer_4s_rad | speed_4s_mps | lat_4s_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 0 | baseline | 0.525 | 3.833 | -5.158 | 0.785 | 3.833 | -9 | 75.42 | 74.874 | -0.009 | 19.683 | 2.762 | 0.252 | 18.577 | -2.411 | -0.131 | 19.03 | -0.89 | 0.114 | 10.336 | -1.62 |
| 04749bb9 | 0 | A1 | 0.005 | 0.042 | -3.372 | 0.679 | 1.351 | -9 | 75.42 | 72.152 | -0.001 | 20.97 | 0.029 | -0 | 18.573 | 0.013 | 0.018 | 16.157 | -0.03 | -0.679 | 8.495 | 1.35 |
| 04749bb9 | 0 | A2 | 0.005 | 0.042 | -3.372 | 0.679 | 1.351 | -9 | 75.42 | 72.152 | -0.001 | 20.97 | 0.029 | -0 | 18.573 | 0.013 | 0.018 | 16.157 | -0.03 | -0.679 | 8.495 | 1.35 |
| 04749bb9 | 0 | A3 | 0.21 | 1.232 | -5.82 | 0.785 | 2.978 | -9 | 75.42 | 73.187 | -0.017 | 19.157 | 0.207 | 0.21 | 17.709 | -1.232 | -0.114 | 18.875 | -0.679 | 0.148 | 10.258 | -1.619 |
| 04749bb9 | 0 | B1 | 0.019 | 0.209 | -2.148 | 0.76 | 2.111 | -9 | 84.367 | 80.232 | -0.007 | 22.65 | 0.195 | 0 | 22.248 | 0.051 | 0.025 | 17.995 | 0.133 | -0.752 | 10.431 | 2.067 |
| 04749bb9 | 0 | B2 | 0.003 | 0.02 | -1.739 | 0.785 | 2.619 | -9 | 84.366 | 80.43 | -0.001 | 22.207 | 0.011 | 0.001 | 20.947 | 0.014 | -0.023 | 19.392 | -0.091 | 0.785 | 11.623 | -2.193 |
| acd22430 | 0 | baseline | 0.401 | 3.802 | -3.475 | 0.785 | 3.802 | -9 | 81.479 | 79.982 | 0.18 | 20.844 | -3.238 | 0.029 | 21.187 | 2.05 | -0.04 | 19.945 | 0.436 | 0.491 | 11.524 | -1.883 |
| acd22430 | 0 | A1 | 0.001 | 0.014 | -2.315 | 0.785 | 2.291 | -9 | 81.479 | 77.743 | 0.001 | 21.725 | 0.007 | 0 | 20.172 | 0.014 | 0.024 | 18.421 | 0.055 | -0.714 | 10.685 | 1.986 |
| acd22430 | 0 | A2 | 0.001 | 0.014 | -2.315 | 0.785 | 2.291 | -9 | 81.479 | 77.743 | 0.001 | 21.725 | 0.007 | 0 | 20.172 | 0.014 | 0.024 | 18.421 | 0.055 | -0.714 | 10.685 | 1.986 |
| acd22430 | 0 | A3 | 0.127 | 0.989 | -4.169 | 0.785 | 2.676 | -9 | 81.479 | 78.064 | -0.005 | 20.348 | -0.012 | 0.017 | 20.119 | -0.943 | 0.032 | 19.734 | -0.175 | -0.56 | 11.469 | 1.911 |
| acd22430 | 0 | B1 | 0.289 | 3.388 | -1.949 | 0.785 | 3.388 | -9 | 81.874 | 80.527 | 0.151 | 22.203 | -3.215 | 0.017 | 22.627 | 0.271 | 0.125 | 18.46 | -0.076 | 0.016 | 9.523 | 1.165 |
| acd22430 | 0 | B2 | 0.003 | 0.019 | -2.243 | 0.785 | 2.313 | -9 | 81.873 | 78.109 | 0 | 21.779 | 0.008 | 0.003 | 20.276 | 0.019 | 0.023 | 18.562 | -0.001 | -0.733 | 10.836 | 1.998 |
| bd07fb91 | 6 | baseline | 0.783 | 1.784 | -9 | 0.783 | 1.784 | -9 | 26.858 | 26.416 | 0.08 | 7.472 | 1.002 | -0.097 | 5.703 | -0.023 | -0.019 | 4.817 | 0.195 | -0.328 | 2.38 | 0.234 |
| bd07fb91 | 6 | A1 | 0.109 | 0.415 | -5.375 | 0.109 | 0.415 | -5.375 | 26.767 | 26.654 | -0.075 | 10.092 | 0.337 | -0.061 | 6.214 | -0.169 | 0.021 | 2.912 | 0.024 | 0.037 | 0.532 | 0.002 |
| bd07fb91 | 6 | A2 | 0.109 | 0.415 | -5.375 | 0.109 | 0.415 | -5.375 | 26.767 | 26.654 | -0.075 | 10.092 | 0.337 | -0.061 | 6.214 | -0.169 | 0.021 | 2.912 | 0.024 | 0.037 | 0.532 | 0.002 |
| bd07fb91 | 6 | A3 | 0.191 | 0.794 | -8.426 | 0.334 | 0.794 | -8.426 | 26.844 | 25.825 | -0.134 | 7.862 | 0.706 | -0.101 | 4.942 | 0.105 | 0.016 | 4.568 | 0.177 | -0.334 | 2.39 | 0.219 |
| bd07fb91 | 6 | B1 | 0.158 | 0.292 | -3.593 | 0.319 | 0.292 | -5.79 | 41.691 | 39.683 | -0.108 | 11.452 | 0.255 | -0.147 | 9.88 | -0.052 | -0.116 | 8.747 | 0.196 | -0.319 | 4.264 | 0.197 |
| bd07fb91 | 6 | B2 | 0.134 | 0.243 | -2.674 | 0.171 | 0.263 | -5.429 | 41.542 | 39.733 | -0.108 | 12.139 | 0.061 | -0.132 | 10.133 | -0.243 | -0.109 | 8.18 | -0.057 | -0.147 | 3.611 | -0.026 |

## 8. Fix C: controller settings

Controller config of every 0.1.6 run (`runs/drivejepa/drivejepa_016_<scene>/controller-config.yaml`; the task's path `controller/*.yaml` does not exist, `controller/` holds only the controller CSV):

```
mpc_implementation: nonlinear
n_horizon: 20
dt_mpc: 0.1
gains:
  long_position_weight: 2.0
  lat_position_weight: 1.0
  heading_weight: 1.0
  acceleration_weight: 0.1
  rel_front_steering_angle_weight: 5.0
  rel_acceleration_weight: 1.0
  idx_start_penalty: 10
```

Cost as written in `nonlinear_mpc.py::_setup_mpc` (re-read; stage term and terminal term are the same expression): `tracking_enabled_k * ( long_w * (x - x_ref)^2 + lat_w * (y - y_ref)^2 + acc_w * acceleration^2 + heading_w * wrap(yaw - heading_ref)^2 )`, x, y in the rig frame at the solve (no rotation into the reference heading), `tracking_enabled_k` = 1 for horizon step k >= idx_start_penalty else 0 (`_tvp_fun`). **The gate multiplies the whole bracket**, so for the first 10 steps (0-0.9 s) the heading and acceleration-state terms are off as well, not only the position terms. Plus do_mpc's rterm `rel_front_steering_angle_weight * (d steer_cmd)^2 + rel_acceleration_weight * (d acc_cmd)^2` on the change of each command between consecutive steps (active on every step). With the run's values: [2 x^2 + 1 y^2 + 0.1 acc^2 + 1 yaw^2] on steps 10..20, rterm 5 / 1. Bounds: acceleration_cmd in [-9, +6] m/s2.

**Exposed without a code change** (fields of `MPCGains` / `ControllerConfig`, i.e. the controller yaml): the six weights, `idx_start_penalty`, `n_horizon`, `dt_mpc`, `mpc_implementation`. **Would need a code change**: the command bounds (acceleration_cmd in [-9, +6] m/s2, steering cmd +-2 rad), the state bounds (steering +-0.785 rad, yaw +-0.78 rad, ...), any penalty on the steering angle itself or on lateral acceleration (none exists), the vehicle parameters. The only exposed steering penalty is the steering-command rate term `rel_front_steering_angle_weight`.

Variants: C1 idx_start_penalty 10 -> 0; C2 long_position_weight 2 -> 0.5; C3 heading_weight 1 -> 10; C4 rel_front_steering_angle_weight 5 -> 50; six further tries T1-T6 (names say what was changed). **C5 = `T2_long0p5_steer50`**, the try with the smallest sum of peak steer over the three original-plan cases below (criterion fixed after the fact; T4 is within 0.002 rad of it).

Peak |steer| (rad) vs constant deceleration of a straight plan, car at 23.6 m/s (04749bb9 handover state):

| controller | 0.0 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0 | 9.0 | 10.0 | 12.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 0 | 0 | 0 | 0 | 0 | 0.06 | 0.24 | 0.43 | 0.7 | 0.69 | 0.78 | 0.78 |
| C1_track_from_step0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.3 | 0.71 | 0.78 | 0.78 | 0.78 |
| C2_long_weight_0p5 | 0 | 0 | 0 | 0 | 0 | 0.02 | 0.4 | 0.51 | 0.62 | 0.74 | 0.77 | 0.78 |
| C3_heading_weight_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03 | 0.43 | 0.7 | 0.7 | 0.77 | 0.78 |
| C4_steer_rate_weight_50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.28 | 0.51 | 0.68 | 0.71 |
| T1_long0p5_heading10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.44 | 0.58 | 0.71 | 0.74 | 0.78 |
| T2_long0p5_steer50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.17 | 0.48 | 0.53 |
| T3_heading10_steer50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.02 | 0.48 | 0.67 | 0.76 |
| T4_long0p5_heading10_steer50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.24 | 0.51 |
| T5_long0p2_heading10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.5 | 0.6 | 0.7 | 0.68 |
| T6_lat_weight_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.41 | 0.67 | 0.78 | 0.76 | 0.77 |

Car at 15.7 m/s (bd07fb91 handover state):

| controller | 0.0 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0 | 9.0 | 10.0 | 12.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | 0.49 | 0.78 | 0.78 | 0.79 | 0.79 |
| C1_track_from_step0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | 0.23 | 0.78 | 0.78 | 0.79 | 0.79 |
| C2_long_weight_0p5 | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | 0.11 | 0.68 | 0.78 | 0.78 | 0.79 |
| C3_heading_weight_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.02 | 0.74 | 0.78 | 0.79 | 0.79 |
| C4_steer_rate_weight_50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.33 | 0.59 | 0.71 | 0.79 |
| T1_long0p5_heading10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.03 | 0.74 | 0.79 | 0.79 |
| T2_long0p5_steer50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.02 | 0.31 | 0.62 |
| T3_heading10_steer50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.02 | 0.71 | 0.78 |
| T4_long0p5_heading10_steer50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.01 |
| T5_long0p2_heading10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | 0.05 | 0.78 |
| T6_lat_weight_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03 | 0.69 | 0.78 | 0.78 | 0.79 |

Speed after the 1.0 s hold (m/s), 23.6 m/s case (how much braking was actually delivered):

| controller | 0.0 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0 | 9.0 | 10.0 | 12.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 23.6 | 22.5 | 21.4 | 20.3 | 19.1 | 18.1 | 17.2 | 16.6 | 16.4 | 16.1 | 15.8 | 15.5 |
| C1_track_from_step0 | 23.6 | 22.5 | 21.4 | 20.3 | 19.2 | 18.1 | 17 | 16.2 | 16.3 | 16.2 | 16 | 15.8 |
| C2_long_weight_0p5 | 23.6 | 22.5 | 21.4 | 20.3 | 19.1 | 18 | 17.5 | 16.8 | 16.5 | 16.3 | 16.2 | 16 |
| C3_heading_weight_10 | 23.6 | 22.5 | 21.4 | 20.3 | 19.1 | 18 | 16.9 | 16.5 | 16.2 | 16 | 15.8 | 15.8 |
| C4_steer_rate_weight_50 | 23.6 | 22.5 | 21.4 | 20.3 | 19.1 | 18 | 16.9 | 16.1 | 15.9 | 16 | 15.8 | 15.6 |
| T1_long0p5_heading10 | 23.6 | 22.5 | 21.4 | 20.3 | 19.1 | 18 | 16.9 | 16.6 | 16.4 | 16.2 | 16.1 | 16 |
| T2_long0p5_steer50 | 23.6 | 22.5 | 21.4 | 20.3 | 19.1 | 18 | 16.9 | 16.4 | 16.1 | 15.9 | 16 | 15.9 |
| T3_heading10_steer50 | 23.6 | 22.5 | 21.4 | 20.3 | 19.1 | 18 | 16.9 | 16.1 | 15.7 | 15.8 | 15.8 | 15.8 |
| T4_long0p5_heading10_steer50 | 23.6 | 22.5 | 21.4 | 20.3 | 19.1 | 18 | 16.9 | 16.4 | 16.1 | 15.9 | 15.8 | 15.8 |
| T5_long0p2_heading10 | 23.6 | 22.5 | 21.5 | 20.4 | 19.3 | 18.3 | 17.2 | 16.7 | 16.6 | 16.5 | 16.5 | 16.3 |
| T6_lat_weight_10 | 23.6 | 22.5 | 21.4 | 20.3 | 19.1 | 18 | 16.9 | 16.4 | 16.1 | 16 | 15.7 | 15.7 |

Onset = smallest swept deceleration whose peak steer exceeds the threshold (blank = none up to 12 m/s2):

| controller | onset_gt_0p1_rad_at_23p6mps | onset_gt_0p3_rad_at_23p6mps | onset_gt_0p1_rad_at_15p7mps | onset_gt_0p3_rad_at_15p7mps |
|---|---|---|---|---|
| default | 6 | 7 | 7 | 7 |
| C1_track_from_step0 | 7 | 8 | 7 | 8 |
| C2_long_weight_0p5 | 6 | 6 | 7 | 8 |
| C3_heading_weight_10 | 7 | 7 | 8 | 8 |
| C4_steer_rate_weight_50 | 8 | 9 | 8 | 8 |
| T1_long0p5_heading10 | 7 | 7 | 9 | 9 |
| T2_long0p5_steer50 | 9 | 10 | 10 | 10 |
| T3_heading10_steer50 | 9 | 9 | 10 | 10 |
| T4_long0p5_heading10_steer50 | 10 | 12 |  |  |
| T5_long0p2_heading10 | 8 | 8 | 12 | 12 |
| T6_lat_weight_10 | 7 | 7 | 8 | 8 |

ORIGINAL chosen plan held 1.0 s; peak |steer| (rad). Highway ticks are warm (replayed to handover with the variant controller); bd07fb91 +6.0 s is seeded cold from the recorded state for every controller (default warm shown for comparison):

| controller | 04749bb9+0.0s warm | acd22430+0.0s warm | bd07fb91+6.0s cold | bd07fb91+6.0s warm |
|---|---|---|---|---|
| default | 0.525 | 0.401 | 0.783 | 0.783 |
| C1_track_from_step0 | 0.075 | 0.006 | 0.785 |  |
| C2_long_weight_0p5 | 0.41 | 0.014 | 0.495 |  |
| C3_heading_weight_10 | 0.496 | 0.332 | 0.759 |  |
| C4_steer_rate_weight_50 | 0.003 | 0.002 | 0.151 |  |
| T1_long0p5_heading10 | 0.061 | 0.002 | 0.39 |  |
| T2_long0p5_steer50 | 0.003 | 0.002 | 0.098 |  |
| T3_heading10_steer50 | 0.003 | 0.002 | 0.113 |  |
| T4_long0p5_heading10_steer50 | 0.003 | 0.002 | 0.1 |  |
| T5_long0p2_heading10 | 0.012 | 0.002 | 0.166 |  |
| T6_lat_weight_10 | 0.438 | 0.074 | 0.783 |  |

Speed after that 1.0 s hold (m/s):

| controller | 04749bb9+0.0s warm | acd22430+0.0s warm | bd07fb91+6.0s cold | bd07fb91+6.0s warm |
|---|---|---|---|---|
| default | 19.68 | 20.84 | 7.51 | 7.47 |
| C1_track_from_step0 | 18.62 | 19.9 | 7.21 |  |
| C2_long_weight_0p5 | 19.76 | 20.32 | 7.93 |  |
| C3_heading_weight_10 | 19.61 | 20.66 | 7.47 |  |
| C4_steer_rate_weight_50 | 18.8 | 20.05 | 7.15 |  |
| T1_long0p5_heading10 | 19.11 | 20.32 | 7.71 |  |
| T2_long0p5_steer50 | 19.09 | 20.32 | 7.55 |  |
| T3_heading10_steer50 | 18.8 | 20.05 | 7.15 |  |
| T4_long0p5_heading10_steer50 | 19.09 | 20.32 | 7.54 |  |
| T5_long0p2_heading10 | 19.36 | 20.51 | 7.9 |  |
| T6_lat_weight_10 | 19.5 | 20.08 | 7.48 |  |

Tracking check, 5 clean scenes (02e075b9, 16b83346, 6725df2d, 01d503d4, 048b974e), 5 s after handover, closed loop on the simulator's own pose with the recorded plans fixed in the world; worst scene per controller. **All five scenes are near-straight in that window (recorded peak steer <= 0.004 rad), so this check exercises longitudinal tracking and says almost nothing about steering response.**

| controller | n_scenes | max_lateral_dev_m | max_longitudinal_dev_m | peak_steer_rad | recorded_peak_steer_rad | max_steer_diff_from_recorded_rad | solves_not_solved |
|---|---|---|---|---|---|---|---|
| default | 5 | 0.0063 | 0.0086 | 0.0039 | 0.0041 | 0.0013 | 0 |
| C1_track_from_step0 | 5 | 0.0085 | 0.22 | 0.0067 | 0.0041 | 0.003 | 0 |
| C2_long_weight_0p5 | 5 | 0.0063 | 0.3959 | 0.0039 | 0.0041 | 0.0013 | 0 |
| C3_heading_weight_10 | 5 | 0.0083 | 0.0086 | 0.0045 | 0.0041 | 0.0034 | 0 |
| C4_steer_rate_weight_50 | 5 | 0.0059 | 0.0086 | 0.003 | 0.0041 | 0.0021 | 0 |
| T1_long0p5_heading10 | 5 | 0.0084 | 0.3959 | 0.0044 | 0.0041 | 0.003 | 0 |
| T2_long0p5_steer50 | 5 | 0.0058 | 0.3959 | 0.0031 | 0.0041 | 0.0022 | 0 |
| T3_heading10_steer50 | 5 | 0.0076 | 0.0086 | 0.0028 | 0.0041 | 0.0022 | 0 |
| T4_long0p5_heading10_steer50 | 5 | 0.0074 | 0.3959 | 0.0028 | 0.0041 | 0.0023 | 0 |
| T5_long0p2_heading10 | 5 | 0.0084 | 0.6634 | 0.0044 | 0.0041 | 0.003 | 0 |
| T6_lat_weight_10 | 5 | 0.0082 | 0.0086 | 0.008 | 0.0041 | 0.0055 | 0 |

Supplementary (not asked): the same check on the scenes whose recorded peak steer in those 5 s is 0.05-0.35 rad (1acf2f5d, 791d68ce, 8023dce5, bae463d3, d1a25a99, d2715023; all handed over below 7 m/s). Per scene x controller rows for all 30 scenes are in `fixC_tracking.csv`.

| controller | n_scenes | max_lateral_dev_m | max_longitudinal_dev_m | peak_steer_rad | recorded_peak_steer_rad | max_steer_diff_from_recorded_rad | solves_not_solved |
|---|---|---|---|---|---|---|---|
| default | 6 | 0.0101 | 0.0075 | 0.3045 | 0.3053 | 0.0023 | 0 |
| C1_track_from_step0 | 6 | 0.0276 | 0.1179 | 0.3209 | 0.3053 | 0.0156 | 0 |
| C2_long_weight_0p5 | 6 | 0.0194 | 0.1127 | 0.3082 | 0.3053 | 0.0171 | 0 |
| C3_heading_weight_10 | 6 | 0.0232 | 0.0076 | 0.4479 | 0.3053 | 0.253 | 0 |
| C4_steer_rate_weight_50 | 6 | 0.1089 | 0.0178 | 0.2785 | 0.3053 | 0.2146 | 0 |
| T1_long0p5_heading10 | 6 | 0.0212 | 0.1127 | 0.4319 | 0.3053 | 0.2379 | 0 |
| T2_long0p5_steer50 | 6 | 0.1096 | 0.1127 | 0.2809 | 0.3053 | 0.2161 | 0 |
| T3_heading10_steer50 | 6 | 0.1175 | 0.0167 | 0.3021 | 0.3053 | 0.0312 | 0 |
| T4_long0p5_heading10_steer50 | 6 | 0.1269 | 0.1127 | 0.2975 | 0.3053 | 0.0354 | 0 |
| T5_long0p2_heading10 | 6 | 0.0221 | 0.1993 | 0.4094 | 0.3053 | 0.217 | 0 |
| T6_lat_weight_10 | 6 | 0.0542 | 0.0141 | 0.318 | 0.3053 | 0.0972 | 0 |

## 9. The task's predictions next to the measurements

- F-A, handover ticks < 0.05 rad: A1 0.005 / 0.001; A2 the same (cap does not bite there); A3 0.017 / 0.005.
- F-A, pre-swerve replans at car speed >= 10 m/s steering > 0.3 rad: baseline 3 of 262, A1 0, A2 0, A3 0 (B1 0, B2 0). The baseline count is 3, so the '>= 90 %' statement rests on 3 replans.
- F-A, 4 s end point unchanged: A1 max position difference at 4 s = 0.000 m (A2: 20.26 m where the cap bites).
- F-A, slow replans (< 10 m/s) change by <= 0.02 rad: under A1 52 of 739 slow replans change by more than 0.02 rad (max increase 0.200 rad), 12 of them pre-swerve. A3 and B1 change none.
- F-A, plans needing > 5 m/s2 constant deceleration: 1 of 1020 replans (0.1 %).
- F-B, handover ticks < 0.05 rad: B1 0.019 / 0.289; +0.5 s: 0.429 / 0.292; +3.0 s: 0.430 / 0.270. Pick changed on 16/281 replans at car speed >= 10 m/s.
- F-C: see section 8 (C1 = tracking from step 0; C4 = steering-rate weight x10; C2 = lower longitudinal weight).

## 10. What this test bed cannot show

- Each row is one plan held for 1.0 s from the recorded state. After the first changed plan the car would be elsewhere and the model would see other images, so later replans of the recorded run are not what a fixed system would face. `pre_swerve` rows are the valid ones; `disturbed` rows are indicative only (the recorded car is already steering hard or off-road there, e.g. all highway ticks from +0.5 s).
- A held plan is not the 20 Hz stream: in the run the adapter re-anchors the plan every 50 ms and replans every 0.5 s.
- Nothing here measures collisions, comfort or progress of a fixed system; the 'cost side' columns come from the recorded (unfixed) run.
- Fix C variants were not tested on any turning scene above 7 m/s; the clean-scene tracking check is on near-straight driving.
- At car speed below ~2 m/s, steering angles of 0.2-0.5 rad occur in ordinary manoeuvring; '> 0.3 rad' is not a swerve criterion there (lateral distance from the plan path is in the CSV).

## Files

`testbed.csv` (scene x replan x variant), `fixA_alteration.csv`, `fixB_picks.csv`, `later_braking_cases.csv`, `long_hold.csv` (time series), `long_hold_summary.csv`, `fixC_sweep.csv` (kind = sweep / original_plan), `fixC_tracking.csv`, `metrics.json`, plots 1-8.
