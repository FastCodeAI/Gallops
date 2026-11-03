# T05 - anti-stall rule replayed on the recorded menus, and what the retrained scorer says about a pull-out

Offline, CPU only (`CUDA_VISIBLE_DEVICES=""`). Nothing under `alpasim/` or `adapters/` was edited. Scripts (this directory): `t05_common.py` (imports `part4_why_it_stays_stopped/t04_common.py`; adds clearance and the standing clock), `t05_menu.py` (per-proposal table, 30 scenes), `t05_rule.py` (Part 1), `t05_probe_inputs.py` / `t05_probe_paths.py` / `t05_probe.py` (Part 2), `t05_report.py` (tables, plots, this file - generated). Measured facts only.

## 0. Definitions as applied

- leave / prompt / box-free / usable escape: as in `part4_why_it_stays_stopped/README.md` (leave = end point >= 2.0 m; prompt = arc >= 0.5 m by 1 s and >= 1.5 m by 2 s; box-free = ego footprint at 0.1 s steps over 4 s never overlaps a recorded actor box at the same time stamp; margin 0.3 m = ego box grown 0.3 m on every side). Recomputed here for all 30 scenes; on the 8704 rows shared with `part4_why_it_stays_stopped/proposals.csv` leave, prompt, both box checks and the chosen flag are identical (0 differences).
- **Actors are a playback** and do not react. **The box guard uses the recorded actor boxes, i.e. information a real system would have to get from perception; it does not come from the scorer.** The actor log ends at +16.8 s: for replans after +12.8 s the box check covers less than 4 s (`plan_box_checked_horizon_s`), and a controller check started after +14.8 s is box-checked for less than 2 s.
- **Rule R(T, d, guard)**: at a replan, if the car has been below 0.5 m/s for >= T seconds and the chosen plan is a stay (end point < 2.0 m), take the leave+prompt proposal (passing the guard) with the highest final score (`pdm_score`); switch if its final score >= chosen score - d.
- **Standing clock.** Main (`incl_pre_handover`): continuous time below 0.5 m/s in the controller-state rows of the run, which start 3.0 s before handover (there the state is the simulator-reported replay), so at most 3.0 s of pre-handover standing is credited. Variant (`from_handover`): the same, capped at the time since handover. Both are in the CSVs; tables below are the main clock unless stated.
- **Open loop**: after a first firing the real car would move and later menus would differ. Per scene the FIRST firing is reported; `n_replans_rule_fires_recorded_menus` counts the replans at which the rule fires on the recorded (still-standing) menus and is indicative only.
- **Controller check**: standalone AlpaSim controller + vehicle model (`t02_common.Sim`), teacher-forced from the session start to the firing tick, then the switched-to plan held 2.0 s (`t03_common.branch`).
- **Labels** (by rule): `unsafe go` = the plan overlaps a recorded actor box at 0 m margin within the checked horizon, or the driven footprint overlaps one within 2 s; else `wanted go` = human above 0.5 m/s at that time or exceeding it within 2 s; else `premature go` (human standing for more than 2 s more, or never moving again in the clip).
- **Clearance** = minimum distance between the ego box and any recorded actor box over the 41 samples (0 = overlap).

## Part 1 - anti-stall rule
### 1.1 Stationary episodes in the 30 runs (0.1.6)
Every run of >= 1.0 s below 0.5 m/s that extends past handover (start can be negative = already standing in the replayed part). 21 of 30 scenes have none. `ahead_*` = nearest actor box ahead inside the ego-width corridor, gap from the front bumper, at the episode start / middle; `episode_class` by rule from the human's path length during the episode.

| scene | start_s | end_s | post_handover_duration_s | runs_to_end_of_clip | car_path_m | human_path_m | human_speed_at_start_mps | human_starts_moving_after_s | human_share_below_0p5 | human_max_speed_mps | n_replans | n_replans_chosen_stay | ahead_gap_at_start_m | ahead_label_at_start | ahead_gap_mid_m | ahead_label_mid | nearest_box_mid_m | nearest_label_mid | episode_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 13.2 | 16.75 | 3.55 | True | 0.27 | 77.48 | 21.98 | 0.09 | 0 | 22.18 | 7 | 6 |  |  |  |  | 73.81 | automobile | human drives off (human path >= car path + 2 m) |
| 5f064b02 | -3 | 1.05 | 1.05 | False | 0.21 | 0.01 | 0.01 | 3.38 | 1 | 0.01 | 3 | 0 | 11.94 | automobile | 12.14 | automobile | 1.54 | automobile | human also standing (human path < 2 m) |
| 8023dce5 | 12.6 | 16.8 | 4.2 | True | 0.16 | 0.05 | 0.01 |  | 1 | 0.11 | 8 | 8 | 0 | heavy_truck | 3.91 | automobile | 1.05 | automobile | human also standing (human path < 2 m) |
| acd22430 | 12.8 | 16.75 | 3.95 | True | 1.26 | 92.67 | 22.77 | 0.08 | 0 | 24.1 | 8 | 2 |  |  |  |  | 215.5 | automobile | human drives off (human path >= car path + 2 m) |
| b44a4a44 | -3 | 16.75 | 16.75 | True | 0.07 | 99.49 | 0 | 2.98 | 0.17 | 15.62 | 34 | 34 | 2.38 | automobile | 0 | person | 0 | person | human drives off (human path >= car path + 2 m) |
| bae463d3 | 11.75 | 16.8 | 5.05 | True | 0.66 | 39.42 | 8.9 | 0.04 | 0 | 9.08 | 10 | 6 |  |  |  |  | 19.79 | automobile | human drives off (human path >= car path + 2 m) |
| bd07fb91 | 14 | 16.75 | 2.75 | True | 0.8 | 8.77 | 1.2 | 0.1 | 0 | 5.1 | 6 | 0 |  |  |  |  | 5.36 | automobile | human drives off (human path >= car path + 2 m) |
| d2715023 | -3 | 16.75 | 16.75 | True | 1.22 | 44.14 | 0.07 | 0.99 | 0.14 | 5.37 | 34 | 34 | 2.26 | automobile | 1.6 | automobile | 1.61 | automobile | human drives off (human path >= car path + 2 m) |
| e3ed554f | -3 | 0.55 | 0.55 | False | 0.22 | 0.21 | 0.3 | 0.67 | 1 | 0.49 | 2 | 0 | 3.88 | automobile | 4.16 | automobile | 1.33 | automobile | human also standing (human path < 2 m) |
| e3ed554f | 7.25 | 8.4 | 1.15 | False | 0.4 | 1.74 | 1.55 | 0.02 | 0 | 1.55 | 2 | 0 | 4.36 | automobile | 4.42 | automobile | 0.92 | automobile | human also standing (human path < 2 m) |
| e3ed554f | 9.45 | 15.1 | 5.65 | False | 0.81 | 1.12 | 0.94 | 0.02 | 0.82 | 0.94 | 12 | 6 | 3.82 | automobile | 3.38 | automobile | 1.11 | automobile | human also standing (human path < 2 m) |

Counts: 11 episodes in 9 scenes; by class: {'human drives off (human path >= car path + 2 m)': 6, 'human also standing (human path < 2 m)': 5}. Replans with the car standing >= 3 s and a chosen stay (the rule's eligible set at T = 3): 74 in all 30 scenes, of which 68 on the two stuck scenes.

### 1.2 Grid (main clock): scenes with a first firing, split wanted / premature / unsafe
Cell = `fired: W/P/U`.

**guard none**

| T_s \ d | d=0.1 | d=0.2 | d=0.3 | d=0.5 | d=1.0 |
|---|---|---|---|---|---|
| 1 | 3: 1/1/1 | 6: 3/2/1 | 7: 3/3/1 | 7: 3/3/1 | 7: 3/3/1 |
| 2 | 3: 1/1/1 | 5: 2/2/1 | 7: 3/3/1 | 7: 3/3/1 | 7: 3/3/1 |
| 3 | 2: 1/0/1 | 3: 1/1/1 | 7: 3/3/1 | 7: 3/3/1 | 7: 3/3/1 |
| 5 | 1: 0/0/1 | 1: 0/0/1 | 1: 1/0/0 | 1: 1/0/0 | 2: 1/0/1 |

**guard G_box0**

| T_s \ d | d=0.1 | d=0.2 | d=0.3 | d=0.5 | d=1.0 |
|---|---|---|---|---|---|
| 1 | 3: 2/1/0 | 6: 4/2/0 | 6: 3/3/0 | 6: 3/3/0 | 7: 4/3/0 |
| 2 | 3: 2/1/0 | 5: 3/2/0 | 6: 3/3/0 | 6: 3/3/0 | 7: 4/3/0 |
| 3 | 2: 2/0/0 | 3: 2/1/0 | 6: 3/3/0 | 6: 3/3/0 | 7: 4/3/0 |
| 5 | 1: 1/0/0 | 1: 1/0/0 | 1: 1/0/0 | 1: 1/0/0 | 2: 2/0/0 |

**guard G_box03**

| T_s \ d | d=0.1 | d=0.2 | d=0.3 | d=0.5 | d=1.0 |
|---|---|---|---|---|---|
| 1 | 3: 2/1/0 | 6: 4/2/0 | 6: 4/2/0 | 6: 4/2/0 | 7: 5/2/0 |
| 2 | 3: 2/1/0 | 5: 3/2/0 | 6: 4/2/0 | 6: 4/2/0 | 7: 5/2/0 |
| 3 | 2: 2/0/0 | 3: 2/1/0 | 6: 4/2/0 | 6: 4/2/0 | 7: 5/2/0 |
| 5 | 1: 1/0/0 | 1: 1/0/0 | 1: 1/0/0 | 1: 1/0/0 | 2: 2/0/0 |

First-firing time (s after handover) of the two stuck scenes (blank = never fires):

**d2715023, clock incl_pre_handover**

| guard, T | d=0.1 | d=0.2 | d=0.3 | d=0.5 | d=1.0 |
|---|---|---|---|---|---|
| G_box0, T=1 |  |  |  |  | 0.5 |
| G_box0, T=2 |  |  |  |  | 0.5 |
| G_box0, T=3 |  |  |  |  | 0.5 |
| G_box0, T=5 |  |  |  |  | 2 |
| G_box03, T=1 |  |  |  |  | 15.5 |
| G_box03, T=2 |  |  |  |  | 15.5 |
| G_box03, T=3 |  |  |  |  | 15.5 |
| G_box03, T=5 |  |  |  |  | 15.5 |
| none, T=1 |  |  | 0 | 0 | 0 |
| none, T=2 |  |  | 0 | 0 | 0 |
| none, T=3 |  |  | 0 | 0 | 0 |
| none, T=5 |  |  |  |  | 2 |

**d2715023, clock from_handover**

| guard, T | d=0.1 | d=0.2 | d=0.3 | d=0.5 | d=1.0 |
|---|---|---|---|---|---|
| G_box0, T=1 |  |  |  |  | 1 |
| G_box0, T=2 |  |  |  |  | 2 |
| G_box0, T=3 |  |  |  |  | 3.5 |
| G_box0, T=5 |  |  |  |  | 5.5 |
| G_box03, T=1 |  |  |  |  | 15.5 |
| G_box03, T=2 |  |  |  |  | 15.5 |
| G_box03, T=3 |  |  |  |  | 15.5 |
| G_box03, T=5 |  |  |  |  | 15.5 |
| none, T=1 |  |  |  | 1 | 1 |
| none, T=2 |  |  |  |  | 2 |
| none, T=3 |  |  |  |  | 3 |
| none, T=5 |  |  |  |  | 5 |

**b44a4a44, clock incl_pre_handover**

| guard, T | d=0.1 | d=0.2 | d=0.3 | d=0.5 | d=1.0 |
|---|---|---|---|---|---|
| G_box0, T=1 | 11.5 | 9.5 | 0 | 0 | 0 |
| G_box0, T=2 | 11.5 | 9.5 | 0 | 0 | 0 |
| G_box0, T=3 | 11.5 | 9.5 | 0 | 0 | 0 |
| G_box0, T=5 | 11.5 | 9.5 | 2 | 2 | 2 |
| G_box03, T=1 | 12 | 12 | 2.5 | 2.5 | 2.5 |
| G_box03, T=2 | 12 | 12 | 2.5 | 2.5 | 2.5 |
| G_box03, T=3 | 12 | 12 | 2.5 | 2.5 | 2.5 |
| G_box03, T=5 | 12 | 12 | 2.5 | 2.5 | 2.5 |
| none, T=1 | 6 | 4 | 0 | 0 | 0 |
| none, T=2 | 6 | 4 | 0 | 0 | 0 |
| none, T=3 | 6 | 4 | 0 | 0 | 0 |
| none, T=5 | 6 | 4 | 2 | 2 | 2 |

**b44a4a44, clock from_handover**

| guard, T | d=0.1 | d=0.2 | d=0.3 | d=0.5 | d=1.0 |
|---|---|---|---|---|---|
| G_box0, T=1 | 11.5 | 9.5 | 1 | 1 | 1 |
| G_box0, T=2 | 11.5 | 9.5 | 2 | 2 | 2 |
| G_box0, T=3 | 11.5 | 9.5 | 3 | 3 | 3 |
| G_box0, T=5 | 11.5 | 9.5 | 9.5 | 5 | 5 |
| G_box03, T=1 | 12 | 12 | 2.5 | 2.5 | 2.5 |
| G_box03, T=2 | 12 | 12 | 2.5 | 2.5 | 2.5 |
| G_box03, T=3 | 12 | 12 | 3 | 3 | 3 |
| G_box03, T=5 | 12 | 12 | 10.5 | 10 | 10 |
| none, T=1 | 6 | 4 | 1 | 1 | 1 |
| none, T=2 | 6 | 4 | 2 | 2 | 2 |
| none, T=3 | 6 | 4 | 3 | 3 | 3 |
| none, T=5 | 6 | 5 | 5 | 5 | 5 |

Scene ids per cell and the `from_handover` clock: `antistall_grid.csv`. Same grid, `from_handover` clock:

**guard none** (from_handover)

| T_s \ d | d=0.1 | d=0.2 | d=0.3 | d=0.5 | d=1.0 |
|---|---|---|---|---|---|
| 1 | 3: 1/1/1 | 6: 3/2/1 | 6: 4/2/0 | 7: 4/2/1 | 7: 4/2/1 |
| 2 | 3: 1/1/1 | 5: 2/2/1 | 6: 4/2/0 | 6: 4/2/0 | 7: 4/2/1 |
| 3 | 2: 1/0/1 | 3: 1/1/1 | 6: 4/2/0 | 6: 4/2/0 | 7: 4/2/1 |
| 5 | 1: 0/0/1 | 1: 0/0/1 | 1: 0/0/1 | 1: 0/0/1 | 2: 0/0/2 |

**guard G_box0** (from_handover)

| T_s \ d | d=0.1 | d=0.2 | d=0.3 | d=0.5 | d=1.0 |
|---|---|---|---|---|---|
| 1 | 3: 2/1/0 | 6: 4/2/0 | 6: 4/2/0 | 6: 4/2/0 | 7: 5/2/0 |
| 2 | 3: 2/1/0 | 5: 3/2/0 | 6: 4/2/0 | 6: 4/2/0 | 7: 5/2/0 |
| 3 | 2: 2/0/0 | 3: 2/1/0 | 6: 4/2/0 | 6: 4/2/0 | 7: 4/2/1 |
| 5 | 1: 1/0/0 | 1: 1/0/0 | 1: 1/0/0 | 1: 1/0/0 | 2: 1/0/1 |

**guard G_box03** (from_handover)

| T_s \ d | d=0.1 | d=0.2 | d=0.3 | d=0.5 | d=1.0 |
|---|---|---|---|---|---|
| 1 | 3: 2/1/0 | 6: 4/2/0 | 6: 4/2/0 | 6: 4/2/0 | 7: 5/2/0 |
| 2 | 3: 2/1/0 | 5: 3/2/0 | 6: 4/2/0 | 6: 4/2/0 | 7: 5/2/0 |
| 3 | 2: 2/0/0 | 3: 2/1/0 | 6: 4/2/0 | 6: 4/2/0 | 7: 5/2/0 |
| 5 | 1: 1/0/0 | 1: 1/0/0 | 1: 1/0/0 | 1: 1/0/0 | 2: 2/0/0 |

### 1.3 Replans where the car and the human both stand (other 28 scenes)
Eligible = car standing >= T, chosen plan a stay, human below 0.5 m/s at that time, scene not one of the two stuck ones. The count does not depend on the clock or the guard in this data.

| T_s | d | other28_human_standing_eligible_replans | other28_human_standing_firing_replans | other28_human_standing_firing_share | other28_human_standing_firings_plan_overlaps_0m | other28_human_standing_firings_plan_overlaps_0p3m | other28_human_standing_firing_scenes |
|---|---|---|---|---|---|---|---|
| 1 | 0.1 | 10 | 1 | 0.1 | 0 | 0 | e3ed554f |
| 1 | 0.2 | 10 | 6 | 0.6 | 0 | 0 | 8023dce5 e3ed554f |
| 1 | 0.3 | 10 | 10 | 1 | 0 | 0 | 8023dce5 e3ed554f |
| 1 | 0.5 | 10 | 10 | 1 | 0 | 0 | 8023dce5 e3ed554f |
| 1 | 1 | 10 | 10 | 1 | 0 | 0 | 8023dce5 e3ed554f |
| 2 | 0.1 | 6 | 1 | 0.17 | 0 | 0 | e3ed554f |
| 2 | 0.2 | 6 | 3 | 0.5 | 0 | 0 | 8023dce5 e3ed554f |
| 2 | 0.3 | 6 | 6 | 1 | 0 | 0 | 8023dce5 e3ed554f |
| 2 | 0.5 | 6 | 6 | 1 | 0 | 0 | 8023dce5 e3ed554f |
| 2 | 1 | 6 | 6 | 1 | 0 | 0 | 8023dce5 e3ed554f |
| 3 | 0.1 | 3 | 0 | 0 | 0 | 0 |  |
| 3 | 0.2 | 3 | 1 | 0.33 | 0 | 0 | e3ed554f |
| 3 | 0.3 | 3 | 3 | 1 | 0 | 0 | 8023dce5 e3ed554f |
| 3 | 0.5 | 3 | 3 | 1 | 0 | 0 | 8023dce5 e3ed554f |
| 3 | 1 | 3 | 3 | 1 | 0 | 0 | 8023dce5 e3ed554f |
| 5 | 0.1 | 0 | 0 |  | 0 | 0 |  |
| 5 | 0.2 | 0 | 0 |  | 0 | 0 |  |
| 5 | 0.3 | 0 | 0 |  | 0 | 0 |  |
| 5 | 0.5 | 0 | 0 |  | 0 | 0 |  |
| 5 | 1 | 0 | 0 |  | 0 | 0 |  |

### 1.4 Per-scene tables for selected settings
Shown: R(3, 0.3) with each guard under both clocks (requested), and R(5, 0.3, G_box0) and R(3, 0.1, G_box0) because in the grid they are the cells with zero unsafe and zero premature first firings at the two ends of the T and d axes. This is a selection for display, not a recommendation.

**R(T=3 s, d=0.3, none), clock incl_pre_handover** - 7 scenes fire; the other 23 never do.

| scene | first_firing_t_s | n_replans_rule_fires_recorded_menus | proposal | proposal_score_final | proposal_rank_final | chosen_stay_score_final | standing_s | human_state | human_speed_mps | human_starts_moving_in_s | plan_box_free | plan_first_overlap_s | plan_first_overlap_label | plan_box_free_margin0p3 | plan_first_overlap_margin0p3_s | plan_first_overlap_margin0p3_label | plan_min_clearance_m | plan_box_checked_horizon_s | moved_1s_m | moved_2s_m | peak_abs_steer_rad | driven_overlap_2s | driven_min_clearance_2s_m | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 16.5 | 1 | 6 | 0.67 | 14 | 0.92 | 3.3 | moving | 22.09 | 0.09 | True |  |  | True |  |  | inf | 0.3 | 0.5 | 2.86 | 0.61 | False | inf | wanted go |
| 8023dce5 | 16 | 2 | 19 | 0.65 | 12 | 0.91 | 3.4 | standing | 0.02 |  | True |  |  | True |  |  | 0.53 | 0.8 | 0.64 | 2.1 | 0.08 | False | 0.52 | premature go |
| acd22430 | 16.5 | 1 | 16 | 0.9 | 7 | 0.94 | 3.7 | moving | 24.01 | 0.08 | True |  |  | True |  |  | inf | 0.3 | 0.48 | 1.7 | 0.14 | False | inf | wanted go |
| b44a4a44 | 0 | 32 | 19 | 0.67 | 16 | 0.93 | 3 | standing | 0 | 2.98 | True |  |  | False | 2.3 | automobile | 0.14 | 4 | 0.43 | 1.56 | 0.15 | False | 0.55 | premature go |
| bae463d3 | 15 | 1 | 19 | 0.63 | 14 | 0.86 | 3.25 | moving | 7.32 | 0.09 | True |  |  | True |  |  | 11.14 | 1.8 | 0.55 | 2.22 | 0.55 | False | 11.07 | wanted go |
| d2715023 | 0 | 1 | 19 | 0.66 | 15 | 0.92 | 3 | standing | 0.07 | 0.99 | False | 2.7 | automobile | False | 2.3 | automobile | 0 | 2.7 | 0.51 | 1.59 | 0.36 | False | 0.6 | unsafe go |
| e3ed554f | 12.5 | 1 | 19 | 0.87 | 16 | 0.99 | 3.05 | standing | 0 |  | True |  |  | True |  |  | 0.87 | 4 | 0.77 | 2.33 | 0.01 | False | 0.94 | premature go |

**R(T=3 s, d=0.3, G_box0), clock incl_pre_handover** - 6 scenes fire; the other 24 never do.

| scene | first_firing_t_s | n_replans_rule_fires_recorded_menus | proposal | proposal_score_final | proposal_rank_final | chosen_stay_score_final | standing_s | human_state | human_speed_mps | human_starts_moving_in_s | plan_box_free | plan_first_overlap_s | plan_first_overlap_label | plan_box_free_margin0p3 | plan_first_overlap_margin0p3_s | plan_first_overlap_margin0p3_label | plan_min_clearance_m | plan_box_checked_horizon_s | moved_1s_m | moved_2s_m | peak_abs_steer_rad | driven_overlap_2s | driven_min_clearance_2s_m | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 16.5 | 1 | 6 | 0.67 | 14 | 0.92 | 3.3 | moving | 22.09 | 0.09 | True |  |  | True |  |  | inf | 0.3 | 0.5 | 2.86 | 0.61 | False | inf | wanted go |
| 8023dce5 | 16 | 2 | 19 | 0.65 | 12 | 0.91 | 3.4 | standing | 0.02 |  | True |  |  | True |  |  | 0.53 | 0.8 | 0.64 | 2.1 | 0.08 | False | 0.52 | premature go |
| acd22430 | 16.5 | 1 | 16 | 0.9 | 7 | 0.94 | 3.7 | moving | 24.01 | 0.08 | True |  |  | True |  |  | inf | 0.3 | 0.48 | 1.7 | 0.14 | False | inf | wanted go |
| b44a4a44 | 0 | 23 | 19 | 0.67 | 16 | 0.93 | 3 | standing | 0 | 2.98 | True |  |  | False | 2.3 | automobile | 0.14 | 4 | 0.43 | 1.56 | 0.15 | False | 0.55 | premature go |
| bae463d3 | 15 | 1 | 19 | 0.63 | 14 | 0.86 | 3.25 | moving | 7.32 | 0.09 | True |  |  | True |  |  | 11.14 | 1.8 | 0.55 | 2.22 | 0.55 | False | 11.07 | wanted go |
| e3ed554f | 12.5 | 1 | 19 | 0.87 | 16 | 0.99 | 3.05 | standing | 0 |  | True |  |  | True |  |  | 0.87 | 4 | 0.77 | 2.33 | 0.01 | False | 0.94 | premature go |

**R(T=3 s, d=0.3, G_box03), clock incl_pre_handover** - 6 scenes fire; the other 24 never do.

| scene | first_firing_t_s | n_replans_rule_fires_recorded_menus | proposal | proposal_score_final | proposal_rank_final | chosen_stay_score_final | standing_s | human_state | human_speed_mps | human_starts_moving_in_s | plan_box_free | plan_first_overlap_s | plan_first_overlap_label | plan_box_free_margin0p3 | plan_first_overlap_margin0p3_s | plan_first_overlap_margin0p3_label | plan_min_clearance_m | plan_box_checked_horizon_s | moved_1s_m | moved_2s_m | peak_abs_steer_rad | driven_overlap_2s | driven_min_clearance_2s_m | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 16.5 | 1 | 6 | 0.67 | 14 | 0.92 | 3.3 | moving | 22.09 | 0.09 | True |  |  | True |  |  | inf | 0.3 | 0.5 | 2.86 | 0.61 | False | inf | wanted go |
| 8023dce5 | 16 | 2 | 19 | 0.65 | 12 | 0.91 | 3.4 | standing | 0.02 |  | True |  |  | True |  |  | 0.53 | 0.8 | 0.64 | 2.1 | 0.08 | False | 0.52 | premature go |
| acd22430 | 16.5 | 1 | 16 | 0.9 | 7 | 0.94 | 3.7 | moving | 24.01 | 0.08 | True |  |  | True |  |  | inf | 0.3 | 0.48 | 1.7 | 0.14 | False | inf | wanted go |
| b44a4a44 | 2.5 | 13 | 19 | 0.69 | 17 | 0.9 | 5.5 | standing | 0.15 | 0.48 | True |  |  | True |  |  | 0.33 | 4 | 0.48 | 1.88 | 0.18 | False | 0.46 | wanted go |
| bae463d3 | 15 | 1 | 19 | 0.63 | 14 | 0.86 | 3.25 | moving | 7.32 | 0.09 | True |  |  | True |  |  | 11.14 | 1.8 | 0.55 | 2.22 | 0.55 | False | 11.07 | wanted go |
| e3ed554f | 12.5 | 1 | 19 | 0.87 | 16 | 0.99 | 3.05 | standing | 0 |  | True |  |  | True |  |  | 0.87 | 4 | 0.77 | 2.33 | 0.01 | False | 0.94 | premature go |

**R(T=3 s, d=0.3, none), clock from_handover** - 6 scenes fire; the other 24 never do.

| scene | first_firing_t_s | n_replans_rule_fires_recorded_menus | proposal | proposal_score_final | proposal_rank_final | chosen_stay_score_final | standing_s | human_state | human_speed_mps | human_starts_moving_in_s | plan_box_free | plan_first_overlap_s | plan_first_overlap_label | plan_box_free_margin0p3 | plan_first_overlap_margin0p3_s | plan_first_overlap_margin0p3_label | plan_min_clearance_m | plan_box_checked_horizon_s | moved_1s_m | moved_2s_m | peak_abs_steer_rad | driven_overlap_2s | driven_min_clearance_2s_m | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 16.5 | 1 | 6 | 0.67 | 14 | 0.92 | 3.3 | moving | 22.09 | 0.09 | True |  |  | True |  |  | inf | 0.3 | 0.5 | 2.86 | 0.61 | False | inf | wanted go |
| 8023dce5 | 16 | 2 | 19 | 0.65 | 12 | 0.91 | 3.4 | standing | 0.02 |  | True |  |  | True |  |  | 0.53 | 0.8 | 0.64 | 2.1 | 0.08 | False | 0.52 | premature go |
| acd22430 | 16.5 | 1 | 16 | 0.9 | 7 | 0.94 | 3.7 | moving | 24.01 | 0.08 | True |  |  | True |  |  | inf | 0.3 | 0.48 | 1.7 | 0.14 | False | inf | wanted go |
| b44a4a44 | 3 | 27 | 30 | 0.67 | 20 | 0.9 | 6 | moving | 0.62 | 0.08 | True |  |  | False | 1.6 | automobile | 0.23 | 4 | 0.74 | 3.14 | 0.14 | False | 0.16 | wanted go |
| bae463d3 | 15 | 1 | 19 | 0.63 | 14 | 0.86 | 3.25 | moving | 7.32 | 0.09 | True |  |  | True |  |  | 11.14 | 1.8 | 0.55 | 2.22 | 0.55 | False | 11.07 | wanted go |
| e3ed554f | 12.5 | 1 | 19 | 0.87 | 16 | 0.99 | 3.05 | standing | 0 |  | True |  |  | True |  |  | 0.87 | 4 | 0.77 | 2.33 | 0.01 | False | 0.94 | premature go |

**R(T=3 s, d=0.3, G_box0), clock from_handover** - 6 scenes fire; the other 24 never do.

| scene | first_firing_t_s | n_replans_rule_fires_recorded_menus | proposal | proposal_score_final | proposal_rank_final | chosen_stay_score_final | standing_s | human_state | human_speed_mps | human_starts_moving_in_s | plan_box_free | plan_first_overlap_s | plan_first_overlap_label | plan_box_free_margin0p3 | plan_first_overlap_margin0p3_s | plan_first_overlap_margin0p3_label | plan_min_clearance_m | plan_box_checked_horizon_s | moved_1s_m | moved_2s_m | peak_abs_steer_rad | driven_overlap_2s | driven_min_clearance_2s_m | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 16.5 | 1 | 6 | 0.67 | 14 | 0.92 | 3.3 | moving | 22.09 | 0.09 | True |  |  | True |  |  | inf | 0.3 | 0.5 | 2.86 | 0.61 | False | inf | wanted go |
| 8023dce5 | 16 | 2 | 19 | 0.65 | 12 | 0.91 | 3.4 | standing | 0.02 |  | True |  |  | True |  |  | 0.53 | 0.8 | 0.64 | 2.1 | 0.08 | False | 0.52 | premature go |
| acd22430 | 16.5 | 1 | 16 | 0.9 | 7 | 0.94 | 3.7 | moving | 24.01 | 0.08 | True |  |  | True |  |  | inf | 0.3 | 0.48 | 1.7 | 0.14 | False | inf | wanted go |
| b44a4a44 | 3 | 18 | 30 | 0.67 | 20 | 0.9 | 6 | moving | 0.62 | 0.08 | True |  |  | False | 1.6 | automobile | 0.23 | 4 | 0.74 | 3.14 | 0.14 | False | 0.16 | wanted go |
| bae463d3 | 15 | 1 | 19 | 0.63 | 14 | 0.86 | 3.25 | moving | 7.32 | 0.09 | True |  |  | True |  |  | 11.14 | 1.8 | 0.55 | 2.22 | 0.55 | False | 11.07 | wanted go |
| e3ed554f | 12.5 | 1 | 19 | 0.87 | 16 | 0.99 | 3.05 | standing | 0 |  | True |  |  | True |  |  | 0.87 | 4 | 0.77 | 2.33 | 0.01 | False | 0.94 | premature go |

**R(T=3 s, d=0.3, G_box03), clock from_handover** - 6 scenes fire; the other 24 never do.

| scene | first_firing_t_s | n_replans_rule_fires_recorded_menus | proposal | proposal_score_final | proposal_rank_final | chosen_stay_score_final | standing_s | human_state | human_speed_mps | human_starts_moving_in_s | plan_box_free | plan_first_overlap_s | plan_first_overlap_label | plan_box_free_margin0p3 | plan_first_overlap_margin0p3_s | plan_first_overlap_margin0p3_label | plan_min_clearance_m | plan_box_checked_horizon_s | moved_1s_m | moved_2s_m | peak_abs_steer_rad | driven_overlap_2s | driven_min_clearance_2s_m | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 16.5 | 1 | 6 | 0.67 | 14 | 0.92 | 3.3 | moving | 22.09 | 0.09 | True |  |  | True |  |  | inf | 0.3 | 0.5 | 2.86 | 0.61 | False | inf | wanted go |
| 8023dce5 | 16 | 2 | 19 | 0.65 | 12 | 0.91 | 3.4 | standing | 0.02 |  | True |  |  | True |  |  | 0.53 | 0.8 | 0.64 | 2.1 | 0.08 | False | 0.52 | premature go |
| acd22430 | 16.5 | 1 | 16 | 0.9 | 7 | 0.94 | 3.7 | moving | 24.01 | 0.08 | True |  |  | True |  |  | inf | 0.3 | 0.48 | 1.7 | 0.14 | False | inf | wanted go |
| b44a4a44 | 3 | 12 | 20 | 0.62 | 28 | 0.9 | 6 | moving | 0.62 | 0.08 | True |  |  | True |  |  | 0.3 | 4 | 0.69 | 3.74 | 0.19 | False | 0.24 | wanted go |
| bae463d3 | 15 | 1 | 19 | 0.63 | 14 | 0.86 | 3.25 | moving | 7.32 | 0.09 | True |  |  | True |  |  | 11.14 | 1.8 | 0.55 | 2.22 | 0.55 | False | 11.07 | wanted go |
| e3ed554f | 12.5 | 1 | 19 | 0.87 | 16 | 0.99 | 3.05 | standing | 0 |  | True |  |  | True |  |  | 0.87 | 4 | 0.77 | 2.33 | 0.01 | False | 0.94 | premature go |

**R(T=5 s, d=0.3, G_box0), clock incl_pre_handover** - 1 scenes fire; the other 29 never do.

| scene | first_firing_t_s | n_replans_rule_fires_recorded_menus | proposal | proposal_score_final | proposal_rank_final | chosen_stay_score_final | standing_s | human_state | human_speed_mps | human_starts_moving_in_s | plan_box_free | plan_first_overlap_s | plan_first_overlap_label | plan_box_free_margin0p3 | plan_first_overlap_margin0p3_s | plan_first_overlap_margin0p3_label | plan_min_clearance_m | plan_box_checked_horizon_s | moved_1s_m | moved_2s_m | peak_abs_steer_rad | driven_overlap_2s | driven_min_clearance_2s_m | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b44a4a44 | 2 | 20 | 19 | 0.67 | 17 | 0.91 | 5 | standing | 0.01 | 0.98 | True |  |  | False | 2.1 | automobile | 0.13 | 4 | 0.53 | 2.02 | 0.13 | False | 0.27 | wanted go |

**R(T=3 s, d=0.1, G_box0), clock incl_pre_handover** - 2 scenes fire; the other 28 never do.

| scene | first_firing_t_s | n_replans_rule_fires_recorded_menus | proposal | proposal_score_final | proposal_rank_final | chosen_stay_score_final | standing_s | human_state | human_speed_mps | human_starts_moving_in_s | plan_box_free | plan_first_overlap_s | plan_first_overlap_label | plan_box_free_margin0p3 | plan_first_overlap_margin0p3_s | plan_first_overlap_margin0p3_label | plan_min_clearance_m | plan_box_checked_horizon_s | moved_1s_m | moved_2s_m | peak_abs_steer_rad | driven_overlap_2s | driven_min_clearance_2s_m | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acd22430 | 16.5 | 1 | 16 | 0.9 | 7 | 0.94 | 3.7 | moving | 24.01 | 0.08 | True |  |  | True |  |  | inf | 0.3 | 0.48 | 1.7 | 0.14 | False | inf | wanted go |
| b44a4a44 | 11.5 | 3 | 19 | 0.84 | 12 | 0.94 | 14.5 | moving | 11.18 | 0.08 | True |  |  | False | 2.6 | automobile | 0.29 | 4 | 0.52 | 1.77 | 0.18 | False | 0.91 | wanted go |

First firings whose plan box check covers less than 4 s because the actor log ends (their label rests on the shorter check):

| scene | first_firing_t_s | plan_box_checked_horizon_s |
|---|---|---|
| acd22430 | 16.5 | 0.3 |
| b44a4a44 | 6 | 0.6 |
| 04749bb9 | 15.5 | 1.3 |
| 8023dce5 | 14 | 2.8 |
| acd22430 | 15.5 | 1.3 |
| bae463d3 | 13 | 3.8 |
| b44a4a44 | 4 | 2.6 |
| 04749bb9 | 14.5 | 2.3 |
| d2715023 | 1 | 2.6 |
| d2715023 | 15.5 | 1.3 |
| 8023dce5 | 15 | 1.8 |
| bae463d3 | 15 | 1.8 |
| bae463d3 | 14.5 | 2.3 |
| bae463d3 | 14 | 2.8 |
| d2715023 | 2 | 2.4 |
| 04749bb9 | 16.5 | 0.3 |
| 8023dce5 | 16 | 0.8 |
| d2715023 | 3 | 1.6 |
| b44a4a44 | 5 | 1.6 |
| d2715023 | 5 | 1.7 |
| d2715023 | 0 | 2.7 |

## Part 2 - the retrained scorer and a pull-out
### 2.1 What the scorer takes as input (code)

- The score head reads no waypoints. `Scorer.forward(proposals, bev_feature)` uses `proposals` only for its shape (`adapters/drivejepa/src/navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py:37-39`); the per-proposal input is `bev_feature.reshape(B, 32, 8, -1).amax(-2)` - the element-wise maximum over the 8 poses of a 256-d feature per (slot, pose) - fed to one MLP with 6 outputs (`scorer.py:41-42`); the score is the sigmoid of the last output (`drive_jepa_model.py:85`).
- Those features are the generator's own query features: `bev_feature = hist_encoding(ego_status) + init_feature.weight` - one learned embedding per (slot, pose), 32 x 8 (`drive_jepa_model.py:33, 61-62`) - passed 4 times (`drive_jepa_config.py:14` `ref_num=4`) through ONE shared `Traj_refiner` (`drive_jepa_model.py:35-36, 65-67`).
- Each pass decodes waypoints from the features with an MLP (`traj_refiner.py:24`) and then refines the features with `Bev_refiner(proposals, bev_feature, image_feature)` (`traj_refiner.py:26-27`), where the just-decoded waypoints enter only as detached reference points (`bevformer/bev_refiner.py:112-114`): position `(xy + 32) / 64` for the self-attention and the four corners of the ego footprint at that pose, projected into the camera, as the sampling locations of the image cross-attention (`bevformer/encoder.py:234-253, 273-274`). The returned menu is the waypoints decoded in the 4th pass (`drive_jepa_model.py:69`); the scorer gets the features after the 4th refinement (`drive_jepa_model.py:71-73`).
- So: (1) the scorer consumes a per-slot learned feature from the generator, with image features sampled at the proposal's footprint; (2) it has no interface for a trajectory the generator did not produce; (3) a foreign path can only be put in as the reference points of a refinement pass, while the slot's query feature still comes from the slot's own earlier passes (and that feature is what the waypoint MLP decodes the slot's own path from). Such a score is a hybrid, not the scorer's opinion of the path alone. Neither case (a) nor case (b) of the task holds cleanly; both sets of outputs were produced.
- Final score = `(14 * S + 2 * S_c) / 16` with `S_c` in {0, 1} a two-frame comfort term computed from the proposal's simulated states against the previous plan (`drive_jepa_model.py:98-151`, formula at `:149`; adapter side `adapters/drivejepa016/momentum.py:234-296`).

### 2.2 Forward-pass gate
Step 1 (`t05_probe_inputs.py`, method of `s9_trial_inputs.py`: 0.1.6 planner, bridge worker on CPU, replans in time order from the session start): picks equal the logged ones on 32 of 32 replans run (through +5.0 s; 0 mismatches); worst waypoint difference against the stored S5 menus 0.0e+00 m. Step 2 (`t05_probe.py`, `MomentumDriveJEPA` loaded directly in the worker interpreter, same weights, with an instance-level `Traj_refiner.forward` and no replacement): against the values saved in step 1 -

| scene | t_after_handover_s | max_abs_waypoint_diff_m | max_abs_all_diff | max_abs_plain_score_diff | max_abs_final_score_diff | pick | logged_pick | momentum_request_present |
|---|---|---|---|---|---|---|---|---|
| d2715023 | 0 | 0 | 0 | 0 | 0 | 13 | 13 | True |
| d2715023 | 1 | 0 | 0 | 0 | 0 | 26 | 26 | True |
| d2715023 | 5 | 0 | 0 | 0 | 0 | 27 | 27 | True |
| b44a4a44 | 0 | 0 | 0 | 0 | 0 | 27 | 27 | True |
| b44a4a44 | 3 | 0 | 0 | 0 | 0 | 15 | 15 | True |
| b44a4a44 | 5 | 0 | 0 | 0 | 0 | 15 | 15 | True |

Injecting a slot's own final waypoints: last-pass mode changes its plain score by at most 0.0e+00; all-pass mode by up to 0.22 (the slot's waypoints of passes 1-3 differ from its final ones, so this mode is not an identity). Largest change of any OTHER slot's plain score caused by one injection: last-pass 0.0e+00, all-pass 0.16.

### 2.3 Injection probe (hybrid; see 2.1)
One path replaces the waypoints of ONE slot per forward pass. Mode `last`: in the 4th refinement pass only. Mode `all`: in all 4 passes. Four slots per tick: the chosen stay's slot, the base escape's slot, the best-scored late-go slot, the lowest-scored slot. Paths: (i) the human's own future from the tick in the car's frame (8 poses, 0.5 s, recorded heading relative to the car's); (ii) the human's pull-out from min(tick, human-start tick) placed at the car's pose; (iii) the base escape (best usable escape of that menu, or the best leave+prompt where the menu has no usable one) with its end shifted 0 / 0.5 / 1.0 / 1.5 / 2.0 m to the left, scaled by arc-length fraction; (iv) all poses at the origin; (v) the chosen stay's own waypoints. `control_no_replacement` rows in `scorer_probe.csv` are the unmodified menu. At +5 s the human is already 2.5 m (d2715023), 3.2 m (b44a4a44) away from the standing car, so path (i) starts that far ahead.

**d2715023, +0.0 s** - logged final score of the slots: chosen stay slot 13: 0.92 (plain 0.91), base escape slot 19: 0.66 (plain 0.75), best late go slot 9: 0.80 (plain 0.77), lowest ranked slot 20: 0.08 (plain 0.09); base = best leave+prompt (no usable escape on this menu). Last-pass mode:

| path | path_class | path_min_clearance_m | final, chosen stay slot (13) | final, base escape slot (19) | final, best late go slot (9) | final, lowest ranked slot (20) | plain, min..max over the 4 slots | final, all-pass mode, min..max |
|---|---|---|---|---|---|---|---|---|
| i_human_future_from_tick | leave/not-prompt/box-free(0.3) | 0.67 | 0.91 | 0.82 | 0.73 | 0.29 | 0.19..0.90 | 0.28..0.82 |
| ii_human_pullout_transplanted | leave/not-prompt/box-free(0.3) | 0.67 | 0.91 | 0.82 | 0.73 | 0.29 | 0.19..0.90 | 0.28..0.82 |
| iii_base_escape_shift_0.0m | leave/prompt/overlap | 0 | 0.77 | 0.66 | 0.46 | 0.1 | 0.12..0.88 | 0.08..0.74 |
| iii_base_escape_shift_0.5m | leave/prompt/overlap | 0 | 0.79 | 0.7 | 0.64 | 0.15 | 0.17..0.90 | 0.17..0.75 |
| iii_base_escape_shift_1.0m | leave/prompt/box-free(0.3) | 0.62 | 0.77 | 0.75 | 0.66 | 0.2 | 0.23..0.88 | 0.21..0.71 |
| iii_base_escape_shift_1.5m | leave/prompt/box-free(0.3) | 1.13 | 0.77 | 0.76 | 0.68 | 0.19 | 0.22..0.88 | 0.22..0.73 |
| iii_base_escape_shift_2.0m | leave/prompt/box-free(0.3) | 1.69 | 0.78 | 0.75 | 0.65 | 0.14 | 0.16..0.89 | 0.15..0.72 |
| iv_stay_zeros | stay/not-prompt/box-free(0.3) | 2.26 | 0.9 | 0.9 | 0.87 | 0.4 | 0.31..0.88 | 0.69..0.89 |
| v_chosen_stay_path | stay/not-prompt/box-free(0.3) | 1.84 | 0.92 | 0.93 | 0.92 | 0.46 | 0.38..0.92 | 0.72..0.92 |

**d2715023, +1.0 s** - logged final score of the slots: chosen stay slot 26: 0.95 (plain 0.95), base escape slot 7: 0.10 (plain 0.11), best late go slot 28: 0.78 (plain 0.75), lowest ranked slot 20: 0.06 (plain 0.07); base = best usable escape. Last-pass mode:

| path | path_class | path_min_clearance_m | final, chosen stay slot (26) | final, base escape slot (7) | final, best late go slot (28) | final, lowest ranked slot (20) | plain, min..max over the 4 slots | final, all-pass mode, min..max |
|---|---|---|---|---|---|---|---|---|
| i_human_future_from_tick | leave/prompt/box-free(0.3) | 0.67 | 0.79 | 0.15 | 0.6 | 0.12 | 0.14..0.91 | 0.04..0.53 |
| ii_human_pullout_transplanted | leave/not-prompt/box-free(0.3) | 0.63 | 0.78 | 0.11 | 0.49 | 0.09 | 0.10..0.89 | 0.10..0.52 |
| iii_base_escape_shift_0.0m | leave/prompt/box-free(0) | 0.05 | 0.41 | 0.1 | 0.17 | 0.07 | 0.08..0.47 | 0.03..0.19 |
| iii_base_escape_shift_0.5m | leave/prompt/box-free(0) | 0.31 | 0.57 | 0.08 | 0.21 | 0.06 | 0.07..0.65 | 0.03..0.25 |
| iii_base_escape_shift_1.0m | leave/prompt/box-free(0.3) | 0.53 | 0.66 | 0.13 | 0.33 | 0.11 | 0.13..0.75 | 0.06..0.30 |
| iii_base_escape_shift_1.5m | leave/prompt/box-free(0.3) | 0.74 | 0.71 | 0.21 | 0.44 | 0.17 | 0.20..0.81 | 0.13..0.43 |
| iii_base_escape_shift_2.0m | leave/prompt/box-free(0.3) | 0.95 | 0.76 | 0.17 | 0.52 | 0.15 | 0.18..0.86 | 0.08..0.56 |
| iv_stay_zeros | stay/not-prompt/box-free(0.3) | 2.12 | 0.92 | 0.46 | 0.9 | 0.39 | 0.30..0.91 | 0.60..0.91 |
| v_chosen_stay_path | stay/not-prompt/box-free(0.3) | 0.93 | 0.95 | 0.39 | 0.9 | 0.34 | 0.25..0.95 | 0.24..0.95 |

**d2715023, +5.0 s** - logged final score of the slots: chosen stay slot 27: 0.95 (plain 0.94), base escape slot 30: 0.16 (plain 0.18), best late go slot 9: 0.57 (plain 0.50), lowest ranked slot 4: 0.05 (plain 0.05); base = best leave+prompt (no usable escape on this menu). Last-pass mode:

| path | path_class | path_min_clearance_m | final, chosen stay slot (27) | final, base escape slot (30) | final, best late go slot (9) | final, lowest ranked slot (4) | plain, min..max over the 4 slots | final, all-pass mode, min..max |
|---|---|---|---|---|---|---|---|---|
| i_human_future_from_tick | leave/prompt/box-free(0) | 0.36 | 0.18 | 0.03 | 0.04 | 0.06 | 0.04..0.21 | 0.05..0.10 |
| ii_human_pullout_transplanted | leave/not-prompt/box-free(0) | 0.39 | 0.79 | 0.17 | 0.47 | 0.08 | 0.09..0.91 | 0.10..0.70 |
| iii_base_escape_shift_0.0m | leave/prompt/overlap | 0 | 0.74 | 0.16 | 0.34 | 0.05 | 0.06..0.85 | 0.05..0.55 |
| iii_base_escape_shift_0.5m | leave/prompt/overlap | 0 | 0.79 | 0.16 | 0.33 | 0.07 | 0.09..0.90 | 0.07..0.62 |
| iii_base_escape_shift_1.0m | leave/prompt/box-free(0) | 0.34 | 0.78 | 0.25 | 0.56 | 0.13 | 0.15..0.89 | 0.20..0.68 |
| iii_base_escape_shift_1.5m | leave/prompt/box-free(0.3) | 0.73 | 0.78 | 0.32 | 0.65 | 0.18 | 0.20..0.89 | 0.15..0.71 |
| iii_base_escape_shift_2.0m | leave/prompt/box-free(0.3) | 1.08 | 0.78 | 0.28 | 0.65 | 0.1 | 0.12..0.89 | 0.12..0.70 |
| iv_stay_zeros | stay/not-prompt/box-free(0.3) | 1.47 | 0.94 | 0.54 | 0.93 | 0.32 | 0.22..0.93 | 0.54..0.95 |
| v_chosen_stay_path | stay/not-prompt/box-free(0.3) | 1.45 | 0.95 | 0.68 | 0.94 | 0.36 | 0.27..0.94 | 0.46..0.95 |

**b44a4a44, +0.0 s** - logged final score of the slots: chosen stay slot 27: 0.93 (plain 0.92), base escape slot 19: 0.67 (plain 0.77), best late go slot 12: 0.87 (plain 0.85), lowest ranked slot 20: 0.15 (plain 0.17); base = best usable escape. Last-pass mode:

| path | path_class | path_min_clearance_m | final, chosen stay slot (27) | final, base escape slot (19) | final, best late go slot (12) | final, lowest ranked slot (20) | plain, min..max over the 4 slots | final, all-pass mode, min..max |
|---|---|---|---|---|---|---|---|---|
| i_human_future_from_tick | stay/not-prompt/box-free(0.3) | 0.75 | 0.88 | 0.86 | 0.86 | 0.55 | 0.49..0.86 | 0.75..0.86 |
| ii_human_pullout_transplanted | stay/not-prompt/box-free(0.3) | 0.75 | 0.88 | 0.85 | 0.86 | 0.55 | 0.49..0.86 | 0.75..0.86 |
| iii_base_escape_shift_0.0m | leave/prompt/box-free(0) | 0.14 | 0.71 | 0.67 | 0.7 | 0.37 | 0.42..0.81 | 0.52..0.69 |
| iii_base_escape_shift_0.5m | leave/prompt/overlap | 0 | 0.72 | 0.64 | 0.7 | 0.36 | 0.41..0.83 | 0.48..0.66 |
| iii_base_escape_shift_1.0m | leave/prompt/overlap | 0 | 0.68 | 0.62 | 0.64 | 0.42 | 0.48..0.78 | 0.45..0.61 |
| iii_base_escape_shift_1.5m | leave/prompt/overlap | 0 | 0.7 | 0.59 | 0.6 | 0.29 | 0.33..0.80 | 0.30..0.60 |
| iii_base_escape_shift_2.0m | leave/prompt/overlap | 0 | 0.74 | 0.53 | 0.6 | 0.17 | 0.19..0.84 | 0.21..0.59 |
| iv_stay_zeros | stay/not-prompt/box-free(0.3) | 1.1 | 0.93 | 0.91 | 0.91 | 0.5 | 0.43..0.92 | 0.75..0.93 |
| v_chosen_stay_path | stay/not-prompt/box-free(0.3) | 1.08 | 0.93 | 0.9 | 0.91 | 0.54 | 0.47..0.92 | 0.70..0.94 |

**b44a4a44, +3.0 s** - logged final score of the slots: chosen stay slot 15: 0.90 (plain 0.88), base escape slot 30: 0.67 (plain 0.77), best late go slot 26: 0.86 (plain 0.83), lowest ranked slot 4: 0.48 (plain 0.55); base = best usable escape. Last-pass mode:

| path | path_class | path_min_clearance_m | final, chosen stay slot (15) | final, base escape slot (30) | final, best late go slot (26) | final, lowest ranked slot (4) | plain, min..max over the 4 slots | final, all-pass mode, min..max |
|---|---|---|---|---|---|---|---|---|
| i_human_future_from_tick | leave/prompt/box-free(0.3) | 0.75 | 0.69 | 0.63 | 0.67 | 0.58 | 0.67..0.78 | 0.61..0.67 |
| ii_human_pullout_transplanted | leave/prompt/box-free(0.3) | 0.64 | 0.67 | 0.6 | 0.67 | 0.55 | 0.63..0.77 | 0.58..0.66 |
| iii_base_escape_shift_0.0m | leave/prompt/box-free(0) | 0.23 | 0.68 | 0.67 | 0.7 | 0.58 | 0.67..0.80 | 0.62..0.70 |
| iii_base_escape_shift_0.5m | leave/prompt/box-free(0.3) | 0.61 | 0.7 | 0.66 | 0.69 | 0.62 | 0.71..0.81 | 0.61..0.69 |
| iii_base_escape_shift_1.0m | leave/prompt/box-free(0) | 0.13 | 0.69 | 0.64 | 0.67 | 0.58 | 0.66..0.79 | 0.60..0.67 |
| iii_base_escape_shift_1.5m | leave/prompt/overlap | 0 | 0.69 | 0.65 | 0.66 | 0.61 | 0.70..0.79 | 0.61..0.66 |
| iii_base_escape_shift_2.0m | leave/prompt/overlap | 0 | 0.66 | 0.62 | 0.64 | 0.6 | 0.69..0.75 | 0.49..0.64 |
| iv_stay_zeros | stay/not-prompt/overlap | 0 | 0.9 | 0.86 | 0.89 | 0.72 | 0.68..0.88 | 0.80..0.91 |
| v_chosen_stay_path | stay/not-prompt/overlap | 0 | 0.9 | 0.86 | 0.89 | 0.73 | 0.70..0.88 | 0.82..0.91 |

**b44a4a44, +5.0 s** - logged final score of the slots: chosen stay slot 15: 0.89 (plain 0.88), base escape slot 4: 0.48 (plain 0.55), best late go slot 16: 0.83 (plain 0.80), lowest ranked slot 18: 0.69 (plain 0.79); base = best usable escape. Last-pass mode:

| path | path_class | path_min_clearance_m | final, chosen stay slot (15) | final, base escape slot (4) | final, best late go slot (16) | final, lowest ranked slot (18) | plain, min..max over the 4 slots | final, all-pass mode, min..max |
|---|---|---|---|---|---|---|---|---|
| i_human_future_from_tick | leave/prompt/box-free(0.3) | 0.62 | 0.61 | 0.5 | 0.6 | 0.57 | 0.57..0.70 | 0.54..0.59 |
| ii_human_pullout_transplanted | leave/prompt/overlap | 0 | 0.71 | 0.61 | 0.71 | 0.69 | 0.69..0.82 | 0.60..0.68 |
| iii_base_escape_shift_0.0m | leave/prompt/box-free(0) | 0.02 | 0.67 | 0.48 | 0.64 | 0.6 | 0.55..0.76 | 0.51..0.61 |
| iii_base_escape_shift_0.5m | leave/prompt/box-free(0) | 0.08 | 0.67 | 0.63 | 0.69 | 0.69 | 0.71..0.78 | 0.62..0.66 |
| iii_base_escape_shift_1.0m | leave/prompt/overlap | 0 | 0.67 | 0.64 | 0.68 | 0.69 | 0.73..0.79 | 0.63..0.64 |
| iii_base_escape_shift_1.5m | leave/prompt/overlap | 0 | 0.66 | 0.6 | 0.65 | 0.65 | 0.69..0.75 | 0.58..0.62 |
| iii_base_escape_shift_2.0m | leave/prompt/overlap | 0 | 0.68 | 0.62 | 0.67 | 0.68 | 0.71..0.78 | 0.58..0.62 |
| iv_stay_zeros | stay/not-prompt/overlap | 0 | 0.89 | 0.77 | 0.89 | 0.91 | 0.74..0.89 | 0.82..0.91 |
| v_chosen_stay_path | stay/not-prompt/overlap | 0 | 0.89 | 0.78 | 0.89 | 0.91 | 0.75..0.89 | 0.83..0.92 |

Slot dependence (last-pass mode, plain score): mean range over the 4 slots for one and the same injected path 0.45; mean range over the 9 paths inside one slot 0.30. Spearman between the injected score and the slot's own original score 0.78 (n = 216).

Key numbers (last-pass mode, final score; min and max over the 4 slots): `scorer_probe_key.csv`.

| scene | t_after_handover_s | logged_chosen_stay_final | logged_base_escape_final | human_future_class | human_future_final_min | human_future_final_max | human_pullout_transplanted_class | human_pullout_transplanted_final_min | human_pullout_transplanted_final_max | stay_zeros_final_min | stay_zeros_final_max | base_shift_2m_class | base_shift_2m_final_min | base_shift_2m_final_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d2715023 | 0 | 0.92 | 0.66 | leave/not-prompt/box-free(0.3) | 0.29 | 0.91 | leave/not-prompt/box-free(0.3) | 0.29 | 0.91 | 0.4 | 0.9 | leave/prompt/box-free(0.3) | 0.14 | 0.78 |
| d2715023 | 1 | 0.95 | 0.1 | leave/prompt/box-free(0.3) | 0.12 | 0.79 | leave/not-prompt/box-free(0.3) | 0.09 | 0.78 | 0.39 | 0.92 | leave/prompt/box-free(0.3) | 0.15 | 0.76 |
| d2715023 | 5 | 0.95 | 0.16 | leave/prompt/box-free(0) | 0.03 | 0.18 | leave/not-prompt/box-free(0) | 0.08 | 0.79 | 0.32 | 0.94 | leave/prompt/box-free(0.3) | 0.1 | 0.78 |
| b44a4a44 | 0 | 0.93 | 0.67 | stay/not-prompt/box-free(0.3) | 0.55 | 0.88 | stay/not-prompt/box-free(0.3) | 0.55 | 0.88 | 0.5 | 0.93 | leave/prompt/overlap | 0.17 | 0.74 |
| b44a4a44 | 3 | 0.9 | 0.67 | leave/prompt/box-free(0.3) | 0.58 | 0.69 | leave/prompt/box-free(0.3) | 0.55 | 0.67 | 0.72 | 0.9 | leave/prompt/overlap | 0.6 | 0.66 |
| b44a4a44 | 5 | 0.89 | 0.48 | leave/prompt/box-free(0.3) | 0.5 | 0.61 | leave/prompt/overlap | 0.61 | 0.71 | 0.77 | 0.91 | leave/prompt/overlap | 0.62 | 0.68 |

### 2.4 Score against the proposals' own geometry (no injection)
All 32 proposals of all 34 post-handover replans, both scenes, 0.1.6 and 0.1.4b (`scorer_geometry.csv`); Spearman rank correlations (`scorer_geometry_spearman.csv`):

| set | scene | version | subset | n | spearman_final_vs_end_lateral_left_m | spearman_plain_vs_end_lateral_left_m | spearman_final_vs_min_clearance_m | spearman_plain_vs_min_clearance_m | spearman_final_vs_arc_4s_m | spearman_plain_vs_arc_4s_m |
|---|---|---|---|---|---|---|---|---|---|---|
| stuck scene | b44a4a44 | 0.1.6 | all proposals | 1088 | -0.91 | -0.9 | 0.17 | 0.18 | -0.9 | -0.88 |
| stuck scene | b44a4a44 | 0.1.6 | leave+prompt only | 243 | -0.62 | -0.6 | 0.14 | 0.16 | -0.5 | -0.49 |
| stuck scene | b44a4a44 | 0.1.6 | leave only | 812 | -0.81 | -0.78 | 0.12 | 0.13 | -0.77 | -0.73 |
| stuck scene | d2715023 | 0.1.6 | all proposals | 1088 | -0.82 | -0.79 | 0.71 | 0.7 | -0.89 | -0.87 |
| stuck scene | d2715023 | 0.1.6 | leave+prompt only | 203 | 0.13 | 0.14 | -0.02 | -0.05 | -0.17 | -0.15 |
| stuck scene | d2715023 | 0.1.6 | leave only | 705 | -0.41 | -0.33 | 0.32 | 0.3 | -0.66 | -0.59 |
| stuck scene | d2715023 | 0.1.4b | all proposals | 1088 | -0.68 | -0.65 | 0.78 | 0.75 | -0.91 | -0.84 |
| stuck scene | d2715023 | 0.1.4b | leave+prompt only | 324 | -0.52 | -0.38 | 0.02 | -0.08 | -0.69 | -0.41 |
| stuck scene | d2715023 | 0.1.4b | leave only | 638 | -0.51 | -0.36 | 0.12 | -0.05 | -0.69 | -0.38 |
| stuck scene | b44a4a44 | 0.1.4b | all proposals | 1088 | -0.43 | -0.4 | 0.21 | 0.21 | -0.73 | -0.7 |
| stuck scene | b44a4a44 | 0.1.4b | leave+prompt only | 341 | 0.78 | 0.81 | 0.07 | 0.08 | 0.39 | 0.44 |
| stuck scene | b44a4a44 | 0.1.4b | leave only | 849 | 0.1 | 0.15 | 0.16 | 0.17 | -0.44 | -0.39 |
| comparable replan in another scene | e3ed554f | 0.1.6 | all proposals | 96 | 0.12 | 0.25 | 0.75 | 0.74 | -0.73 | -0.55 |
| comparable replan in another scene | e3ed554f | 0.1.6 | leave+prompt only | 42 | 0.71 | 0.71 | 0.55 | 0.55 | -0.13 | -0.13 |
| comparable replan in another scene | e3ed554f | 0.1.6 | leave only | 74 | 0.42 | 0.54 | 0.68 | 0.69 | -0.57 | -0.39 |

Comparable replans in the other 28 scenes (0.1.6; car below 0.5 m/s, human above 0.5 m/s or exceeding it within 2 s, an actor box within 4 m ahead in the ego-width corridor): 3 replans in 1 scene(s):

| scene | t_after_handover_s | ahead_gap_m | ahead_label | human_speed_mps | human_starts_moving_in_s | chosen_leave |
|---|---|---|---|---|---|---|
| e3ed554f | 0 | 3.88 | automobile | 0.3 | 0.67 | True |
| e3ed554f | 9.5 | 3.8 | automobile | 0.92 | 0.07 | False |
| e3ed554f | 10 | 3.61 | automobile | 0.67 | 0.07 | False |

### 2.5 Score of leave+prompt proposals by clearance bin
| scene | version | clearance_bin_m | n_leave_prompt | share_of_leave_prompt | median_score_final | q25_score_final | q75_score_final | median_score_plain | median_rank_final | best_score_final |
|---|---|---|---|---|---|---|---|---|---|---|
| b44a4a44 | 0.1.6 | 0 (overlap) | 74 | 0.3 | 0.5 | 0.42 | 0.61 | 0.58 | 28.5 | 0.85 |
| b44a4a44 | 0.1.6 | 0-0.3 | 85 | 0.35 | 0.57 | 0.51 | 0.65 | 0.65 | 29 | 0.84 |
| b44a4a44 | 0.1.6 | 0.3-0.6 | 34 | 0.14 | 0.58 | 0.54 | 0.62 | 0.67 | 28.5 | 0.84 |
| b44a4a44 | 0.1.6 | 0.6-1.0 | 7 | 0.03 | 0.55 | 0.49 | 0.57 | 0.63 | 30 | 0.63 |
| b44a4a44 | 0.1.6 | > 1.0 | 43 | 0.18 | 0.55 | 0.46 | 0.62 | 0.63 | 27 | 0.85 |
| d2715023 | 0.1.6 | 0 (overlap) | 164 | 0.81 | 0.05 | 0.02 | 0.08 | 0.05 | 28 | 0.66 |
| d2715023 | 0.1.6 | 0-0.3 | 29 | 0.14 | 0.05 | 0.04 | 0.08 | 0.06 | 27 | 0.12 |
| d2715023 | 0.1.6 | 0.3-0.6 | 2 | 0.01 | 0.02 | 0.01 | 0.02 | 0.02 | 31.5 | 0.03 |
| d2715023 | 0.1.6 | 0.6-1.0 | 3 | 0.01 | 0.04 | 0.02 | 0.05 | 0.05 | 25 | 0.06 |
| d2715023 | 0.1.6 | > 1.0 | 5 | 0.02 | 0.01 | 0.01 | 0.03 | 0.01 | 29 | 0.15 |
| d2715023 | 0.1.4b | 0 (overlap) | 297 | 0.92 | 0.03 | 0.02 | 0.14 | 0.03 | 26 | 0.74 |
| d2715023 | 0.1.4b | 0-0.3 | 10 | 0.03 | 0.22 | 0.04 | 0.38 | 0.11 | 16.5 | 0.64 |
| d2715023 | 0.1.4b | 0.3-0.6 | 3 | 0.01 | 0.03 | 0.02 | 0.32 | 0.03 | 26 | 0.62 |
| d2715023 | 0.1.4b | 0.6-1.0 | 4 | 0.01 | 0.07 | 0.01 | 0.15 | 0.01 | 27 | 0.18 |
| d2715023 | 0.1.4b | > 1.0 | 10 | 0.03 | 0.07 | 0.01 | 0.17 | 0.01 | 27 | 0.28 |
| b44a4a44 | 0.1.4b | 0 (overlap) | 223 | 0.65 | 0.12 | 0.05 | 0.19 | 0.13 | 27 | 0.31 |
| b44a4a44 | 0.1.4b | 0-0.3 | 40 | 0.12 | 0.16 | 0.12 | 0.19 | 0.18 | 23 | 0.28 |
| b44a4a44 | 0.1.4b | 0.3-0.6 | 10 | 0.03 | 0.18 | 0.17 | 0.22 | 0.2 | 19 | 0.26 |
| b44a4a44 | 0.1.4b | 0.6-1.0 | 5 | 0.01 | 0.17 | 0.11 | 0.18 | 0.19 | 27 | 0.24 |
| b44a4a44 | 0.1.4b | > 1.0 | 63 | 0.18 | 0.14 | 0.06 | 0.19 | 0.16 | 26 | 0.27 |

For reference, stays on the same replans:

| scene | version | n_stay | median_stay_score_final | median_chosen_score_final |
|---|---|---|---|---|
| b44a4a44 | 0.1.4b | 239 | 0.68 | 0.72 |
| b44a4a44 | 0.1.6 | 276 | 0.91 | 0.93 |
| d2715023 | 0.1.4b | 450 | 0.84 | 0.92 |
| d2715023 | 0.1.6 | 383 | 0.92 | 0.95 |

## Plots
- `plot1_grid_heatmap.png`
- `plot2_b44a4a44_first_firing_topview.png`
- `plot3_timelines.png`
- `plot4_probe_d2715023.png`
- `plot5_probe_b44a4a44.png`
- `plot6_clearance_bins.png`
- `plot7_d2715023_injected_topview.png`
- `plot8_score_vs_clearance.png`

Colours: the dataviz reference palette in its fixed order; its validator script could not be run here (Node 18 rejects the file as a CommonJS script and prints nothing when it is loaded as a module).

## Files
`antistall_grid.csv`, `antistall_first_firings.csv`, `stationary_episodes.csv`, `scorer_probe.csv`, `scorer_probe_key.csv`, `scorer_geometry.csv`, `scorer_geometry_spearman.csv`, `clearance_bins.csv`. Scratch (model inputs of the 6 probe ticks, camera tensors, 66 MB, not deleted): `{runs}/drivejepa_retrain/t05_scratch_20261009/`.
