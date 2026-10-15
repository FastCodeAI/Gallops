# T04 - the car that never pulls out (Drive-JEPA 0.1.6): what is on the menu and how it is scored

Offline, CPU only. Nothing under `alpasim/` or `adapters/` was edited. Scripts (this directory): `t04_extract.py` (one pass over each `rollout.asl` into `cache/`), `t04_common.py` (box check, classes), `t04_menu.py` (Part B + validation), `t04_scene.py` (Parts A and C), `t04_plots.py`, `t04_readme.py` (this file is generated). Measured facts only.

## 0. Where the data came from, and the definitions as applied

- **Actor boxes are in the run logs.** `rollout.asl` holds `rollout_metadata.actor_definitions.actor_aabb` (id, label, box size, including `EGO`) and one `actor_poses` entry per 50 ms simulator step (centre pose of every actor box), `alpasim/src/grpc/alpasim_grpc/v0/logging.proto:43-70`. `tools/make_replays.py` does **not** read or draw actor boxes (its only `aabb` use is the ego rig-to-box offset, lines 213-279); the task text said it does.
- **Ego footprint**: the `EGO` entry of the same table: 4.688 x 1.999 m on d2715023, 5.207 x 2.157 m on b44a4a44 (the vehicle differs per scene), centred `transform_ego_coords_rig_to_aabb.x` = 1.312 m / 1.467 m ahead of the rig origin (checked: the logged EGO box pose equals rig pose + that offset).
- **Actors do not react to the car.** The logs contain no `traffic_request`/`traffic_return` entries (wizard config: 'no traffic simulation configured'); actor poses are the recording played back, identical in the 0.1.6 and 0.1.4b runs. So every box check here is 'against where the recorded actors were', with a car that in the recording had already left.
- **Replans**: every 0.5 s from +0.0 s to +16.5 s, 34 per run. `pdm_score` (called *final* below) has its maximum at the chosen index on 1156 of 1156 replans checked; `S_plain` is the scorer output before the momentum re-score (called *plain*). Rank 1 = best of 32.
- **leave**: end point >= 2.0 m from the car. **prompt**: arc >= 0.5 m by 1.0 s and >= 1.5 m by 2.0 s. **late-go**: leave and not prompt. **usable escape**: leave and prompt and box-free (0 m margin).
- **box-free**: rig origin + 8 waypoints interpolated linearly (x, y, and the proposal's own heading) at 0.1 s, 41 samples over 0..4 s; ego box against every recorded actor box at the same time stamp (separating-axis test on the oriented rectangles). Adjustment: the actor log ends at +16.8 s, so for replans after +12.8 s the samples beyond the end of the log are not checked (`box_checked_horizon_s`).
- **human-like** (adjusted): nearest-point distance <= 1.0 m, at the plan's 2 s and 4 s points, to the human's ground-truth path *from handover to the end of the clip*. The path 'from the same tick' was not used because the car stands still while the human drives off: from +5 s on the human's remaining path starts many metres ahead of the car and nothing could be within 1 m of it. Same-time distance is in `proposals.csv` (`human_same_time_*`).
- **gap**: score of the chosen plan minus the best score among usable escapes (`gap_chosen_minus_best_usable_*`); the best-stay-minus-best-usable gap is also in `replans.csv`. On the two stuck scenes the chosen plan is a stay on every replan, so the two are equal there.

### Validation of the box check

| margin_m | ticks | passes | pass_rate |
|---|---|---|---|
| 0.000 | 1156 | 1156 | 1 |
| 0.300 | 1156 | 1156 | 1 |

The human's own future path (ground truth at the same time stamps, same footprint) from every replan tick of all 30 scenes (0.1.6 runs) and of the four 0.1.4b runs. For replans after +12.85 s the ground truth ends before 4 s and the shorter remainder was checked. Second check, in the other direction: the recorded (stationary) car of b44a4a44 first overlaps an actor box at +6.55 s (actor 15, label `person`, 0.74 x 0.77 m) in both versions, against a logged collision time of +6.60 s; 5f064b02 0.1.6 at +14.50 s (actor 32, automobile) against +14.55 s logged; d2715023 never, and none is logged.

## Part A - the two scenes
### A1. Human timeline

**d2715023** - human first above 0.5 m/s at **+0.99 s** after handover; largest leftward offset 4.03 m (at +8.50 s); human forward distance at +16.75 s 43.1 m; car (0.1.6) 1.22 m, (0.1.4b) 0.84 m. Frame: the car's pose at handover (x forward, y left).

| t_after_handover_s | human_speed_mps | human_forward_m | human_lateral_left_m | ego_016_speed_mps | ego_016_forward_m | ego_016_lateral_left_m | ego_014b_forward_m |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.07 | 0.00 | 0.00 | 0.07 | -0.00 | 0.00 | -0.00 |
| 1 | 0.54 | 0.30 | 0.02 | 0.16 | 0.13 | 0.00 | 0.09 |
| 2 | 0.80 | 1.00 | 0.16 | 0.33 | 0.38 | 0.01 | 0.13 |
| 3 | 0.48 | 1.60 | 0.40 | 0.24 | 0.68 | 0.02 | 0.17 |
| 4 | 0.86 | 2.14 | 0.73 | 0.07 | 0.80 | 0.03 | 0.21 |
| 5 | 1.38 | 3.00 | 1.38 | 0.11 | 0.91 | 0.03 | 0.24 |
| 6 | 2.66 | 4.65 | 2.51 | 0.02 | 0.97 | 0.04 | 0.27 |
| 8 | 4.44 | 11.59 | 4.01 | 0.03 | 1.00 | 0.04 | 0.33 |
| 10 | 5.28 | 21.77 | 3.73 | 0.06 | 1.13 | 0.05 | 0.38 |
| 12 | 4.37 | 31.45 | 3.06 | 0.04 | 1.18 | 0.06 | 0.45 |
| 14 | 3.46 | 39.56 | 2.60 | 0.00 | 1.19 | 0.06 | 0.56 |
| 16 | 0.15 | 43.11 | 2.40 | 0.02 | 1.21 | 0.06 | 0.74 |

**b44a4a44** - human first above 0.5 m/s at **+2.98 s** after handover; largest leftward offset 1.67 m (at +8.00 s); human forward distance at +16.75 s 99.3 m; car (0.1.6) 0.07 m, (0.1.4b) 0.18 m. Frame: the car's pose at handover (x forward, y left).

| t_after_handover_s | human_speed_mps | human_forward_m | human_lateral_left_m | ego_016_speed_mps | ego_016_forward_m | ego_016_lateral_left_m | ego_014b_forward_m |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.00 | -0.00 | 0.00 | 0.02 | 0.01 | 0.00 | 0.01 |
| 2 | 0.01 | -0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.02 |
| 3 | 0.62 | 0.20 | -0.00 | 0.01 | 0.03 | 0.00 | 0.02 |
| 4 | 1.44 | 1.30 | 0.12 | 0.00 | 0.03 | 0.00 | 0.03 |
| 5 | 2.74 | 3.15 | 0.53 | 0.00 | 0.03 | 0.00 | 0.04 |
| 6 | 5.18 | 7.00 | 1.26 | 0.00 | 0.04 | 0.00 | 0.04 |
| 8 | 12.21 | 24.66 | 1.67 | 0.00 | 0.04 | 0.00 | 0.06 |
| 10 | 14.92 | 53.92 | 1.36 | 0.01 | 0.04 | 0.00 | 0.08 |
| 12 | 9.84 | 78.97 | 1.08 | 0.01 | 0.06 | 0.00 | 0.12 |
| 14 | 4.29 | 92.45 | 0.95 | 0.00 | 0.07 | 0.00 | 0.14 |
| 16 | 1.85 | 98.10 | 0.63 | 0.00 | 0.07 | 0.00 | 0.17 |

Human start on the two contrast scenes: 5f064b02 +3.38 s, e3ed554f +0.67 s.

### A2. What is in front at handover
All actors whose box is within 8 m of the ego box at handover (`front_actors.csv` has 15 m). 'ahead in corridor' = the box lies ahead of the rig origin and overlaps the ego's width; gap is from the ego front bumper to the nearest corner. Displacement and peak speed are over the whole clip (-3 s .. +16.8 s).

| scene | actor | label | length_m | width_m | centre_x_fwd_m | centre_y_left_m | box_to_box_distance_m | ahead_in_ego_width_corridor | gap_from_ego_front_bumper_m | displacement_over_clip_m | max_speed_mps | present_from_s | present_to_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b44a4a44 | 20 | automobile | 3.89 | 1.83 | 8.43 | -1.60 | 2.38 | True | 2.38 | 0.40 | 0.39 | -3 | 14.35 |
| b44a4a44 | 51 | automobile | 4.35 | 1.87 | 9.81 | 4.84 | 4.55 | False |  | 0.83 | 1.37 | -3 | 12.95 |
| b44a4a44 | 15 | person | 0.74 | 0.77 | 2.04 | -8.50 | 7.03 | False |  | 15.35 | 1.38 | -3 | 10.65 |
| b44a4a44 | 14 | person | 0.67 | 0.72 | 2.85 | -8.75 | 7.32 | False |  | 15.49 | 1.37 | -3 | 10.65 |
| b44a4a44 | 19 | automobile | 4.14 | 1.83 | 13.53 | -1.81 | 7.34 | True | 7.34 | 0.38 | 0.72 | -3 | 16.80 |
| d2715023 | 32 | automobile | 4.16 | 1.90 | 8.06 | 0.00 | 2.26 | True | 2.26 | 0.55 | 0.38 | -3.15 | 16.80 |
| d2715023 | 31 | automobile | 4.50 | 2.08 | -7.36 | -0.13 | 4.04 | False |  | 0.16 | 0.32 | -3 | 16.75 |
| d2715023 | 33 | automobile | 4.11 | 1.87 | -4.38 | 9.36 | 7.62 | False |  | 0.14 | 0.45 | -3.15 | 16.80 |

Moving actors on b44a4a44 that come within 12 m of the car's handover position (handover frame): actor 25 (automobile) drives from 35 m ahead / 2.1 m left to behind the car, passing 4.3 m to the left at +4.9 s; actors 15 and 14 (`person`) walk from 12 m to the right across the car's position, actor 15 reaching (2.5 m forward, 1.0 m right) at +7.0 s; actor 149 (automobile) crosses 2.8 m behind the rig origin at +7.3 s. On d2715023 no actor within 8 m moves more than 0.55 m.

### A3. Front camera frames (0.1.6 run; PNGs in this directory)
- `frame_d2715023_016_plus0s.png` (handover): a dark-red SUV directly ahead in the same lane at the right kerb, brake lights not obviously lit; open lane to its left; a white SUV parked on the far left side; lawn and pavement on the right.
- `frame_d2715023_016_plus5s.png`: the same red SUV, larger in the frame (the car has crept forward), still filling the lane ahead; lane to the left still open.
- `frame_b44a4a44_016_plus0s.png` (handover): narrow uphill street with parked cars on both sides; a dark-green VW hatchback parked directly ahead on the right; a silver car is approaching in the middle of the street about 3-4 car lengths ahead.
- `frame_b44a4a44_016_plus5s.png`: same view, unchanged position; the approaching silver car is gone from the view and the middle of the street is empty.
- 0.1.4b frames at +5 s are saved too (`frame_*_014b_plus5s.png`).

### A4. What the model was fed

| scene | version | replans | byte_identical_frame_pairs | frame_gap_min_s | frame_gap_max_s | newest_frame_lag_max_s | fed_speed_median_mps | fed_speed_max_mps | reported_accel_x_absmax_mps2 |
|---|---|---|---|---|---|---|---|---|---|
| b44a4a44 | 0.1.4b | 34 | 0 | 0.500 | 0.500 | 0 | 0.011 | 0.020 | 0.019 |
| b44a4a44 | 0.1.6 | 34 | 0 | 0.500 | 0.500 | 0 | 0.001 | 0.018 | 0.039 |
| d2715023 | 0.1.4b | 34 | 0 | 0.500 | 0.500 | 0 | 0.037 | 0.133 | 0.153 |
| d2715023 | 0.1.6 | 34 | 0 | 0.500 | 0.500 | 0 | 0.030 | 0.331 | 0.318 |

The adapter builds the model's ego-status row in `adapters/drivejepa016/planner.py:815-857` (`_build_ego_status`: relative pose, rig-frame velocity, reported acceleration, command), takes the newest row's velocity from `_newest_velocity` (`planner.py:746-813`, reported velocity verbatim unless it is exactly zero beside moving poses) and zeroes the two acceleration slots in `_model_ego_status` (`planner.py:733-745`, `constants.py:57-58`); the two context frames are chosen by time stamp in `_select_frames` (`planner.py:605-670`), 0.5 s apart. Measured on the two stuck scenes at all post-handover replans: the speed fed to the model equals the speed the simulator reported to the driver to 0.0e+00 m/s and is within 0.068 m/s of the controller-state speed at the same tick; the two frames are always 0.500 s apart, the newest ending at the replan time; no pair of context frames is byte-identical (table). The car itself is near standstill at these ticks (fed speed max in the table), so a low fed speed is the car's own speed, not a substituted value; before handover the human is also below 0.5 m/s on both scenes (A1), so the replayed part contains no human motion that a zero speed would contradict. The acceleration the simulator reported (last column, peak magnitude) is what 0.1.6 replaces by zero and 0.1.4b passes through.

## Part B - the menu at every replan
### B1. Per scene and version (all 34 replans, handover to end)

| scene | version | n_replans | n_chosen_stay | median_n_leave | median_n_leave_prompt | median_n_late_go | median_n_usable | share_leave_box_free | share_of_leave_box_free_with_prompt_one | share_usable_escape | share_usable_margin0p3 | share_usable_and_human_like |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5f064b02 | 0.1.6 | 34 | 0 | 32 | 32 | 0.00 | 24 | 0.85 | 1 | 0.85 | 0.85 | 0.85 |
| b44a4a44 | 0.1.6 | 34 | 34 | 24 | 7 | 17 | 7 | 0.76 | 1 | 0.76 | 0.47 | 0.76 |
| d2715023 | 0.1.6 | 34 | 34 | 21 | 5 | 15 | 1 | 1 | 0.56 | 0.56 | 0.09 | 0.53 |
| e3ed554f | 0.1.6 | 34 | 7 | 28.50 | 22 | 6 | 5.50 | 1 | 0.91 | 0.91 | 0.88 | 0.47 |
| d2715023 | 0.1.4b | 34 | 34 | 19 | 10 | 9 | 0.00 | 0.82 | 0.32 | 0.26 | 0.12 | 0.21 |
| b44a4a44 | 0.1.4b | 34 | 34 | 25 | 10 | 15 | 3 | 0.74 | 0.92 | 0.68 | 0.41 | 0.68 |
| 5f064b02 | 0.1.4b | 34 | 2 | 31 | 30.50 | 1 | 28 | 0.82 | 1 | 0.82 | 0.82 | 0.82 |
| e3ed554f | 0.1.4b | 34 | 17 | 30 | 12 | 5.50 | 0.00 | 0.71 | 0.58 | 0.41 | 0.35 | 0.21 |

`share_leave_box_free`: replans with at least one box-free leave proposal; `share_of_leave_box_free_with_prompt_one`: of those, replans where at least one box-free leave proposal is also prompt (= a usable escape).

Scores and ranks, over the replans that have a usable escape:

| scene | version | n_usable_escape | median_chosen_score_final | median_best_usable_score_final | median_gap_chosen_minus_best_usable_final | median_gap_chosen_minus_best_usable_plain | median_best_usable_rank_final | best_usable_rank_final | median_best_usable_rank_plain | best_usable_rank_plain | share_usable_replans_rank_worse_than_5_final | share_usable_replans_rank_worse_than_5_plain | share_usable_replans_below_every_stay_final | median_n_stay_above_best_usable_final | median_best_leave_rank_final | median_best_leave_score_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5f064b02 | 0.1.6 | 29 | 0.95 | 0.93 | 0.00 | -0.02 | 1 | 1 | 1 | 1 | 0.24 | 0.14 | 0.69 | 0.00 | 1 | 0.95 |
| b44a4a44 | 0.1.6 | 26 | 0.93 | 0.67 | 0.22 | 0.14 | 16 | 11 | 17 | 11 | 1 | 1 | 1 | 8 | 9 | 0.85 |
| d2715023 | 0.1.6 | 19 | 0.95 | 0.06 | 0.88 | 0.87 | 26 | 18 | 26 | 20 | 1 | 1 | 1 | 11 | 12 | 0.38 |
| e3ed554f | 0.1.6 | 31 | 0.97 | 0.96 | 0.00 | 0.00 | 2 | 1 | 2 | 1 | 0.29 | 0.19 | 0.23 | 0.00 | 1 | 0.97 |
| d2715023 | 0.1.4b | 9 | 0.92 | 0.37 | 0.56 | 0.64 | 13 | 11 | 13 | 11 | 1 | 1 | 0.22 | 12 | 11.50 | 0.36 |
| b44a4a44 | 0.1.4b | 23 | 0.72 | 0.21 | 0.51 | 0.44 | 19 | 13 | 19 | 13 | 1 | 1 | 1 | 7 | 8 | 0.49 |
| 5f064b02 | 0.1.4b | 28 | 0.90 | 0.93 | 0.00 | 0.00 | 1 | 1 | 1 | 1 | 0.04 | 0.04 | 0.46 | 0.00 | 1 | 0.90 |
| e3ed554f | 0.1.4b | 14 | 0.85 | 0.80 | 0.04 | 0.00 | 2.50 | 1 | 2.50 | 1 | 0.43 | 0.29 | 0.21 | 0.00 | 1.50 | 0.81 |

### B2. Same summary on restricted sets of replans

| scene | version | replans_subset | n_replans | n_chosen_stay | share_usable_escape | share_usable_and_human_like | median_n_leave_prompt | median_n_late_go | median_best_usable_rank_final | best_usable_rank_final | median_gap_chosen_minus_best_usable_final | share_usable_replans_rank_worse_than_5_final | median_best_leave_rank_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5f064b02 | 0.1.6 | ego < 0.5 m/s | 3 | 0 | 1 | 1 | 18 | 10 | 2 | 2 | 0.02 | 0.33 | 1 |
| b44a4a44 | 0.1.6 | ego < 0.5 m/s | 34 | 34 | 0.76 | 0.76 | 7 | 17 | 16 | 11 | 0.22 | 1 | 9 |
| d2715023 | 0.1.6 | ego < 0.5 m/s | 34 | 34 | 0.56 | 0.53 | 5 | 15 | 26 | 18 | 0.88 | 1 | 12 |
| e3ed554f | 0.1.6 | ego < 0.5 m/s | 16 | 6 | 0.88 | 0.12 | 12.50 | 10 | 13 | 1 | 0.12 | 0.64 | 1 |
| d2715023 | 0.1.4b | ego < 0.5 m/s | 34 | 34 | 0.26 | 0.21 | 10 | 9 | 13 | 11 | 0.56 | 1 | 11.50 |
| b44a4a44 | 0.1.4b | ego < 0.5 m/s | 34 | 34 | 0.68 | 0.68 | 10 | 15 | 19 | 13 | 0.51 | 1 | 8 |
| 5f064b02 | 0.1.4b | ego < 0.5 m/s | 5 | 1 | 0.20 | 0.20 | 11 | 19 | 1 | 1 | 0.00 | 0.00 | 1 |
| e3ed554f | 0.1.4b | ego < 0.5 m/s | 18 | 9 | 0.22 | 0.11 | 11 | 15.50 | 2.50 | 1 | 0.00 | 0.25 | 1.50 |
| 5f064b02 | 0.1.6 | human moving (> 0.5 m/s) | 27 | 0 | 0.81 | 0.81 | 32 | 0.00 | 1 | 1 | 0.00 | 0.27 | 1 |
| b44a4a44 | 0.1.6 | human moving (> 0.5 m/s) | 28 | 28 | 0.71 | 0.71 | 7 | 17 | 16.50 | 11 | 0.21 | 1 | 9 |
| d2715023 | 0.1.6 | human moving (> 0.5 m/s) | 29 | 29 | 0.55 | 0.52 | 5 | 16 | 26.50 | 20 | 0.89 | 1 | 12 |
| e3ed554f | 0.1.6 | human moving (> 0.5 m/s) | 19 | 3 | 0.84 | 0.74 | 27 | 2 | 2 | 1 | 0.00 | 0.06 | 1 |
| d2715023 | 0.1.4b | human moving (> 0.5 m/s) | 29 | 29 | 0.24 | 0.24 | 10 | 9 | 13 | 11 | 0.54 | 1 | 12 |
| b44a4a44 | 0.1.4b | human moving (> 0.5 m/s) | 28 | 28 | 0.68 | 0.68 | 10 | 14 | 19 | 13 | 0.52 | 1 | 8 |
| 5f064b02 | 0.1.4b | human moving (> 0.5 m/s) | 27 | 2 | 0.78 | 0.78 | 31 | 1 | 1 | 1 | 0.00 | 0.05 | 1 |
| e3ed554f | 0.1.4b | human moving (> 0.5 m/s) | 19 | 11 | 0.37 | 0.32 | 12 | 1 | 19 | 2 | 0.40 | 0.71 | 2 |
| b44a4a44 | 0.1.6 | ego < 0.5 m/s and human moving | 28 | 28 | 0.71 | 0.71 | 7 | 17 | 16.50 | 11 | 0.21 | 1 | 9 |
| d2715023 | 0.1.6 | ego < 0.5 m/s and human moving | 29 | 29 | 0.55 | 0.52 | 5 | 16 | 26.50 | 20 | 0.89 | 1 | 12 |
| e3ed554f | 0.1.6 | ego < 0.5 m/s and human moving | 4 | 2 | 0.50 | 0.00 | 15 | 9.50 | 7.50 | 3 | 0.07 | 0.50 | 2.50 |
| d2715023 | 0.1.4b | ego < 0.5 m/s and human moving | 29 | 29 | 0.24 | 0.24 | 10 | 9 | 13 | 11 | 0.54 | 1 | 12 |
| b44a4a44 | 0.1.4b | ego < 0.5 m/s and human moving | 28 | 28 | 0.68 | 0.68 | 10 | 14 | 19 | 13 | 0.52 | 1 | 8 |
| 5f064b02 | 0.1.4b | ego < 0.5 m/s and human moving | 4 | 1 | 0.00 | 0.00 | 11.50 | 19 |  |  |  |  | 1 |
| e3ed554f | 0.1.4b | ego < 0.5 m/s and human moving | 8 | 3 | 0.12 | 0.12 | 11.50 | 17 | 2 | 2 | 0.00 | 0.00 | 1 |
| 5f064b02 | 0.1.6 | full 4 s box horizon (t <= 12.5 s) | 26 | 0 | 1 | 1 | 32 | 0.00 | 1 | 1 | 0.00 | 0.15 | 1 |
| b44a4a44 | 0.1.6 | full 4 s box horizon (t <= 12.5 s) | 26 | 26 | 0.69 | 0.69 | 7 | 17 | 16.50 | 11 | 0.22 | 1 | 9 |
| d2715023 | 0.1.6 | full 4 s box horizon (t <= 12.5 s) | 26 | 26 | 0.58 | 0.58 | 5.50 | 15 | 27 | 20 | 0.88 | 1 | 12 |
| e3ed554f | 0.1.6 | full 4 s box horizon (t <= 12.5 s) | 26 | 7 | 0.88 | 0.62 | 22.50 | 5 | 2 | 1 | 0.00 | 0.26 | 1 |
| d2715023 | 0.1.4b | full 4 s box horizon (t <= 12.5 s) | 26 | 26 | 0.19 | 0.19 | 10 | 9.50 | 13 | 11 | 0.54 | 1 | 11 |
| b44a4a44 | 0.1.4b | full 4 s box horizon (t <= 12.5 s) | 26 | 26 | 0.58 | 0.58 | 10 | 14.50 | 18 | 13 | 0.51 | 1 | 8 |
| 5f064b02 | 0.1.4b | full 4 s box horizon (t <= 12.5 s) | 26 | 0 | 0.96 | 0.96 | 31 | 1 | 1 | 1 | 0.00 | 0.04 | 1 |
| e3ed554f | 0.1.4b | full 4 s box horizon (t <= 12.5 s) | 26 | 16 | 0.31 | 0.27 | 11.50 | 5.50 | 14 | 2 | 0.25 | 0.62 | 2 |

### B3. Every replan of the two stuck scenes, 0.1.6

**d2715023**

| t_after_handover_s | ego_speed_mps | human_speed_mps | n_stay | n_stay_box_free | n_late_go | n_leave_prompt | n_leave_box_free | n_usable | n_usable_margin0p3 | n_usable_human_like | chosen_score_final | chosen_score_plain | best_usable_score_final | best_usable_score_plain | best_usable_rank_final | best_usable_rank_plain | n_stay_above_best_usable_final | best_leave_score_final | best_leave_rank_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.07 | 0.07 | 11 | 11 | 14 | 7 | 12 | 0 | 0 | 0 | 0.92 | 0.91 |  |  |  |  |  | 0.80 | 12 |
| 0.50 | 0.15 | 0.30 | 11 | 11 | 12 | 9 | 10 | 1 | 0 | 1 | 0.94 | 0.93 | 0.09 | 0.11 | 29 | 29 | 11 | 0.80 | 12 |
| 1 | 0.16 | 0.54 | 11 | 11 | 10 | 11 | 11 | 2 | 0 | 2 | 0.95 | 0.95 | 0.10 | 0.11 | 27 | 27 | 11 | 0.78 | 12 |
| 1.50 | 0.24 | 0.74 | 10 | 10 | 10 | 12 | 14 | 4 | 0 | 4 | 0.93 | 0.93 | 0.12 | 0.14 | 20 | 20 | 10 | 0.76 | 11 |
| 2 | 0.33 | 0.80 | 10 | 10 | 10 | 12 | 11 | 3 | 0 | 2 | 0.94 | 0.93 | 0.08 | 0.09 | 23 | 23 | 10 | 0.76 | 11 |
| 2.50 | 0.31 | 0.66 | 11 | 11 | 11 | 10 | 10 | 1 | 0 | 1 | 0.92 | 0.91 | 0.09 | 0.11 | 22 | 22 | 11 | 0.65 | 12 |
| 3 | 0.24 | 0.48 | 11 | 11 | 12 | 9 | 8 | 0 | 0 | 0 | 0.93 | 0.92 |  |  |  |  |  | 0.64 | 12 |
| 3.50 | 0.11 | 0.61 | 10 | 10 | 16 | 6 | 11 | 1 | 0 | 1 | 0.94 | 0.93 | 0.04 | 0.04 | 29 | 29 | 10 | 0.73 | 11 |
| 4 | 0.07 | 0.86 | 12 | 12 | 14 | 6 | 13 | 2 | 0 | 2 | 0.96 | 0.95 | 0.09 | 0.10 | 27 | 27 | 12 | 0.76 | 12 |
| 4.50 | 0.11 | 1.08 | 11 | 11 | 15 | 6 | 14 | 1 | 0 | 1 | 0.95 | 0.94 | 0.06 | 0.06 | 30 | 30 | 11 | 0.74 | 12 |
| 5 | 0.11 | 1.38 | 13 | 13 | 13 | 6 | 12 | 0 | 0 | 0 | 0.95 | 0.94 |  |  |  |  |  | 0.57 | 13 |
| 5.50 | 0.07 | 2.00 | 13 | 13 | 13 | 6 | 16 | 3 | 0 | 3 | 0.96 | 0.95 | 0.05 | 0.05 | 27 | 27 | 13 | 0.59 | 14 |
| 6 | 0.02 | 2.66 | 12 | 12 | 15 | 5 | 8 | 0 | 0 | 0 | 0.95 | 0.94 |  |  |  |  |  | 0.57 | 12 |
| 6.50 | 0.03 | 3.14 | 13 | 13 | 14 | 5 | 6 | 0 | 0 | 0 | 0.95 | 0.95 |  |  |  |  |  | 0.40 | 14 |
| 7 | 0.01 | 3.63 | 11 | 11 | 17 | 4 | 10 | 0 | 0 | 0 | 0.95 | 0.94 |  |  |  |  |  | 0.44 | 12 |
| 7.50 | 0.00 | 4.02 | 13 | 13 | 13 | 6 | 1 | 0 | 0 | 0 | 0.97 | 0.96 |  |  |  |  |  | 0.32 | 13 |
| 8 | 0.03 | 4.44 | 12 | 12 | 15 | 5 | 5 | 0 | 0 | 0 | 0.96 | 0.96 |  |  |  |  |  | 0.23 | 13 |
| 8.50 | 0.06 | 4.87 | 11 | 11 | 17 | 4 | 17 | 1 | 0 | 1 | 0.96 | 0.95 | 0.04 | 0.04 | 26 | 26 | 11 | 0.34 | 12 |
| 9 | 0.07 | 5.26 | 11 | 11 | 16 | 5 | 17 | 2 | 0 | 2 | 0.95 | 0.95 | 0.07 | 0.08 | 24 | 24 | 11 | 0.50 | 12 |
| 9.50 | 0.09 | 5.36 | 9 | 8 | 18 | 5 | 16 | 2 | 0 | 2 | 0.93 | 0.92 | 0.05 | 0.06 | 27 | 27 | 9 | 0.53 | 10 |
| 10 | 0.06 | 5.28 | 10 | 9 | 18 | 4 | 17 | 1 | 0 | 1 | 0.90 | 0.89 | 0.05 | 0.06 | 22 | 22 | 10 | 0.35 | 10 |
| 10.50 | 0.02 | 5.09 | 10 | 10 | 18 | 4 | 15 | 1 | 0 | 1 | 0.94 | 0.93 | 0.05 | 0.06 | 26 | 26 | 10 | 0.31 | 11 |
| 11 | 0.00 | 4.89 | 11 | 11 | 16 | 5 | 3 | 0 | 0 | 0 | 0.95 | 0.95 |  |  |  |  |  | 0.20 | 12 |
| 11.50 | 0.03 | 4.59 | 11 | 10 | 16 | 5 | 4 | 0 | 0 | 0 | 0.98 | 0.98 |  |  |  |  |  | 0.19 | 12 |
| 12 | 0.04 | 4.37 | 10 | 10 | 18 | 4 | 12 | 0 | 0 | 0 | 0.95 | 0.95 |  |  |  |  |  | 0.29 | 10 |
| 12.50 | 0.01 | 4.21 | 11 | 11 | 17 | 4 | 15 | 1 | 0 | 1 | 0.95 | 0.94 | 0.03 | 0.03 | 28 | 28 | 11 | 0.27 | 12 |
| 13 | 0.00 | 4.15 | 11 | 11 | 16 | 5 | 8 | 0 | 0 | 0 | 0.97 | 0.97 |  |  |  |  |  | 0.25 | 12 |
| 13.50 | 0.00 | 3.93 | 11 | 11 | 16 | 5 | 10 | 0 | 0 | 0 | 0.98 | 0.98 |  |  |  |  |  | 0.30 | 12 |
| 14 | 0.00 | 3.46 | 11 | 11 | 16 | 5 | 8 | 0 | 0 | 0 | 0.99 | 0.98 |  |  |  |  |  | 0.26 | 12 |
| 14.50 | 0.00 | 2.75 | 11 | 11 | 16 | 5 | 13 | 0 | 0 | 0 | 0.98 | 0.98 |  |  |  |  |  | 0.35 | 12 |
| 15 | 0.00 | 1.81 | 13 | 13 | 14 | 5 | 15 | 1 | 0 | 0 | 0.99 | 0.98 | 0.01 | 0.01 | 32 | 32 | 13 | 0.35 | 14 |
| 15.50 | 0.03 | 0.77 | 12 | 12 | 15 | 5 | 19 | 4 | 2 | 1 | 0.98 | 0.97 | 0.02 | 0.02 | 25 | 25 | 12 | 0.28 | 13 |
| 16 | 0.02 | 0.15 | 11 | 11 | 17 | 4 | 21 | 4 | 4 | 3 | 0.94 | 0.93 | 0.06 | 0.06 | 21 | 21 | 11 | 0.36 | 12 |
| 16.50 | 0.00 | 0.01 | 14 | 14 | 14 | 4 | 18 | 4 | 4 | 1 | 0.97 | 0.97 | 0.15 | 0.03 | 18 | 20 | 14 | 0.27 | 13 |

**b44a4a44**

| t_after_handover_s | ego_speed_mps | human_speed_mps | n_stay | n_stay_box_free | n_late_go | n_leave_prompt | n_leave_box_free | n_usable | n_usable_margin0p3 | n_usable_human_like | chosen_score_final | chosen_score_plain | best_usable_score_final | best_usable_score_plain | best_usable_rank_final | best_usable_rank_plain | n_stay_above_best_usable_final | best_leave_score_final | best_leave_rank_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 8 | 8 | 18 | 6 | 14 | 1 | 0 | 1 | 0.93 | 0.92 | 0.67 | 0.77 | 16 | 16 | 8 | 0.87 | 9 |
| 0.50 | 0.02 | 0.00 | 8 | 8 | 17 | 7 | 21 | 4 | 0 | 4 | 0.93 | 0.92 | 0.63 | 0.72 | 15 | 14 | 8 | 0.85 | 8 |
| 1 | 0.02 | 0.00 | 7 | 7 | 18 | 7 | 25 | 7 | 0 | 7 | 0.92 | 0.91 | 0.67 | 0.76 | 16 | 16 | 7 | 0.86 | 8 |
| 1.50 | 0.01 | 0.00 | 7 | 7 | 18 | 7 | 25 | 7 | 0 | 7 | 0.93 | 0.92 | 0.67 | 0.77 | 16 | 16 | 7 | 0.88 | 8 |
| 2 | 0.00 | 0.01 | 6 | 6 | 18 | 8 | 26 | 8 | 0 | 8 | 0.91 | 0.90 | 0.67 | 0.76 | 17 | 17 | 6 | 0.86 | 7 |
| 2.50 | 0.01 | 0.15 | 8 | 8 | 17 | 7 | 24 | 7 | 3 | 7 | 0.90 | 0.89 | 0.69 | 0.79 | 17 | 17 | 8 | 0.85 | 9 |
| 3 | 0.01 | 0.62 | 7 | 0 | 17 | 8 | 21 | 8 | 1 | 8 | 0.90 | 0.88 | 0.67 | 0.77 | 20 | 20 | 7 | 0.86 | 8 |
| 3.50 | 0.01 | 1.15 | 7 | 0 | 17 | 8 | 18 | 8 | 0 | 8 | 0.88 | 0.86 | 0.66 | 0.76 | 22 | 22 | 7 | 0.80 | 8 |
| 4 | 0.00 | 1.44 | 7 | 0 | 17 | 8 | 12 | 7 | 0 | 7 | 0.88 | 0.87 | 0.67 | 0.76 | 24 | 24 | 7 | 0.82 | 8 |
| 4.50 | 0.00 | 1.82 | 7 | 0 | 17 | 8 | 7 | 7 | 0 | 7 | 0.88 | 0.86 | 0.67 | 0.76 | 25 | 25 | 7 | 0.83 | 8 |
| 5 | 0.00 | 2.74 | 8 | 0 | 16 | 8 | 1 | 1 | 0 | 1 | 0.89 | 0.88 | 0.48 | 0.55 | 32 | 32 | 8 | 0.83 | 9 |
| 5.50 | 0.00 | 3.92 | 7 | 0 | 18 | 7 | 0 | 0 | 0 | 0 | 0.90 | 0.89 |  |  |  |  |  | 0.85 | 8 |
| 6 | 0.00 | 5.18 | 8 | 0 | 17 | 7 | 0 | 0 | 0 | 0 | 0.93 | 0.92 |  |  |  |  |  | 0.86 | 9 |
| 6.50 | 0.00 | 7.02 | 8 | 0 | 17 | 7 | 0 | 0 | 0 | 0 | 0.92 | 0.91 |  |  |  |  |  | 0.85 | 9 |
| 7 | 0.00 | 8.96 | 8 | 0 | 17 | 7 | 0 | 0 | 0 | 0 | 0.94 | 0.93 |  |  |  |  |  | 0.85 | 9 |
| 7.50 | 0.00 | 10.62 | 10 | 0 | 15 | 7 | 0 | 0 | 0 | 0 | 0.95 | 0.94 |  |  |  |  |  | 0.80 | 11 |
| 8 | 0.00 | 12.21 | 9 | 0 | 15 | 8 | 0 | 0 | 0 | 0 | 0.93 | 0.92 |  |  |  |  |  | 0.83 | 10 |
| 8.50 | 0.00 | 13.85 | 9 | 0 | 16 | 7 | 0 | 0 | 0 | 0 | 0.92 | 0.91 |  |  |  |  |  | 0.85 | 10 |
| 9 | 0.00 | 15.38 | 9 | 0 | 16 | 7 | 0 | 0 | 0 | 0 | 0.91 | 0.90 |  |  |  |  |  | 0.81 | 10 |
| 9.50 | 0.00 | 15.51 | 9 | 9 | 16 | 7 | 23 | 7 | 0 | 7 | 0.92 | 0.91 | 0.77 | 0.74 | 11 | 20 | 9 | 0.83 | 10 |
| 10 | 0.01 | 14.92 | 9 | 9 | 16 | 7 | 23 | 7 | 5 | 7 | 0.93 | 0.92 | 0.82 | 0.79 | 15 | 17 | 9 | 0.88 | 10 |
| 10.50 | 0.01 | 13.88 | 9 | 9 | 16 | 7 | 23 | 7 | 2 | 7 | 0.92 | 0.91 | 0.71 | 0.81 | 18 | 18 | 9 | 0.85 | 10 |
| 11 | 0.00 | 12.64 | 9 | 9 | 17 | 6 | 23 | 6 | 4 | 6 | 0.93 | 0.92 | 0.60 | 0.69 | 18 | 18 | 9 | 0.84 | 10 |
| 11.50 | 0.01 | 11.18 | 8 | 8 | 17 | 7 | 24 | 7 | 1 | 7 | 0.94 | 0.93 | 0.84 | 0.82 | 12 | 12 | 8 | 0.87 | 9 |
| 12 | 0.01 | 9.84 | 9 | 9 | 16 | 7 | 23 | 7 | 4 | 7 | 0.93 | 0.92 | 0.84 | 0.82 | 12 | 12 | 9 | 0.85 | 10 |
| 12.50 | 0.01 | 8.10 | 9 | 9 | 16 | 7 | 23 | 7 | 6 | 7 | 0.93 | 0.92 | 0.85 | 0.82 | 11 | 11 | 9 | 0.85 | 10 |
| 13 | 0.00 | 6.55 | 9 | 9 | 16 | 7 | 23 | 7 | 7 | 7 | 0.94 | 0.93 | 0.82 | 0.80 | 12 | 12 | 9 | 0.85 | 10 |
| 13.50 | 0.00 | 5.31 | 8 | 8 | 17 | 7 | 24 | 7 | 7 | 7 | 0.94 | 0.93 | 0.81 | 0.78 | 12 | 14 | 8 | 0.87 | 9 |
| 14 | 0.00 | 4.29 | 8 | 8 | 17 | 7 | 24 | 7 | 7 | 7 | 0.94 | 0.93 | 0.81 | 0.78 | 12 | 17 | 8 | 0.86 | 9 |
| 14.50 | 0.00 | 3.40 | 8 | 8 | 17 | 7 | 24 | 7 | 7 | 7 | 0.94 | 0.94 | 0.66 | 0.76 | 16 | 16 | 8 | 0.87 | 9 |
| 15 | 0.00 | 2.69 | 8 | 8 | 17 | 7 | 24 | 7 | 7 | 7 | 0.94 | 0.93 | 0.65 | 0.74 | 23 | 23 | 8 | 0.86 | 9 |
| 15.50 | 0.00 | 2.22 | 8 | 8 | 17 | 7 | 24 | 7 | 7 | 7 | 0.94 | 0.93 | 0.69 | 0.79 | 16 | 16 | 8 | 0.87 | 9 |
| 16 | 0.00 | 1.85 | 9 | 9 | 16 | 7 | 23 | 7 | 7 | 7 | 0.95 | 0.94 | 0.69 | 0.78 | 17 | 17 | 9 | 0.86 | 10 |
| 16.50 | 0.00 | 1.58 | 8 | 8 | 17 | 7 | 24 | 7 | 7 | 7 | 0.95 | 0.94 | 0.65 | 0.74 | 18 | 18 | 8 | 0.88 | 9 |

b44a4a44: no stay proposal is box-free from +3.0 s to +9.0 s and no proposal of any kind from +5.5 s (0.1.6) / +5.0 s (0.1.4b) to +9.0 s. The overlapped actor is 15, the pedestrian, who reaches the stationary car's footprint at +6.55 s and is inside it at the first sample of every proposal from +7.0 s to +9.0 s; from +9.5 s the pedestrian is past. At +0.0 and +0.5 s (0.1.6) the proposals that are not box-free overlap actor 25, the oncoming car, 1.6-2.4 s into the plan. d2715023: the only overlapped actor is 32, the car ahead. Replans after +12.5 s have a shortened box-check horizon (the actor log ends at +16.8 s); the 0.3 m-margin usable escapes on d2715023 0.1.6 occur only at +15.5, +16.0 and +16.5 s, where 1.3 s or less of each plan is checked.

b44a4a44 restricted to the 13 replans before +6.5 s: box-free leave on 85% (0.1.6) / 77% (0.1.4b), usable escape on 85% / 62%.

### B4. What the prompt leave proposals overlap
Proposals that are leave + prompt but not box-free, all replans: which actor they first overlap and when.

| scene | version | prompt_leave_proposals | box_free | box_free_margin0p3 | overlapping | median_first_overlap_s | top_actors | median_end_x_fwd_m | median_end_y_left_m |
|---|---|---|---|---|---|---|---|---|---|
| b44a4a44 | 0.1.6 | 243 | 169 | 82 | 74 | 0.10 | actor 15: 66; actor 25: 8 | 11.92 | 1.15 |
| d2715023 | 0.1.6 | 203 | 39 | 10 | 164 | 1.70 | actor 32: 164 | 7.22 | 1.72 |
| d2715023 | 0.1.4b | 324 | 27 | 17 | 297 | 1.40 | actor 32: 297 | 5.29 | 0.18 |
| b44a4a44 | 0.1.4b | 341 | 118 | 77 | 223 | 1.10 | actor 20: 92; actor 15: 85; actor 72: 17 | 9.78 | 1.07 |

Actor 32 is the car directly ahead on d2715023, actor 20 the car directly ahead on b44a4a44, actor 15 the pedestrian (A2).

### B5. All 30 scenes, 0.1.6: replans with the car below 0.5 m/s, the human above 1.0 m/s and a stay plan chosen

| scene | n_replans | n_stationary_stay_human_going | of_which_usable_escape | of_which_usable_margin0p3 | of_which_usable_human_like | first_t_s | last_t_s | median_best_usable_rank_final | best_usable_rank_final | median_best_usable_rank_plain | median_gap_chosen_minus_best_usable_final | median_n_stay_above_best_usable_final | median_n_stay | max_car_speed_before_first_such_replan_mps | n_with_full_4s_box_horizon | usable_with_full_4s_box_horizon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 34 | 6 | 6 | 6 | 0 | 13.50 | 16.50 | 13.50 | 10 | 13.50 | 0.24 | 2 | 2 | 23.56 | 0 | 0 |
| acd22430 | 34 | 2 | 2 | 2 | 0 | 15.50 | 16.50 | 6 | 5 | 7.50 | 0.09 | 2.50 | 4.50 | 23.43 | 0 | 0 |
| b44a4a44 | 34 | 27 | 19 | 14 | 19 | 3.50 | 16.50 | 16 | 11 | 17 | 0.21 | 8 | 8 | 0.02 | 19 | 11 |
| bae463d3 | 34 | 6 | 6 | 6 | 0 | 12.50 | 15 | 12 | 9 | 12 | 0.23 | 5 | 5 | 3.59 | 1 | 1 |
| d2715023 | 34 | 22 | 9 | 0 | 8 | 4.50 | 15 | 27 | 22 | 27 | 0.89 | 11 | 11 | 0.33 | 17 | 8 |

Totals over 30 scenes / 1020 replans: 63 such replans in 5 scenes, 42 with a usable escape on the menu (28 with 0.3 m margin, 27 usable and human-like). The other 25 scenes have none. On 04749bb9, acd22430 and bae463d3 these replans are late in the clip (first/last time in the table); `max_car_speed_before_first_such_replan_mps` is the car's highest speed at any earlier replan. Replans after +12.5 s have a shortened box-check horizon; the last two columns count only replans with the full 4 s.

## Part C - controller check (0.1.6)
The standalone controller + vehicle model (t02 `Sim`, t03 fork-per-plan), replayed teacher-forced from the first recorded request to the tick, then one plan held for 2.0 s. `moved` = straight-line distance of the rig origin from its position at the tick. `driven_overlap` = the driven ego box against the recorded actor boxes during those 2 s. Plan (v): the human's ground-truth poses at 0.5 s steps expressed relative to the human's own pose at the start time; at handover and at the human-start tick the start time is the tick itself, at +5 s it is the human-start tick (the human's pull-out replayed from where the car is now, because at +5 s the human is already many metres away). Plan (vi), added: the best-ranked leave + prompt proposal whether or not it is box-free. `recorded_run_displacement_2s_m` is what the car did in the run over the same 2 s.

| scene | tick | t_after_handover_s | plan | proposal | plan_class | plan_rank_final | plan_score_final | plan_arc_2s_m | moved_1s_m | moved_2s_m | speed_at_2s_mps | peak_abs_steer_rad | lateral_at_2s_m | driven_overlap | driven_first_overlap_s | driven_overlap_margin0p3 | recorded_run_displacement_2s_m | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d2715023 | handover | 0 | i_chosen | 13 | stay/not-prompt/box-free | 1 | 0.92 | 0.21 | 0.13 | 0.23 | 0.10 | 0.23 | 0.00 | False |  | False | 0.38 |  |
| d2715023 | handover | 0 | ii_best_usable |  | nan |  |  |  |  |  |  |  |  | nan |  | nan | 0.38 | no such proposal on the menu at this tick |
| d2715023 | handover | 0 | iii_best_leave | 9 | leave/not-prompt/box-free | 12 | 0.80 | 0.72 | 0.21 | 0.76 | 0.84 | 0.74 | 0.04 | False |  | False | 0.38 |  |
| d2715023 | handover | 0 | iv_best_late_go | 9 | leave/not-prompt/box-free | 12 | 0.80 | 0.72 | 0.21 | 0.76 | 0.84 | 0.74 | 0.04 | False |  | False | 0.38 |  |
| d2715023 | handover | 0 | v_human_path | -1 | human |  |  | 1.02 | 0.32 | 1.00 | 0.78 | 0.79 | 0.09 | False |  | False | 0.38 |  |
| d2715023 | handover | 0 | vi_best_prompt_leave | 19 | leave/prompt/overlap | 15 | 0.66 | 1.69 | 0.51 | 1.59 | 1.11 | 0.36 | 0.12 | False |  | False | 0.38 |  |
| d2715023 | human_start | 1 | i_chosen | 26 | stay/not-prompt/box-free | 1 | 0.95 | 0.67 | 0.25 | 0.64 | 0.39 | 0.48 | 0.02 | False |  | False | 0.55 |  |
| d2715023 | human_start | 1 | ii_best_usable | 7 | leave/prompt/box-free | 27 | 0.10 | 2.20 | 0.56 | 2.18 | 2.03 | 0.66 | 0.42 | False |  | True | 0.55 |  |
| d2715023 | human_start | 1 | iii_best_leave | 28 | leave/not-prompt/box-free | 12 | 0.78 | 0.99 | 0.33 | 0.99 | 0.76 | 0.75 | 0.07 | False |  | False | 0.55 |  |
| d2715023 | human_start | 1 | iv_best_late_go | 28 | leave/not-prompt/box-free | 12 | 0.78 | 0.99 | 0.33 | 0.99 | 0.76 | 0.75 | 0.07 | False |  | False | 0.55 |  |
| d2715023 | human_start | 1 | v_human_path | -1 | human |  |  | 1.36 | 0.47 | 1.30 | 0.84 | 0.79 | 0.17 | False |  | False | 0.55 |  |
| d2715023 | human_start | 1 | vi_best_prompt_leave | 19 | leave/prompt/overlap | 13 | 0.57 | 1.88 | 0.56 | 1.69 | 1.13 | 0.37 | 0.12 | False |  | False | 0.55 |  |
| d2715023 | plus5s | 5 | i_chosen | 27 | stay/not-prompt/box-free | 1 | 0.95 | 0.08 | 0.07 | 0.11 | 0.08 | 0.31 | 0.00 | False |  | False | 0.09 |  |
| d2715023 | plus5s | 5 | ii_best_usable |  | nan |  |  |  |  |  |  |  |  | nan |  | nan | 0.09 | no such proposal on the menu at this tick |
| d2715023 | plus5s | 5 | iii_best_leave | 9 | leave/not-prompt/box-free | 13 | 0.57 | 0.56 | 0.18 | 0.66 | 0.74 | 0.70 | 0.03 | False |  | False | 0.09 |  |
| d2715023 | plus5s | 5 | iv_best_late_go | 9 | leave/not-prompt/box-free | 13 | 0.57 | 0.56 | 0.18 | 0.66 | 0.74 | 0.70 | 0.03 | False |  | False | 0.09 |  |
| d2715023 | plus5s | 5 | v_human_path | -1 | human |  |  | 1.36 | 0.45 | 1.29 | 0.86 | 0.79 | 0.18 | False |  | True | 0.09 |  |
| d2715023 | plus5s | 5 | vi_best_prompt_leave | 30 | leave/prompt/overlap | 19 | 0.16 | 2.12 | 0.59 | 2.00 | 1.42 | 0.52 | 0.26 | True | 1.70 | True | 0.09 |  |
| b44a4a44 | handover | 0 | i_chosen | 27 | stay/not-prompt/box-free | 1 | 0.93 | 0.07 | 0.01 | 0.04 | 0.04 | 0.00 | 0.00 | False |  | False | 0.02 |  |
| b44a4a44 | handover | 0 | ii_best_usable | 19 | leave/prompt/box-free | 16 | 0.67 | 1.53 | 0.43 | 1.55 | 1.29 | 0.15 | 0.04 | False |  | False | 0.02 |  |
| b44a4a44 | handover | 0 | iii_best_leave | 12 | leave/not-prompt/box-free | 9 | 0.86 | 0.04 | 0.01 | 0.08 | 0.22 | 0.09 | 0.00 | False |  | False | 0.02 |  |
| b44a4a44 | handover | 0 | iv_best_late_go | 12 | leave/not-prompt/box-free | 9 | 0.86 | 0.04 | 0.01 | 0.08 | 0.22 | 0.09 | 0.00 | False |  | False | 0.02 |  |
| b44a4a44 | handover | 0 | v_human_path | -1 | human |  |  | 0.00 | 0.01 | 0.03 | 0.09 | 0.09 | 0.00 | False |  | False | 0.02 |  |
| b44a4a44 | handover | 0 | vi_best_prompt_leave | 19 | leave/prompt/box-free | 16 | 0.67 | 1.53 | 0.43 | 1.55 | 1.29 | 0.15 | 0.04 | False |  | False | 0.02 |  |
| b44a4a44 | human_start | 3 | i_chosen | 15 | stay/not-prompt/overlap | 1 | 0.90 | 0.05 | 0.01 | 0.02 | 0.01 | 0.00 | 0.00 | False |  | False | 0.01 |  |
| b44a4a44 | human_start | 3 | ii_best_usable | 30 | leave/prompt/box-free | 20 | 0.67 | 3.10 | 0.74 | 3.14 | 2.83 | 0.14 | 0.18 | False |  | True | 0.01 |  |
| b44a4a44 | human_start | 3 | iii_best_leave | 26 | leave/not-prompt/overlap | 8 | 0.86 | 0.39 | 0.10 | 0.44 | 0.50 | 0.14 | 0.00 | False |  | False | 0.01 |  |
| b44a4a44 | human_start | 3 | iv_best_late_go | 26 | leave/not-prompt/overlap | 8 | 0.86 | 0.39 | 0.10 | 0.44 | 0.50 | 0.14 | 0.00 | False |  | False | 0.01 |  |
| b44a4a44 | human_start | 3 | v_human_path | -1 | human |  |  | 3.00 | 0.75 | 3.23 | 3.20 | 0.31 | 0.39 | False |  | False | 0.01 |  |
| b44a4a44 | human_start | 3 | vi_best_prompt_leave | 30 | leave/prompt/box-free | 20 | 0.67 | 3.10 | 0.74 | 3.14 | 2.83 | 0.14 | 0.18 | False |  | True | 0.01 |  |
| b44a4a44 | plus5s | 5 | i_chosen | 15 | stay/not-prompt/overlap | 1 | 0.89 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | -0.00 | True | 1.55 | True | 0.00 |  |
| b44a4a44 | plus5s | 5 | ii_best_usable | 4 | leave/prompt/box-free | 32 | 0.48 | 5.30 | 1.08 | 5.36 | 5.47 | 0.15 | 0.44 | False |  | True | 0.00 |  |
| b44a4a44 | plus5s | 5 | iii_best_leave | 16 | leave/not-prompt/overlap | 9 | 0.83 | 0.65 | 0.16 | 0.71 | 0.73 | 0.17 | 0.01 | True | 1.55 | True | 0.00 |  |
| b44a4a44 | plus5s | 5 | iv_best_late_go | 16 | leave/not-prompt/overlap | 9 | 0.83 | 0.65 | 0.16 | 0.71 | 0.73 | 0.17 | 0.01 | True | 1.55 | True | 0.00 |  |
| b44a4a44 | plus5s | 5 | v_human_path | -1 | human |  |  | 3.00 | 0.75 | 3.23 | 3.21 | 0.31 | 0.39 | True | 1.65 | True | 0.00 |  |
| b44a4a44 | plus5s | 5 | vi_best_prompt_leave | 19 | leave/prompt/overlap | 14 | 0.70 | 1.89 | 0.53 | 1.89 | 1.60 | 0.16 | 0.06 | True | 1.60 | True | 0.00 |  |

On b44a4a44 at +5 s every plan that stays near the start (and the replayed human pull-out) overlaps actor 15, the pedestrian who walks through the car's position (first overlap +1.55 to +1.65 s after the tick).

## Numbers next to the stated predictions (no verdicts)

- S1 (> 1 m within 2 s for a prompt leave proposal through the controller): moved at 2 s, plans (ii)/(vi): d2715023 1.59-2.18 m over 4 runs; b44a4a44 1.55-5.36 m over 6 runs.
- S2 (human starts within 3 s): d2715023 +0.99 s, b44a4a44 +2.98 s.
- S3 (d2715023 0.1.6, usable escape on < 25 % of replans): 55.9% of 34 at 0 m margin, 8.8% at 0.3 m margin; 0.1.4b 26.5% / 11.8%. Restricted to the 26 replans with a full 4 s box horizon (t <= 12.5 s): 0.1.6 57.7% / 0.0%, 0.1.4b 19.2% / 0.0%. Where a usable escape exists on d2715023 0.1.6 the median number of usable proposals is 2 of 32 and the median best usable score is 0.056.
- S4 (b44a4a44 0.1.6, box-free leave on >= 75 %, fewer than half of those prompt): box-free leave on 76.5% of 34 replans (84.6% of the 13 replans before +6.5 s; 69.2% of the 26 replans with a full 4 s horizon); of those replans 100% have a prompt box-free leave. Per replan, median 17 late-go and 7 prompt among 24 leave proposals.
- S5 (best usable-escape rank worse than 5 on >= 90 % of replans that have one): final score: d2715023 100%, b44a4a44 100% (0.1.6); plain score the same. Below every stay proposal: 100% / 100% (0.1.6), 22% / 100% (0.1.4b). Median gap (chosen minus best usable, final score): d2715023 0.883 (0.1.6) vs 0.558 (0.1.4b); b44a4a44 0.219 vs 0.512. Plain score: d2715023 0.870 vs 0.638; b44a4a44 0.143 vs 0.442. The medians are over different sets of replans per version (those with a usable escape: 19 vs 9 on d2715023, 26 vs 23 on b44a4a44).
- S6 (same pattern on another scene): table B5 - besides the two stuck scenes, 04749bb9 (6 of 6 replans, +13.5..+16.5 s), acd22430 (2 of 2 replans, +15.5..+16.5 s), bae463d3 (6 of 6 replans, +12.5..+15.0 s).

## Plots
- `plot1_topview_d2715023_016.png`
- `plot2_topview_b44a4a44_016.png`
- `plot3_topview_d2715023_014b.png`
- `plot4_topview_b44a4a44_014b.png`
- `plot5_scores_vs_time.png`
- `plot6_class_counts_016.png`
- `plot7_controller_check.png`
- `plot8_moving_scenes_scores.png`

## Files
`proposals.csv` (8704 rows: 4 scenes x 2 versions x 34 replans x 32 proposals), `replans.csv` (272 rows), `scene_summary.csv`, `all30_stationary_stays.csv`, `controller_check.csv`, `human_timeline.csv`, `boxcheck_validation.csv`, plus `front_actors.csv`, `model_inputs.csv`, `metrics.json`.
