# T02 - highway swerve after retraining: recorded plans through the standalone steering controller, counterfactuals, menus

Offline, CPU only. Scripts (this folder): `t02_common.py` (log reader + standalone controller), `t02_gate.py` (Part A), `t02_run.py` (Parts B, C),
`t02_plots.py`, `t02_readme.py` (this file is generated from the CSVs). Interpreter: `{runs}/alpasim_bulk/nurec-venv/bin/python`
(do_mpc 5.1.2 / casadi 3.8.0), controller imported from the read-only `alpasim/src/controller`. Gains and horizon asserted equal to each run's
`controller-config.yaml` (nonlinear, n_horizon 20, dt 0.1 s, idx_start_penalty 10, weights 2.0/1.0/1.0/0.1/5.0/1.0).

"Steering" below = `front_steering_angle` (the vehicle model's steering state, bounded at +-0.785 rad; the quantity t01 thresholded at 0.3 rad).
The commanded value `u_steering_angle` is in the CSVs as `*_cmd`.

## 0. What the controller actually receives (read from the 0.1.6 and 0.1.4b rollout logs and `divider.py`)

- Every 50 ms tick the runtime sends one request covering 50 ms with `planned_trajectory_in_rig` = poses **0.5 s apart**, pose 0 = the rig origin
  at t = now (yaw 0). Replan ticks (every 10th tick, the handover tick is one): **9 poses** = origin + the model's 8 waypoints. The 9 ticks in between:
  **8 poses** = the same plan linearly re-sampled at now + 0.5 k s and re-expressed in the current rig frame (`divider.py::_rebuild`).
  So "the 20 Hz plan" is a 2 Hz-spaced plan re-sent 20 times a second, not a plan with 20 Hz points. The MPC solves every 100 ms on the latest one.
- `coerce_dynamic_state` is True on every tick up to and including the handover tick and False afterwards.
- Source facts re-read: reference heading = `pose.yaw()` of the interpolated plan pose (`nonlinear_mpc.py::_tvp_fun`), tracking term enabled only for
  horizon steps k >= 10 (1.0 s to 2.0 s ahead).

Relation of the handover controller plan to the raw 8 waypoints of the regenerated menu (chosen proposal), and of the cached-tick plans to the replan plan:

| version | scene | max_abs_pos_diff_ctrl_plan_vs_raw_waypoints_m | max_abs_yaw_diff_rad | pose0_xyyaw | cached_ticks_max_world_dev_from_resampled_replan_plan_m |
|---|---|---|---|---|---|
| 0.1.6 | 04749bb9 | 0.000023 | 0 | [0.0, 0.0, 0.0] | 0.002258 |
| 0.1.6 | acd22430 | 0.000023 | 0 | [0.0, 0.0, 0.0] | 0.001057 |
| 0.1.4b | 04749bb9 | 0.000023 | 0 | [0.0, 0.0, 0.0] | 0.002446 |
| 0.1.4b | acd22430 | 0.000031 | 0 | [0.0, 0.0, 0.0] | 0.001306 |

## Part A - gate

Three standalone replays per case, each fed the recorded plan of every 50 ms tick for 3 s: `cold_tf` = fresh controller + vehicle model seeded with
the recorded `controller_request.state` at the start tick, recorded pose override each tick (what the runtime did); `warm_tf` = same but the
controller was first replayed from the session's first request; `cold_free` = fresh, the simulator's own pose, plans held in the world frame.
Differences are replay minus the run's controller CSV; pct = max difference / recorded peak in that window (meaningless where the recorded peak is ~0).


| scene | start (s after handover) | mode | steer maxdiff 0.5s (rad) | steer rms 0.5s | rec peak steer 0.5s | steer pct 0.5s | long maxdiff 0.5s (m/s2) | long rms 0.5s | rec peak |long| 0.5s | long pct 0.5s | steer maxdiff 3s | steer rms 3s | long maxdiff 3s | long rms 3s | first >0.3 rec (s) | first >0.3 replay (s) | peak steer 3s rec | peak steer 3s replay | solves not 'solved' |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04749bb9 | 0 | cold_tf | 0.0028 | 0.0016 | 0.5337 | 0.5185 | 0.0967 | 0.0538 | 5.2099 | 1.857 | 0.0028 | 0.001 | 0.0967 | 0.0246 | 0.1 | 0.1 | 0.6852 | 0.6856 | 0 |
| 04749bb9 | 0 | warm_tf | 0 | 0 | 0.5337 | 0 | 0 | 0 | 5.2099 | 0 | 0 | 0 | 0 | 0 | 0.1 | 0.1 | 0.6852 | 0.6852 | 0 |
| 04749bb9 | 0 | cold_free | 0.0056 | 0.0032 | 0.5337 | 1.0536 | 0.0967 | 0.0538 | 5.2099 | 1.857 | 0.0056 | 0.0022 | 0.0967 | 0.0263 | 0.1 | 0.1 | 0.6852 | 0.6872 | 0 |
| acd22430 | 0 | cold_tf | 0.0053 | 0.0037 | 0.3897 | 1.3601 | 0.0769 | 0.0486 | 3.504 | 2.194 | 0.0053 | 0.0015 | 0.0769 | 0.0231 | 0.15 | 0.1 | 0.7819 | 0.782 | 0 |
| acd22430 | 0 | warm_tf | 0 | 0 | 0.3897 | 0 | 0 | 0 | 3.504 | 0 | 0 | 0 | 0 | 0 | 0.15 | 0.15 | 0.7819 | 0.7819 | 0 |
| acd22430 | 0 | cold_free | 0.0114 | 0.0067 | 0.3897 | 2.9381 | 0.0769 | 0.0491 | 3.504 | 2.194 | 0.0138 | 0.0045 | 0.0769 | 0.0262 | 0.15 | 0.1 | 0.7819 | 0.782 | 0 |
| bd07fb91 | 0 | cold_tf | 0.0031 | 0.0019 | 0.0095 | 32.3895 | 0.5699 | 0.3001 | 0.4788 | 119.0386 | 0.0031 | 0.0008 | 0.5699 | 0.1372 |  |  | 0.0106 | 0.0107 | 0 |
| bd07fb91 | 0 | warm_tf | 0 | 0 | 0.0095 | 0 | 0 | 0 | 0.4788 | 0 | 0 | 0 | 0 | 0 |  |  | 0.0106 | 0.0106 | 0 |
| bd07fb91 | 0 | cold_free | 0.0031 | 0.0019 | 0.0095 | 32.3895 | 0.5699 | 0.3006 | 0.4788 | 119.0386 | 0.0031 | 0.001 | 0.5699 | 0.1509 |  |  | 0.0106 | 0.0118 | 0 |
| bd07fb91 | 5.5 | cold_tf | 0.0037 | 0.0026 | 0.024 | 15.2381 | 0.3718 | 0.1982 | 0.6235 | 59.6407 | 0.0132 | 0.003 | 0.3718 | 0.1017 | 0.6 | 0.6 | 0.7761 | 0.777 | 0 |
| bd07fb91 | 5.5 | warm_tf | 0 | 0 | 0.024 | 0 | 0 | 0 | 0.6235 | 0 | 0 | 0 | 0 | 0 | 0.6 | 0.6 | 0.7761 | 0.7761 | 0 |
| bd07fb91 | 5.5 | cold_free | 0.0036 | 0.0025 | 0.024 | 14.8602 | 0.3718 | 0.2057 | 0.6235 | 59.6407 | 0.0277 | 0.0092 | 0.3718 | 0.1221 | 0.6 | 0.6 | 0.7761 | 0.7831 | 0 |
| 02e075b9 | 0 | cold_tf | 0.0013 | 0.0007 | 0.0014 | 88.2987 | 0.1848 | 0.0973 | 4.0051 | 4.6147 | 0.0013 | 0.0003 | 0.1848 | 0.0445 |  |  | 0.0014 | 0.0015 | 0 |
| 02e075b9 | 0 | warm_tf | 0 | 0 | 0.0014 | 0 | 0 | 0 | 4.0051 | 0 | 0 | 0 | 0 | 0 |  |  | 0.0014 | 0.0014 | 0 |
| 02e075b9 | 0 | cold_free | 0.0013 | 0.0008 | 0.0014 | 88.2987 | 0.1848 | 0.0975 | 4.0051 | 4.6147 | 0.0013 | 0.0004 | 0.1848 | 0.0492 |  |  | 0.0014 | 0.0009 | 0 |

**Gate: passes.** `warm_tf` reproduces the recorded steering and longitudinal command to floating-point precision on all five cases
(max |diff| over 3 s: steering 1.4e-15 rad, longitudinal 1.2e-14 m/s2). The fresh-controller replay
(`cold_tf`) on the two swerving highway scenes is within 1.4 % (steering) and 2.2 % (longitudinal) of the recorded peak in the first 0.5 s.
bd07fb91 (both starts) and the control 02e075b9 have recorded steering < 0.03 rad in the first 0.5 s, so only absolute differences mean anything there
(<= 0.0037 rad). Negative control 02e075b9: replayed peak |steer| over 3 s = 0.0015 rad.
bd07fb91 from +5.5 s: first |steer| > 0.3 rad at +0.60 s after the start (= handover + 6.10 s), recorded and replayed. Plot 1.

All Part B / C controller runs use the `warm_tf` state at the tick (replay of the recorded requests from session start), then one request with the
variant plan held fixed for 1.0 s (10 solves, 50 ms reporting). `B0_recorded_coldseed` is the STEP2-R4 seeding for comparison.

## Part B - counterfactual plans (0.1.6 run state; plan held 1.0 s)

Constructions (all 9 poses, 0.5 s apart, rig frame at the tick, pose 0 = origin; `Path` = polyline through the recorded 9 poses, arc-length parametrised, continued straight past its end):
- **B0** recorded plan. **B1** points at arc length s = v0 t on the recorded path (v0 = car speed at the tick), heading = recorded heading re-sampled at the same s.
- **B2** recorded positions/timing, heading = path tangent (np.gradient finite differences, 3-point moving average). **B2z** heading = 0 (pose 0's heading) on all poses.
- **B3** x = recorded arc length per pose, y = 0, recorded heading.
- **B4(a)** s(t) = v0 t - a t^2/2 (held after stopping) on the recorded path or on a straight line, heading = tangent. **B4e** = B4 with the a that reaches the recorded plan's 4 s arc length (column `const_decel_to_recorded_4s_endpoint_mps2`).
- **B5** recorded step lengths with step 1 := v0 * 0.5 s and the difference subtracted equally from steps 2..8 (same total length), positions/headings re-sampled on the path.
- **B6** clip ground-truth ego trajectory at tick + 0.5 k s in the car's rig frame at the tick, pose 0 forced to the origin (at handover the human is at the car; at +0.5 s the human pose is 2.1-2.5 m beside the car and 0.43 rad off its heading).
- **B7** x = v0 * 0.5 k, y = 0, heading 0.

Columns: first>0.3 = time of first |steer| > 0.3 rad; sign chg = sign changes of steering (|steer| > 0.02); lat = peak |lateral displacement| in the tick's rig frame;
a1 = first-0.5 s demanded acceleration (step-1 speed - v0)/0.5; a@1s, a@2s = constant acceleration that would reach the plan's 1 s / 2 s arc length.

| ver | scene | tick | variant | v0 | peak steer (rad) | peak cmd | first>0.3 (s) | sign chg | lat (m) | u_long min | u_long max | a1 | a@1s | a@2s | mean v 4s | step spread | hdg-tangent max (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1.6 | 04749bb9 | 0 | B0_recorded | 23.563 | 0.525 | 0.74 | 0.1 | 1 | 3.805 | -5.158 | -2.03 | -6.498 | -7.109 | -4.051 | 18.855 | 17.307 | 0.815 |
| 0.1.6 | 04749bb9 | 0 | B1_no_braking | 23.563 | 0.003 | 0.003 |  | 0 | 0.061 | 0.016 | 0.134 | -0 | -0 | -0 | 23.563 | 0 | 0.996 |
| 0.1.6 | 04749bb9 | 0 | B2_heading_tangent | 23.563 | 0.525 | 0.74 | 0.1 | 1 | 3.805 | -5.158 | -2.03 | -6.498 | -7.109 | -4.051 | 18.855 | 17.307 | 0.15 |
| 0.1.6 | 04749bb9 | 0 | B2z_heading_first_pose | 23.563 | 0.525 | 0.74 | 0.1 | 1 | 3.804 | -5.158 | -2.03 | -6.498 | -7.109 | -4.051 | 18.855 | 17.307 | 1.492 |
| 0.1.6 | 04749bb9 | 0 | B3_straightened | 23.563 | 0.52 | 0.737 | 0.1 | 1 | 3.839 | -5.163 | -2.035 | -6.498 | -7.109 | -4.051 | 18.855 | 17.307 | 0.722 |
| 0.1.6 | 04749bb9 | 0 | B5_first_step_respaced | 23.563 | 0.151 | 0.166 |  | 1 | 1.586 | -5.052 | -1.975 | -0 | -4.324 | -3.587 | 18.855 | 17.307 | 0.788 |
| 0.1.6 | 04749bb9 | 0 | B4e_const_decel_same_endpoint | 23.563 | 0.005 | 0.005 |  | 0 | 0.025 | -3.372 | -1.144 | -1.177 | -2.354 | -2.354 | 18.855 | 8.24 | 0.084 |
| 0.1.6 | 04749bb9 | 0 | B6_human_future | 23.563 | 0.001 | 0.002 |  | 0 | 0.028 | 0.183 | 0.245 | -0.053 | 0.103 | 0.13 | 23.743 | 0.341 | 0.114 |
| 0.1.6 | 04749bb9 | 0 | B7_cv_floor_straight | 23.563 | 0.001 | 0.001 |  | 0 | 0.006 | 0.016 | 0.134 | 0 | 0 | 0 | 23.563 | 0 | 0 |
| 0.1.6 | 04749bb9 | 0 | B0_recorded_coldseed | 23.563 | 0.519 | 0.732 | 0.1 | 1 | 3.797 | -5.137 | -2.126 | -6.498 | -7.109 | -4.051 | 18.855 | 17.307 | 0.815 |
| 0.1.6 | 04749bb9 | 0.5 | B0_recorded | 22.185 | 0.685 | 0.774 | 0.05 | 2 | 5.829 | -9 | -7.367 | -12.525 | -13.312 | -7.791 | 12.828 | 11.949 | 23.665 |
| 0.1.6 | 04749bb9 | 0.5 | B1_no_braking | 22.185 | 0.402 | 0.198 | 0.05 | 2 | 6.677 | -3.063 | 1.052 | -0.329 | -0.169 | -0.268 | 22.051 | 0.614 | 11.601 |
| 0.1.6 | 04749bb9 | 0.5 | B2_heading_tangent | 22.185 | 0.685 | 0.774 | 0.05 | 2 | 5.826 | -9 | -7.364 | -12.525 | -13.312 | -7.791 | 12.828 | 11.949 | 5.733 |
| 0.1.6 | 04749bb9 | 0.5 | B2z_heading_first_pose | 22.185 | 0.686 | 0.775 | 0.05 | 2 | 5.901 | -9 | -7.375 | -12.525 | -13.312 | -7.791 | 12.828 | 11.949 | 52.216 |
| 0.1.6 | 04749bb9 | 0.5 | B3_straightened | 22.185 | 0.731 | 0.934 | 0.05 | 2 | 3.247 | -9 | -7.04 | -12.525 | -13.312 | -7.791 | 12.828 | 11.949 | 37.454 |
| 0.1.6 | 04749bb9 | 0.5 | B5_first_step_respaced | 22.185 | 0.465 | 0.358 | 0.05 | 2 | 5.228 | -8.279 | -6.631 | -0.329 | -8.111 | -6.982 | 12.751 | 16.487 | 21.72 |
| 0.1.6 | 04749bb9 | 0.5 | B4e_const_decel_same_endpoint | 22.185 | 0.415 | 0.231 | 0.05 | 2 | 5.396 | -5.634 | -5.068 | -2.622 | -4.824 | -4.81 | 12.72 | 16.235 | 5.758 |
| 0.1.6 | 04749bb9 | 0.5 | B6_human_future | 22.185 | 0.531 | 0.526 | 0.05 | 1 | 10.283 | -0.901 | 5.465 | 6.813 | 4.905 | 2.034 | 23.996 | 1.933 | 5.626 |
| 0.1.6 | 04749bb9 | 0.5 | B7_cv_floor_straight | 22.185 | 0.284 | 0.353 |  | 2 | 1.426 | -2.726 | 2.044 | 0 | 0 | 0 | 22.185 | 0 | 0 |
| 0.1.6 | acd22430 | 0 | B0_recorded | 23.429 | 0.401 | 0.48 | 0.15 | 1 | 3.767 | -3.475 | -1.709 | -5.45 | -5.196 | -2.812 | 20.37 | 9.021 | 1.386 |
| 0.1.6 | acd22430 | 0 | B1_no_braking | 23.429 | 0.002 | 0.001 |  | 0 | 0.067 | -0.115 | -0.029 | -0 | -0 | -0 | 23.429 | 0.001 | 1.386 |
| 0.1.6 | acd22430 | 0 | B2_heading_tangent | 23.429 | 0.401 | 0.48 | 0.15 | 1 | 3.767 | -3.475 | -1.709 | -5.45 | -5.196 | -2.812 | 20.37 | 9.021 | 0.149 |
| 0.1.6 | acd22430 | 0 | B2z_heading_first_pose | 23.429 | 0.401 | 0.48 | 0.15 | 1 | 3.766 | -3.475 | -1.709 | -5.45 | -5.196 | -2.812 | 20.37 | 9.021 | 0.85 |
| 0.1.6 | acd22430 | 0 | B3_straightened | 23.429 | 0.4 | 0.481 | 0.15 | 1 | 3.821 | -3.478 | -1.707 | -5.45 | -5.196 | -2.812 | 20.37 | 9.021 | 1.493 |
| 0.1.6 | acd22430 | 0 | B5_first_step_respaced | 23.429 | 0.004 | 0.005 |  | 0 | 0.02 | -3.755 | -1.578 | -0 | -2.86 | -2.423 | 20.37 | 9.021 | 1.386 |
| 0.1.6 | acd22430 | 0 | B4e_const_decel_same_endpoint | 23.429 | 0.001 | 0.002 |  | 0 | 0.06 | -2.315 | -0.941 | -0.765 | -1.53 | -1.53 | 20.37 | 5.354 | 0.164 |
| 0.1.6 | acd22430 | 0 | B6_human_future | 23.429 | 0.003 | 0.003 |  | 0 | 0.151 | -0.336 | -0.156 | -0.128 | -0.209 | -0.151 | 23.327 | 0.194 | 0.264 |
| 0.1.6 | acd22430 | 0 | B7_cv_floor_straight | 23.429 | 0.002 | 0.002 |  | 0 | 0.02 | -0.115 | -0.029 | 0 | 0 | 0 | 23.429 | 0 | 0 |
| 0.1.6 | acd22430 | 0 | B0_recorded_coldseed | 23.429 | 0.407 | 0.493 | 0.1 | 1 | 3.779 | -3.482 | -1.632 | -5.45 | -5.196 | -2.812 | 20.37 | 9.021 | 1.386 |
| 0.1.6 | acd22430 | 0.5 | B0_recorded | 22.498 | 0.612 | 0.991 | 0.05 | 2 | 2.45 | -8.612 | -3.524 | -12.238 | -12.032 | -6.881 | 15.137 | 17.822 | 124.343 |
| 0.1.6 | acd22430 | 0.5 | B1_no_braking | 22.498 | 0.205 | 0.271 |  | 2 | 0.907 | -1.867 | 1.248 | -0.067 | -0.159 | -0.104 | 21.371 | 8.372 | 124.343 |
| 0.1.6 | acd22430 | 0.5 | B2_heading_tangent | 22.498 | 0.618 | 0.999 | 0.05 | 2 | 2.44 | -8.614 | -3.528 | -12.238 | -12.032 | -6.881 | 15.137 | 17.822 | 38.607 |
| 0.1.6 | acd22430 | 0.5 | B2z_heading_first_pose | 22.498 | 0.621 | 1.004 | 0.05 | 2 | 2.431 | -8.616 | -3.531 | -12.238 | -12.032 | -6.881 | 15.137 | 17.822 | 117.036 |
| 0.1.6 | acd22430 | 0.5 | B3_straightened | 22.498 | 0.61 | 0.9 | 0.05 | 2 | 3.45 | -8.447 | -3.309 | -12.238 | -12.032 | -6.881 | 15.137 | 17.822 | 8.191 |
| 0.1.6 | acd22430 | 0.5 | B5_first_step_respaced | 22.498 | 0.324 | 0.339 | 0.75 | 1 | 1.145 | -7.077 | -3.727 | -0.067 | -6.899 | -6.049 | 14.929 | 20.707 | 124.343 |
| 0.1.6 | acd22430 | 0.5 | B4e_const_decel_same_endpoint | 22.498 | 0.216 | 0.274 |  | 2 | 0.828 | -4.097 | -3.78 | -1.899 | -3.818 | -3.733 | 14.644 | 16.446 | 5.883 |
| 0.1.6 | acd22430 | 0.5 | B6_human_future | 22.498 | 0.483 | 0.582 | 0.05 | 1 | 9.988 | -0.715 | 3.457 | 4.225 | 2.861 | 1.103 | 23.491 | 1.394 | 5.063 |
| 0.1.6 | acd22430 | 0.5 | B7_cv_floor_straight | 22.498 | 0.184 | 0.191 |  | 2 | 0.974 | -1.844 | 1.368 | 0 | 0 | 0 | 22.498 | 0 | 0 |
| 0.1.4b | 04749bb9 | 0 | B0_recorded | 23.563 | 0.776 | 1.428 | 0.05 | 2 | 4.378 | -9 | -7.203 | -14.318 | -17.992 | -9.886 | 13.039 | 5.754 | 0.685 |
| 0.1.4b | 04749bb9 | 0 | B0_recorded_coldseed | 23.563 | 0.776 | 1.461 | 0.05 | 2 | 4.398 | -9 | -8.021 | -14.318 | -17.992 | -9.886 | 13.039 | 5.754 | 0.685 |
| 0.1.4b | acd22430 | 0 | B0_recorded | 23.429 | 0.771 | 1.404 | 0.05 | 2 | 4.344 | -9 | -7.282 | -17.301 | -17.823 | -9.805 | 13.004 | 3.026 | 1.242 |
| 0.1.4b | acd22430 | 0 | B0_recorded_coldseed | 23.429 | 0.776 | 1.442 | 0.05 | 2 | 4.442 | -9 | -7.825 | -17.301 | -17.823 | -9.805 | 13.004 | 3.026 | 1.242 |

Reduction of peak |steer| relative to B0 (same scene, same tick):

| variant | 04749bb9 +0.0s | 04749bb9 +0.5s | acd22430 +0.0s | acd22430 +0.5s |
|---|---|---|---|---|
| B0_recorded | 0 | 0 | 0 | 0 |
| B1_no_braking | 99.5 | 41.3 | 99.6 | 66.5 |
| B2_heading_tangent | 0 | -0 | -0 | -1 |
| B2z_heading_first_pose | 0 | -0.1 | -0 | -1.5 |
| B3_straightened | 0.9 | -6.7 | 0.2 | 0.4 |
| B5_first_step_respaced | 71.3 | 32.2 | 99 | 47.2 |
| B4e_const_decel_same_endpoint | 99.1 | 39.4 | 99.6 | 64.8 |
| B6_human_future | 99.7 | 22.5 | 99.2 | 21.2 |
| B7_cv_floor_straight | 99.9 | 58.6 | 99.5 | 69.9 |

### B0 plan anatomy at the handover replan (0.1.6 and 0.1.4b)

step speed = distance between consecutive poses / 0.5 s; raw = the 8 model waypoints of the regenerated menu; ctrl = the plan in the controller request; implied accel = change of step speed / 0.5 s (step 1 against the car speed).

| ver | scene | pose | t (s) | step v raw | step v ctrl | step v human | implied accel (m/s2) | lateral y (m) | lateral from human path (m) | heading (deg) | heading - tangent (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1.6 | 04749bb9 | 1 | 0.5 | 20.31 | 20.31 | 23.54 | -6.5 | -0.01 | -0.02 | -0.2 | -0.09 |
| 0.1.6 | 04749bb9 | 2 | 1 | 19.7 | 19.7 | 23.69 | -1.22 | -0.04 | -0.06 | -0.36 | -0.09 |
| 0.1.6 | 04749bb9 | 3 | 1.5 | 19.67 | 19.67 | 23.68 | -0.06 | -0.1 | -0.13 | -0.42 | -0.02 |
| 0.1.6 | 04749bb9 | 4 | 2 | 18.36 | 18.36 | 23.86 | -2.64 | -0.17 | -0.19 | -0.61 | -0.36 |
| 0.1.6 | 04749bb9 | 5 | 2.5 | 17.31 | 17.31 | 23.74 | -2.08 | -0.18 | -0.15 | -0.36 | -0.2 |
| 0.1.6 | 04749bb9 | 6 | 3 | 19.94 | 19.94 | 23.79 | 5.26 | -0.22 | -0.13 | -0.19 | 0.34 |
| 0.1.6 | 04749bb9 | 7 | 3.5 | 26.42 | 26.42 | 23.88 | 12.95 | -0.39 | -0.19 | -0.1 | 0.81 |
| 0.1.6 | 04749bb9 | 8 | 4 | 9.11 | 9.11 | 23.76 | -34.61 | -0.51 | -0.28 | -0.72 | 0.77 |
| 0.1.6 | acd22430 | 1 | 0.5 | 20.7 | 20.7 | 23.37 | -5.45 | 0.02 | -0.01 | 0.22 | 0.11 |
| 0.1.6 | acd22430 | 2 | 1 | 20.96 | 20.96 | 23.28 | 0.51 | 0.04 | -0.08 | 0.08 | -0.15 |
| 0.1.6 | acd22430 | 3 | 1.5 | 20.69 | 20.69 | 23.25 | -0.54 | 0.11 | -0.13 | 0.35 | 0.01 |
| 0.1.6 | acd22430 | 4 | 2 | 20.12 | 20.12 | 23.22 | -1.14 | 0.16 | -0.21 | 0.35 | -0.11 |
| 0.1.6 | acd22430 | 5 | 2.5 | 19.27 | 19.27 | 23.33 | -1.69 | 0.26 | -0.25 | 0.97 | 0.33 |
| 0.1.6 | acd22430 | 6 | 3 | 26.36 | 26.36 | 23.4 | 14.18 | 0.42 | -0.34 | 1.28 | 0.43 |
| 0.1.6 | acd22430 | 7 | 3.5 | 17.34 | 17.34 | 23.41 | -18.04 | 0.59 | -0.37 | 1.28 | 0.67 |
| 0.1.6 | acd22430 | 8 | 4 | 17.52 | 17.52 | 23.36 | 0.36 | 0.6 | -0.58 | 1.49 | 1.39 |
| 0.1.4b | 04749bb9 | 1 | 0.5 | 16.4 | 16.4 | 23.54 | -14.32 | -0 | -0.01 | -0.05 | -0.07 |
| 0.1.4b | 04749bb9 | 2 | 1 | 12.73 | 12.73 | 23.69 | -7.35 | 0.01 | -0.01 | -0.03 | 0.02 |
| 0.1.4b | 04749bb9 | 3 | 1.5 | 13.51 | 13.51 | 23.68 | 1.56 | -0.01 | -0.04 | -0.19 | 0.02 |
| 0.1.4b | 04749bb9 | 4 | 2 | 12.07 | 12.07 | 23.86 | -2.89 | -0.04 | -0.07 | -0.32 | -0.2 |
| 0.1.4b | 04749bb9 | 5 | 2.5 | 14.15 | 14.15 | 23.74 | 4.17 | -0.04 | -0.08 | 0.01 | 0.22 |
| 0.1.4b | 04749bb9 | 6 | 3 | 13.17 | 13.17 | 23.79 | -1.97 | -0.09 | -0.1 | 0.17 | 0.69 |
| 0.1.4b | 04749bb9 | 7 | 3.5 | 11.64 | 11.64 | 23.88 | -3.05 | -0.15 | -0.13 | 0.2 | 0.4 |
| 0.1.4b | 04749bb9 | 8 | 4 | 10.65 | 10.65 | 23.76 | -1.98 | -0.13 | -0.07 | 0.5 | 0.24 |
| 0.1.4b | acd22430 | 1 | 0.5 | 14.78 | 14.78 | 23.37 | -17.3 | 0.01 | -0.01 | 0.15 | -0.07 |
| 0.1.4b | acd22430 | 2 | 1 | 14.26 | 14.26 | 23.28 | -1.04 | 0.06 | -0 | 0.33 | -0.14 |
| 0.1.4b | acd22430 | 3 | 1.5 | 13.23 | 13.23 | 23.25 | -2.05 | 0.12 | -0.01 | 0.54 | 0.07 |
| 0.1.4b | acd22430 | 4 | 2 | 12.23 | 12.23 | 23.22 | -2 | 0.16 | -0.04 | 0.84 | 0.16 |
| 0.1.4b | acd22430 | 5 | 2.5 | 12.87 | 12.87 | 23.33 | 1.27 | 0.27 | -0.01 | 1.19 | 0.19 |
| 0.1.4b | acd22430 | 6 | 3 | 11.75 | 11.75 | 23.4 | -2.23 | 0.37 | 0.03 | 1.2 | 0.59 |
| 0.1.4b | acd22430 | 7 | 3.5 | 12.67 | 12.67 | 23.41 | 1.84 | 0.4 | -0.04 | 1.28 | 0.88 |
| 0.1.4b | acd22430 | 8 | 4 | 12.25 | 12.25 | 23.36 | -0.85 | 0.46 | -0.07 | 1.86 | 1.24 |

Plans of the 9 cached ticks after handover (re-anchored, per-tick): full per-pose table in `plan_anatomy.csv` (`source = cached_tick`). Car-to-first-pose distance / 0.5 s and lateral offset of the first pose, 0.1.6:

| tick_after_handover_s | heading_deg 04749bb9 | heading_deg acd22430 | step_speed_ctrl_plan_mps 04749bb9 | step_speed_ctrl_plan_mps acd22430 | y_lateral_m 04749bb9 | y_lateral_m acd22430 |
|---|---|---|---|---|---|---|
| 0.05 | -0.59 | 0.46 | 19.93 | 20.46 | -0.08 | 0.07 |
| 0.1 | -2.59 | 2 | 19.55 | 20.21 | -0.42 | 0.34 |
| 0.15 | -6.45 | 5.07 | 19.18 | 19.97 | -1.08 | 0.9 |
| 0.2 | -11.71 | 9.35 | 18.82 | 19.73 | -2 | 1.68 |
| 0.25 | -17.66 | 14.26 | 18.5 | 19.52 | -3.06 | 2.62 |
| 0.3 | -23.17 | 18.88 | 18.25 | 19.36 | -4.09 | 3.56 |
| 0.35 | -27.46 | 22.66 | 18.12 | 19.28 | -4.98 | 4.41 |
| 0.4 | -29.36 | 24.93 | 18.16 | 19.31 | -5.61 | 5.1 |
| 0.45 | -28.58 | 25.51 | 18.33 | 19.45 | -5.96 | 5.58 |

### Position jump between consecutive plans at each replan

start jump = distance from where the previous replan's plan put the car 0.5 s later (its pose 1) to the car's actual pose at the new replan; along / lateral / heading are the car's pose seen from that old pose; last column = new plan's pose 1 vs old plan's pose 2 (same instant).

| version | scene | replan_after_handover_s | ego_speed_mps | start_jump_m | start_jump_along_old_plan_m | start_jump_lateral_m | start_heading_diff_deg | jump_at_plus_0p5s_new_vs_old_m |
|---|---|---|---|---|---|---|---|---|
| 0.1.6 | 04749bb9 | 0.5 | 22.19 | 2.62 | 0.89 | 2.46 | 24.79 | 5.5 |
| 0.1.6 | 04749bb9 | 1 | 18.47 | 3.72 | 1.5 | -3.41 | -25.56 | 6.07 |
| 0.1.6 | 04749bb9 | 1.5 | 18.65 | 1.39 | 0.09 | 1.39 | 3.71 | 2.53 |
| 0.1.6 | 04749bb9 | 2 | 20.01 | 1.88 | 0.69 | -1.75 | -21.82 | 5.41 |
| 0.1.6 | 04749bb9 | 2.5 | 21.24 | 1 | 0.44 | 0.89 | 6.51 | 2.04 |
| 0.1.6 | 04749bb9 | 3 | 21.91 | 2.66 | 1.09 | -2.42 | -22.59 | 6.03 |
| 0.1.6 | acd22430 | 0.5 | 22.5 | 2.25 | 0.86 | -2.08 | -24.18 | 6.37 |
| 0.1.6 | acd22430 | 1 | 19.38 | 3.48 | 1.66 | 3.05 | 9.45 | 5.94 |
| 0.1.6 | acd22430 | 1.5 | 18.05 | 1.07 | -0.25 | -1.04 | 3.88 | 3.32 |
| 0.1.6 | acd22430 | 2 | 16.3 | 3.01 | 1.13 | 2.79 | 21.15 | 5.21 |
| 0.1.6 | acd22430 | 2.5 | 12.47 | 2.21 | 0.24 | -2.2 | -16.61 | 3.24 |
| 0.1.6 | acd22430 | 3 | 9.09 | 0.97 | -0.03 | 0.97 | 8.11 | 1.73 |
| 0.1.4b | 04749bb9 | 0.5 | 20.41 | 3.89 | 2.01 | -3.33 | -24.34 | 5.09 |
| 0.1.4b | 04749bb9 | 1 | 15.92 | 6.21 | 6.19 | -0.58 | -48.09 | 6.22 |
| 0.1.4b | 04749bb9 | 1.5 | 11.33 | 4.4 | 3.59 | -2.54 | -26.68 | 4.55 |
| 0.1.4b | 04749bb9 | 2 | 6.81 | 2.69 | 2.67 | 0.34 | -19.49 | 3.61 |
| 0.1.4b | 04749bb9 | 2.5 | 3.62 | 0.25 | 0.06 | -0.24 | -6.78 | 0.61 |
| 0.1.4b | 04749bb9 | 3 | 1.56 | 0.19 | 0.15 | 0.12 | 3.75 | 0.11 |
| 0.1.4b | acd22430 | 0.5 | 20.39 | 4.36 | 2.8 | -3.35 | -24.08 | 5.13 |
| 0.1.4b | acd22430 | 1 | 15.9 | 5.6 | 5.57 | -0.52 | -46.78 | 5.2 |
| 0.1.4b | acd22430 | 1.5 | 11.32 | 4.42 | 3.56 | -2.62 | -25.92 | 4.5 |
| 0.1.4b | acd22430 | 2 | 6.79 | 2.59 | 2.56 | 0.41 | -17.29 | 3.66 |
| 0.1.4b | acd22430 | 2.5 | 3.56 | 0.22 | -0.02 | -0.22 | -5.44 | 0.69 |
| 0.1.4b | acd22430 | 3 | 1.38 | 0.2 | 0.14 | 0.15 | 6.16 | 0.18 |

### B4 braking sweep (handover tick of each seed scene, real recorded states; plan held 1.0 s)

Peak |steer| (rad):

| const_decel_mps2 | 04749bb9 recorded_path (23.6 m/s) | acd22430 recorded_path (23.4 m/s) | 04749bb9 straight (23.6 m/s) | acd22430 straight (23.4 m/s) | bd07fb91 straight (15.7 m/s) | d1a25a99 straight (5.0 m/s) |
|---|---|---|---|---|---|---|
| 0 | 0.003 | 0.002 | 0.001 | 0.002 | 0.003 | 0.004 |
| 1 | 0.003 | 0.002 | 0.001 | 0.002 | 0.003 | 0.004 |
| 2 | 0.004 | 0.002 | 0.001 | 0.003 | 0.003 | 0.004 |
| 3 | 0.006 | 0.003 | 0.001 | 0.004 | 0.004 | 0.004 |
| 4 | 0.013 | 0.006 | 0.002 | 0.009 | 0.004 | 0.004 |
| 5 | 0.099 | 0.118 | 0.055 | 0.124 | 0.005 | 0.005 |
| 6 | 0.237 | 0.259 | 0.24 | 0.258 | 0.008 | 0.005 |
| 7 | 0.43 | 0.456 | 0.43 | 0.455 | 0.487 | 0.004 |
| 8 | 0.693 | 0.744 | 0.698 | 0.74 | 0.785 | 0.004 |
| 9 | 0.692 | 0.683 | 0.691 | 0.683 | 0.785 | 0.004 |
| 10 | 0.776 | 0.776 | 0.776 | 0.776 | 0.785 | 0.004 |
| 12 | 0.782 | 0.782 | 0.783 | 0.783 | 0.785 | 0.004 |

Minimum commanded longitudinal acceleration (m/s2) in the same runs:

| const_decel_mps2 | 04749bb9 recorded_path | acd22430 recorded_path | 04749bb9 straight | acd22430 straight | bd07fb91 straight | d1a25a99 straight |
|---|---|---|---|---|---|---|
| 0 | 0.02 | -0.12 | 0.02 | -0.12 | -0.53 | -1.08 |
| 1 | -1.39 | -1.54 | -1.39 | -1.54 | -1.27 | -1.62 |
| 2 | -2.85 | -3 | -2.85 | -3 | -2.69 | -2.38 |
| 3 | -4.31 | -4.46 | -4.31 | -4.46 | -4.15 | -3.73 |
| 4 | -5.77 | -5.92 | -5.77 | -5.92 | -5.61 | -4.97 |
| 5 | -7.17 | -7.3 | -7.22 | -7.29 | -7.07 | -5.98 |
| 6 | -8.42 | -8.49 | -8.41 | -8.5 | -8.51 | -6.66 |
| 7 | -9 | -9 | -9 | -9 | -9 | -7.09 |
| 8 | -9 | -9 | -9 | -9 | -9 | -7.38 |
| 9 | -9 | -9 | -9 | -9 | -9 | -7.65 |
| 10 | -9 | -9 | -9 | -9 | -9 | -7.87 |
| 12 | -9 | -9 | -9 | -9 | -9 | -8.15 |

Onset = smallest swept constant deceleration with peak |steer| > 0.1 rad (sweep grid 0,1,...,10,12 m/s2; 7 and 9 added to the requested list). `onset_as_first_0p5s_accel` = the first-0.5 s demand of that onset plan (-a/2).

| seed_scene | path | ego_speed_mps | onset_const_decel_mps2 | last_decel_below_onset_mps2 | onset_as_first_0p5s_accel_mps2 | peak_steer_at_max_decel_rad |
|---|---|---|---|---|---|---|
| 04749bb9 | recorded_path | 23.563 | 6 | 5 | -3 | 0.782 |
| 04749bb9 | straight | 23.563 | 6 | 5 | -3 | 0.783 |
| acd22430 | recorded_path | 23.429 | 5 | 4 | -2.5 | 0.782 |
| acd22430 | straight | 23.429 | 5 | 4 | -2.5 | 0.783 |
| bd07fb91 | straight | 15.711 | 7 | 6 | -3.5 | 0.785 |
| d1a25a99 | straight | 5.015 |  | 12 |  | 0.004 |

## Part C - menus

Per-proposal table: `menu_proposals.csv` (32 x 36 replans). Rules fixed before reading the results:
- **clean human-like** = |mean 4 s speed - human mean speed over the same 4 s| <= 15 % of the human's AND step-speed spread < 3 m/s AND |perpendicular offset of the 2 s waypoint from the human path| < 1.0 m.
- **controller-safe** = first-0.5 s demanded acceleration > threshold(v), threshold = the first-0.5 s demand (-a/2) of the B4 straight-path onset plan, linearly interpolated in speed between the four seed speeds (-inf where no onset was found, i.e. at 5 m/s; below 10.4 m/s the 5 m/s value applies).
  `controller_safe_literal` compares the same quantity against -a_onset itself (the reading that ignores the factor 2). `controller_safe_bounded` (supplementary, added after seeing the menus) additionally requires first-0.5 s demand <= +6 m/s2 (the controller's acceleration bound), because the one-sided rule admits proposals demanding +100 m/s2 and more.
- **straight hold-speed** (for bd07fb91) = |y| < 0.5 m at 2 s and < 1.5 m at 4 s in the car frame AND mean 4 s speed >= 0.85 x car speed.
- scorer rank = rank of `S_plain` (argmax equals the logged pick-without-momentum on all 36 replans); `rescore changed` = the adapter's post-scorer momentum re-score picked a different proposal.

nC/rC = number / best scorer rank of clean human-like; nS/rS controller-safe; nSl/rSl literal; nSb/rSb bounded; nH/rH straight hold-speed; ch = chosen plan.

| ver | scene | t | car v | human v 4s | nC | rC | nS | rS | nSl | rSl | nSb | rSb | nH | rH | ch rank | rescore changed | ch mean v | ch a1 | ch spread | ch lat from human 2s | ch lat car frame 2s | ch hdg-tan max | ch score | menu max mean v | menu min spread | safe thr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1.6 | 04749bb9 | 0 | 23.56 | 23.74 | 0 |  | 15 | 5 | 24 | 3 | 13 | 5 | 25 | 3 | 1 | False | 18.86 | -6.5 | 17.31 | -0.19 | -0.17 | 0.81 | 0.86 | 67.82 | 3.56 | -3 |
| 0.1.6 | 04749bb9 | 0.5 | 22.19 | 23.76 | 0 |  | 1 | 29 | 3 | 20 | 1 | 29 | 0 |  | 1 | False | 12.83 | -12.53 | 11.95 | 5.92 | -8.52 | 23.67 | 0.18 | 25.9 | 8.89 | -2.66 |
| 0.1.6 | 04749bb9 | 1 | 18.47 | 23.73 | 0 |  | 9 | 1 | 28 | 1 | 9 | 1 | 0 |  | 1 | False | 22.07 | -1.82 | 13.79 | 2.04 | 8.33 | 25.25 | 0.46 | 28.21 | 3.28 | -3.14 |
| 0.1.6 | 04749bb9 | 1.5 | 18.65 | 23.7 | 0 |  | 26 | 1 | 32 | 1 | 25 | 1 | 0 |  | 1 | False | 18.09 | -1.78 | 4.16 | 2.37 | 1.45 | 1.15 | 0.73 | 27.4 | 2.83 | -3.12 |
| 0.1.6 | 04749bb9 | 2 | 20.01 | 23.6 | 0 |  | 6 | 1 | 17 | 1 | 6 | 1 | 0 |  | 1 | False | 21.07 | -0.93 | 25.47 | -4.35 | 12.98 | 12.48 | 0.37 | 28.02 | 6.62 | -2.94 |
| 0.1.6 | 04749bb9 | 2.5 | 21.24 | 23.51 | 0 |  | 20 | 3 | 28 | 1 | 18 | 3 | 0 |  | 1 | False | 18.96 | -5.08 | 11.76 | -3.47 | 6.15 | 13.8 | 0.41 | 61.42 | 4.52 | -2.78 |
| 0.1.6 | 04749bb9 | 3 | 21.91 | 23.39 | 0 |  | 0 |  | 0 |  | 0 |  | 0 |  | 1 | False | 8.85 | -14.15 | 9.06 | -13.58 | 2.09 | 13.35 | 0.22 | 23.21 | 6.35 | -2.7 |
| 0.1.6 | acd22430 | 0 | 23.43 | 23.33 | 1 | 9 | 12 | 8 | 21 | 3 | 10 | 8 | 24 | 1 | 1 | False | 20.37 | -5.45 | 9.02 | -0.21 | 0.16 | 1.39 | 0.9 | 67.13 | 2.81 | -2.5 |
| 0.1.6 | acd22430 | 0.5 | 22.5 | 23.33 | 0 |  | 1 | 17 | 1 | 17 | 0 |  | 0 |  | 1 | False | 15.14 | -12.24 | 17.82 | -17.2 | -2.41 | 124.34 | 0.12 | 62.85 | 7.56 | -2.62 |
| 0.1.6 | acd22430 | 1 | 19.38 | 23.33 | 0 |  | 17 | 1 | 29 | 1 | 17 | 1 | 0 |  | 1 | False | 20.6 | -1.96 | 4.14 | -14.15 | 4.62 | 16.38 | 0.13 | 26.83 | 3.77 | -3.02 |
| 0.1.6 | acd22430 | 1.5 | 18.05 | 23.33 | 0 |  | 25 | 3 | 31 | 2 | 25 | 3 | 0 |  | 1 | False | 12.21 | -8.08 | 3.97 | -12.86 | 2.19 | 8.39 | 0.34 | 27.38 | 3.11 | -3.2 |
| 0.1.6 | acd22430 | 2 | 16.3 | 23.32 | 0 |  | 26 | 4 | 31 | 2 | 26 | 4 | 0 |  | 1 | False | 9.8 | -7.2 | 8.27 | -7.74 | -3.52 | 22.89 | 0.23 | 27.04 | 2.83 | -3.42 |
| 0.1.6 | acd22430 | 2.5 | 12.47 | 23.28 | 0 |  | 29 | 3 | 32 | 1 | 29 | 3 | 0 |  | 1 | False | 6.62 | -4.04 | 6.79 | -11.61 | -0.46 | 42.7 | 0.21 | 15.46 | 2.81 | -3.5 |
| 0.1.6 | acd22430 | 3 | 9.09 | 23.23 | 0 |  | 32 | 1 | 32 | 1 | 32 | 1 | 0 |  | 1 | False | 4.76 | -2.55 | 6.36 | -9.92 | 0.64 | 5.99 | 0.6 | 14.83 | 0.58 | -inf |
| 0.1.4b | 04749bb9 | 0 | 23.56 | 23.74 | 0 |  | 2 | 29 | 2 | 29 | 1 | 32 | 1 | 29 | 1 | False | 13.04 | -14.32 | 5.75 | -0.07 | -0.04 | 0.69 | 0.9 | 28.05 | 2.18 | -3 |
| 0.1.4b | 04749bb9 | 0.5 | 20.41 | 23.76 | 0 |  | 0 |  | 0 |  | 0 |  | 0 |  | 1 | False | 3.2 | -31.28 | 3.7 | -7.01 | -1.19 | 44.66 | 0.57 | 8.75 | 2.74 | -2.89 |
| 0.1.4b | 04749bb9 | 1 | 15.92 | 23.73 | 0 |  | 0 |  | 0 |  | 0 |  | 0 |  | 1 | False | 3.11 | -22.61 | 4.52 | -14.04 | -1.51 | 67.62 | 0.45 | 8.6 | 4.21 | -3.47 |
| 0.1.4b | 04749bb9 | 1.5 | 11.33 | 23.7 | 0 |  | 0 |  | 21 | 4 | 0 |  | 0 |  | 1 | False | 1.95 | -15.92 | 2.63 | -17.95 | -0.5 | 134.68 | 0.48 | 8.26 | 2.24 | -3.5 |
| 0.1.4b | 04749bb9 | 2 | 6.81 | 23.6 | 0 |  | 32 | 1 | 32 | 1 | 32 | 1 | 0 |  | 1 | False | 2.61 | -4.23 | 3.48 | -21.46 | -2.06 | 16.46 | 0.73 | 6.73 | 0.5 | -inf |
| 0.1.4b | 04749bb9 | 2.5 | 3.62 | 23.51 | 0 |  | 32 | 1 | 32 | 1 | 32 | 1 | 0 |  | 1 | False | 0.52 | -2.83 | 2.16 | -20.35 | -0.19 | 170.45 | 0.71 | 5.24 | 0.19 | -inf |
| 0.1.4b | 04749bb9 | 3 | 1.56 | 23.39 | 0 |  | 32 | 1 | 32 | 1 | 32 | 1 | 4 | 23 | 1 | False | 0.12 | -2.39 | 0.32 | -20.48 | -0.05 |  | 0.77 | 3.02 | 0.21 | -inf |
| 0.1.4b | acd22430 | 0 | 23.43 | 23.33 | 0 |  | 2 | 30 | 3 | 16 | 1 | 32 | 1 | 30 | 1 | False | 13 | -17.3 | 3.03 | -0.04 | 0.16 | 1.24 | 0.96 | 29.26 | 2 | -2.5 |
| 0.1.4b | acd22430 | 0.5 | 20.39 | 23.33 | 0 |  | 0 |  | 0 |  | 0 |  | 0 |  | 1 | False | 2.96 | -28.76 | 5.02 | -8.57 | -2.22 | 81.89 | 0.18 | 8.41 | 3.47 | -2.89 |
| 0.1.4b | acd22430 | 1 | 15.9 | 23.33 | 0 |  | 0 |  | 0 |  | 0 |  | 0 |  | 1 | False | 2.49 | -22.66 | 3.85 | -14.85 | -1.65 | 79.07 | 0.33 | 8.96 | 3.44 | -3.48 |
| 0.1.4b | acd22430 | 1.5 | 11.32 | 23.33 | 0 |  | 0 |  | 17 | 4 | 0 |  | 0 |  | 1 | False | 1.99 | -15.48 | 2.6 | -18.44 | -0.43 | 107.78 | 0.48 | 7.88 | 2.19 | -3.5 |
| 0.1.4b | acd22430 | 2 | 6.79 | 23.32 | 0 |  | 32 | 1 | 32 | 1 | 32 | 1 | 0 |  | 1 | False | 2.08 | -3.95 | 4.63 | -22.22 | -1.42 | 115.21 | 0.67 | 6.53 | 0.64 | -inf |
| 0.1.4b | acd22430 | 2.5 | 3.56 | 23.28 | 0 |  | 32 | 1 | 32 | 1 | 32 | 1 | 0 |  | 1 | False | 0.48 | -2.91 | 2.04 | -20.67 | -0.14 | 7.37 | 0.78 | 4.86 | 0.18 | -inf |
| 0.1.4b | acd22430 | 3 | 1.38 | 23.23 | 0 |  | 32 | 1 | 32 | 1 | 32 | 1 | 3 | 28 | 1 | False | 0.07 | -2.27 | 0.24 | -20.9 | -0.04 |  | 0.77 | 2.26 | 0.1 | -inf |
| 0.1.6 | bd07fb91 | 5 | 14.6 | 9.94 | 0 |  | 28 | 1 | 30 | 1 | 28 | 1 | 4 | 2 | 3 | True | 13.26 | 0.35 | 3.47 | 2.57 | -0.32 | 5.93 | 0.8 | 19.11 | 3.2 | -3.5 |
| 0.1.6 | bd07fb91 | 5.5 | 14.57 | 9.03 | 0 |  | 29 | 1 | 31 | 1 | 29 | 1 | 2 | 1 | 1 | False | 13.89 | -0.21 | 2.35 | 2.92 | -0.45 | 7.73 | 0.64 | 20.15 | 0.75 | -3.5 |
| 0.1.6 | bd07fb91 | 6 | 14.3 | 7.98 | 0 |  | 29 | 3 | 31 | 2 | 29 | 3 | 0 |  | 1 | False | 6.71 | -9.09 | 7.46 | 0.55 | -2.15 | 20.97 | 0.36 | 18.38 | 2.35 | -3.5 |
| 0.1.6 | bd07fb91 | 6.5 | 11.5 | 6.81 | 0 |  | 30 | 2 | 32 | 1 | 30 | 2 | 0 |  | 1 | False | 4.66 | -6.8 | 5.85 | 4.86 | -1.1 | 5.76 | 0.29 | 15.61 | 1.11 | -3.5 |
| 0.1.4b | bd07fb91 | 5 | 6.86 | 9.94 | 0 |  | 32 | 1 | 32 | 1 | 32 | 1 | 0 |  | 1 | False | 0.36 | -11.75 | 0.93 | -3.33 | -0.03 | 179.29 | 0.51 | 5.92 | 0.93 | -inf |
| 0.1.4b | bd07fb91 | 5.5 | 2.69 | 9.03 | 0 |  | 32 | 1 | 32 | 1 | 32 | 1 | 0 |  | 1 | False | 0.16 | -4.19 | 0.57 | -3.47 | -0.07 | 18.02 | 0.58 | 1.29 | 0.57 | -inf |
| 0.1.4b | bd07fb91 | 6 | 0.14 | 7.98 | 0 |  | 32 | 1 | 32 | 1 | 30 | 1 | 26 | 5 | 1 | False | 0.04 | 0.03 | 0.15 | -3.53 | -0.04 |  | 0.62 | 3.39 | 0.15 | -inf |
| 0.1.4b | bd07fb91 | 6.5 | 0.01 | 6.81 | 0 |  | 32 | 1 | 32 | 1 | 32 | 1 | 22 | 1 | 1 | False | 0.02 | 0.14 | 0.08 | -3.51 | -0.03 |  | 0.69 | 2.9 | 0.07 | -inf |

Counts over the 7 replans +0.0 ... +3.0 s:

| version | scene | replans | replans_with_clean | replans_with_safe | replans_with_safe_bounded | replans_chosen_safe | replans_rescore_changed |
|---|---|---|---|---|---|---|---|
| 0.1.4b | 04749bb9 | 7 | 0 | 4 | 4 | 3 | 0 |
| 0.1.4b | acd22430 | 7 | 0 | 4 | 4 | 3 | 0 |
| 0.1.6 | 04749bb9 | 7 | 0 | 6 | 6 | 3 | 0 |
| 0.1.6 | acd22430 | 7 | 1 | 7 | 6 | 2 | 0 |

Proposals with implausible kinematics among all tabulated proposals:

| version | proposals | mean_speed_gt_40mps | first_0p5s_accel_gt_plus20 |
|---|---|---|---|
| 0.1.4b | 576 | 0 | 2 |
| 0.1.6 | 576 | 10 | 6 |

### Chosen / best clean / best safe proposals through the controller (each version's own run state at the tick; 1.0 s hold)

| ver | scene | tick | which | proposal | rank | score | mean v 4s | a1 | a@2s | spread | lat from human 2s | peak steer (rad) | first>0.3 (s) | lat disp (m) | u_long min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1.6 | 04749bb9 | 0 | chosen | 13 | 1 | 0.864 | 18.855 | -6.498 | -4.051 | 17.307 | -0.193 | 0.525 | 0.1 | 3.805 | -5.158 |
| 0.1.6 | 04749bb9 | 0 | best_clean_human_like | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.6 | 04749bb9 | 0 | best_controller_safe | 18 | 5 | 0.857 | 21.092 | -1.746 | -0.663 | 14.041 | -0.178 | 0.019 |  | 0.178 | -1.321 |
| 0.1.6 | 04749bb9 | 0 | best_controller_safe_literal | 26 | 3 | 0.863 | 20.233 | -4.722 | -2.79 | 6.17 | -0.175 | 0.392 | 0.15 | 3.771 | -3.286 |
| 0.1.6 | 04749bb9 | 0 | best_straight_hold_speed | 26 | 3 | 0.863 | 20.233 | -4.722 | -2.79 | 6.17 | -0.175 | 0.392 | 0.15 | 3.771 | -3.286 |
| 0.1.6 | 04749bb9 | 0 | best_controller_safe_bounded | 18 | 5 | 0.857 | 21.092 | -1.746 | -0.663 | 14.041 | -0.178 | 0.019 |  | 0.178 | -1.321 |
| 0.1.6 | 04749bb9 | 0.5 | chosen | 24 | 1 | 0.176 | 12.828 | -12.525 | -7.791 | 11.949 | 5.918 | 0.685 | 0.05 | 5.829 | -9 |
| 0.1.6 | 04749bb9 | 0.5 | best_clean_human_like | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.6 | 04749bb9 | 0.5 | best_controller_safe | 4 | 29 | 0.115 | 18.938 | -1.361 | 0.067 | 17.376 | 3.357 | 0.354 | 0.05 | 5.149 | -3.796 |
| 0.1.6 | 04749bb9 | 0.5 | best_controller_safe_literal | 29 | 20 | 0.131 | 18.84 | -4.81 | 0.383 | 18.904 | 9.93 | 0.38 | 0.05 | 4.633 | -2.622 |
| 0.1.6 | 04749bb9 | 0.5 | best_straight_hold_speed | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.6 | 04749bb9 | 0.5 | best_controller_safe_bounded | 4 | 29 | 0.115 | 18.938 | -1.361 | 0.067 | 17.376 | 3.357 | 0.354 | 0.05 | 5.149 | -3.796 |
| 0.1.6 | acd22430 | 0 | chosen | 26 | 1 | 0.902 | 20.37 | -5.45 | -2.812 | 9.021 | -0.211 | 0.401 | 0.15 | 3.767 | -3.475 |
| 0.1.6 | acd22430 | 0 | best_clean_human_like | 23 | 9 | 0.883 | 20.579 | -5.42 | -2.468 | 2.813 | -0.179 | 0.48 | 0.1 | 3.898 | -3.099 |
| 0.1.6 | acd22430 | 0 | best_controller_safe | 18 | 8 | 0.883 | 21.923 | -1.435 | -0.717 | 4.912 | -0.242 | 0.015 |  | 0.108 | -1.633 |
| 0.1.6 | acd22430 | 0 | best_controller_safe_literal | 28 | 3 | 0.896 | 20.469 | -4.408 | -1.394 | 11.886 | -0.226 | 0.289 |  | 3.346 | -1.949 |
| 0.1.6 | acd22430 | 0 | best_straight_hold_speed | 26 | 1 | 0.902 | 20.37 | -5.45 | -2.812 | 9.021 | -0.211 | 0.401 | 0.15 | 3.767 | -3.475 |
| 0.1.6 | acd22430 | 0 | best_controller_safe_bounded | 18 | 8 | 0.883 | 21.923 | -1.435 | -0.717 | 4.912 | -0.242 | 0.015 |  | 0.108 | -1.633 |
| 0.1.6 | acd22430 | 0.5 | chosen | 13 | 1 | 0.121 | 15.137 | -12.238 | -6.881 | 17.822 | -17.2 | 0.612 | 0.05 | 2.45 | -8.612 |
| 0.1.6 | acd22430 | 0.5 | best_clean_human_like | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.6 | acd22430 | 0.5 | best_controller_safe | 4 | 17 | 0.05 | 62.845 | 279.766 | 62.145 | 145.881 | -29.815 | 0.367 | 1 | 1.024 | -2.638 |
| 0.1.6 | acd22430 | 0.5 | best_controller_safe_literal | 4 | 17 | 0.05 | 62.845 | 279.766 | 62.145 | 145.881 | -29.815 | 0.367 | 1 | 1.024 | -2.638 |
| 0.1.6 | acd22430 | 0.5 | best_straight_hold_speed | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.6 | acd22430 | 0.5 | best_controller_safe_bounded | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | 04749bb9 | 0 | chosen | 30 | 1 | 0.899 | 13.039 | -14.318 | -9.886 | 5.754 | -0.071 | 0.776 | 0.05 | 4.378 | -9 |
| 0.1.4b | 04749bb9 | 0 | best_clean_human_like | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | 04749bb9 | 0 | best_controller_safe | 14 | 29 | 0.862 | 28.053 | 129.378 | 20.332 | 79.473 | -0.069 | 0.769 | 0.05 | 4.367 | -9 |
| 0.1.4b | 04749bb9 | 0 | best_controller_safe_literal | 14 | 29 | 0.862 | 28.053 | 129.378 | 20.332 | 79.473 | -0.069 | 0.769 | 0.05 | 4.367 | -9 |
| 0.1.4b | 04749bb9 | 0 | best_straight_hold_speed | 14 | 29 | 0.862 | 28.053 | 129.378 | 20.332 | 79.473 | -0.069 | 0.769 | 0.05 | 4.367 | -9 |
| 0.1.4b | 04749bb9 | 0 | best_controller_safe_bounded | 4 | 32 | 0.817 | 14.578 | -2.65 | -7.056 | 17.215 | -0.09 | 0.658 | 0.05 | 3.929 | -9 |
| 0.1.4b | 04749bb9 | 0.5 | chosen | 11 | 1 | 0.57 | 3.197 | -31.284 | -17.009 | 3.697 | -7.007 | 0.785 | 0.1 | 7.759 | -9 |
| 0.1.4b | 04749bb9 | 0.5 | best_clean_human_like | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | 04749bb9 | 0.5 | best_controller_safe | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | 04749bb9 | 0.5 | best_controller_safe_literal | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | 04749bb9 | 0.5 | best_straight_hold_speed | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | 04749bb9 | 0.5 | best_controller_safe_bounded | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | acd22430 | 0 | chosen | 25 | 1 | 0.962 | 13.004 | -17.301 | -9.805 | 3.026 | -0.038 | 0.771 | 0.05 | 4.344 | -9 |
| 0.1.4b | acd22430 | 0 | best_clean_human_like | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | acd22430 | 0 | best_controller_safe | 14 | 30 | 0.918 | 29.258 | 136.913 | 22.247 | 83.146 | -0.049 | 0.769 | 0.05 | 4.295 | -9 |
| 0.1.4b | acd22430 | 0 | best_controller_safe_literal | 0 | 16 | 0.944 | 12.194 | -4.998 | -10.208 | 10.946 | -0.026 | 0.776 | 0.05 | 4.471 | -9 |
| 0.1.4b | acd22430 | 0 | best_straight_hold_speed | 14 | 30 | 0.918 | 29.258 | 136.913 | 22.247 | 83.146 | -0.049 | 0.769 | 0.05 | 4.295 | -9 |
| 0.1.4b | acd22430 | 0 | best_controller_safe_bounded | 4 | 32 | 0.856 | 14.572 | -1.244 | -6.868 | 18.303 | -0.042 | 0.669 | 0.05 | 3.854 | -9 |
| 0.1.4b | acd22430 | 0.5 | chosen | 11 | 1 | 0.179 | 2.959 | -28.759 | -16.296 | 5.019 | -8.572 | 0.785 | 0.1 | 7.743 | -9 |
| 0.1.4b | acd22430 | 0.5 | best_clean_human_like | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | acd22430 | 0.5 | best_controller_safe | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | acd22430 | 0.5 | best_controller_safe_literal | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | acd22430 | 0.5 | best_straight_hold_speed | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.4b | acd22430 | 0.5 | best_controller_safe_bounded | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.6 | bd07fb91 | 6 | chosen | 11 | 1 | 0.362 | 6.714 | -9.092 | -6.025 | 7.463 | 0.546 | 0.783 | 0.1 | 1.282 | -9 |
| 0.1.6 | bd07fb91 | 6 | best_clean_human_like | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.6 | bd07fb91 | 6 | best_controller_safe | 15 | 3 | 0.322 | 9.102 | -2.662 | -3.521 | 7.01 | 0.183 | 0.189 |  | 0.249 | -5.139 |
| 0.1.6 | bd07fb91 | 6 | best_controller_safe_literal | 24 | 2 | 0.332 | 7.77 | -3.624 | -4.468 | 10.238 | 0.157 | 0.255 |  | 0.426 | -6.448 |
| 0.1.6 | bd07fb91 | 6 | best_straight_hold_speed | none on menu |  |  |  |  |  |  |  |  |  |  |  |
| 0.1.6 | bd07fb91 | 6 | best_controller_safe_bounded | 15 | 3 | 0.322 | 9.102 | -2.662 | -3.521 | 7.01 | 0.183 | 0.189 |  | 0.249 | -5.139 |

### bd07fb91, replan at +6.0 s (0.1.6)

- Car 14.30 m/s; human mean speed over the next 4 s 7.98 m/s. At this tick the human's recorded position is 11.49 m behind and 2.44 m to the right of the car (heading difference -0.02 rad); the human's path over the next 4 s stays 2.5-3.0 m to the right of the car's axis (plot 8).
- Chosen proposal 11 (scorer rank 1, score 0.362; re-score did not change the pick): y = -2.15 m at 2 s and -6.65 m at 4 s in the car frame, first-0.5 s demand -9.09 m/s2, mean 4 s speed 6.71 m/s, step spread 7.46 m/s; its 2 s waypoint is +0.55 m from the human path, its 4 s waypoint -2.42 m.
- All 32 proposals go right: y at 2 s ranges -7.05 ... -1.55 m. **Straight hold-speed proposals on this menu: 0.** Least-lateral proposal: #19 (y -1.55 m at 2 s, -3.42 m at 4 s, mean speed 13.34 m/s, rank 29, score 0.072). Fastest: #4 (mean 18.38 m/s, y -5.56 m at 2 s, rank 30, score 0.043).
  At +5.0 s / +5.5 s the menu has 4 / 2 straight hold-speed proposals (best ranks in the table above) and the chosen plans there demand +0.35 / -0.21 m/s2.
- Through the controller from the +6.0 s state: chosen 0.783 rad; best controller-safe (rank 3) 0.189 rad.
- Front camera at +6.0 s (`bd07fb91_front_camera_plus6p0s.png`, logged `camera_front_wide_120fov` frame ending at handover + 6.00 s): a signalised intersection with every visible signal head red; a dark sedan directly ahead in the car's lane and two more vehicles in the lane(s) to the left, all showing brake lights; the lane to the right of the car is empty up to the intersection.
- 0.1.4b on the same clip is at 0.14 m/s at +6.0 s (a different state; its rows are in the tables for completeness).

Top 8 of the +6.0 s menu by score:

| proposal | scorer_rank | scorer_score | mean_speed_4s_mps | first_0p5s_accel_mps2 | step_speed_spread_mps | lat_rig_2s_m | lat_rig_4s_m | lat_from_human_path_2s_m | lat_from_human_path_4s_m | chosen |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 1 | 0.36 | 6.71 | -9.09 | 7.46 | -2.15 | -6.65 | 0.55 | -2.42 | True |
| 24 | 2 | 0.33 | 7.77 | -3.62 | 10.24 | -2.73 | -8.95 | 0.16 | -2.95 | False |
| 15 | 3 | 0.32 | 9.1 | -2.66 | 7.01 | -2.92 | -11.26 | 0.18 | -2.73 | False |
| 3 | 4 | 0.28 | 10.42 | -1.66 | 5.9 | -4 | -17.98 | -0.63 | -8.33 | False |
| 9 | 5 | 0.28 | 11.9 | -0.87 | 4.42 | -4.16 | -20.6 | -0.21 | -8.09 | False |
| 13 | 6 | 0.28 | 9.7 | -2.08 | 6.88 | -2.95 | -10.56 | 0.41 | -0.73 | False |
| 23 | 7 | 0.28 | 10.81 | -1.62 | 7.04 | -3.93 | -17.63 | -0.23 | -6.89 | False |
| 28 | 8 | 0.26 | 11.58 | -1.11 | 5.41 | -3.55 | -17.47 | 0.47 | -5.12 | False |

## Predictions vs measured values (no verdicts)


- P1 (first |steer| > 0.3 rad within 0.2 s of handover in the standalone replay): 04749bb9 replay 0.10 s (recorded 0.10 s); acd22430 replay 0.10 s fresh-seeded / 0.15 s replayed-from-start (recorded 0.15 s).
- P2 (no braking cuts peak |steer| in 1 s by > 90 %): B1 at handover 0.525 -> 0.003 rad (99.5 %) on 04749bb9; 0.401 -> 0.002 rad (99.6 %) on acd22430. At the +0.5 s tick: 41.3 % and 66.5 %.
- P3 (heading-only change cuts < 30 %): B2 0.0 % / -0.0 %; B2z 0.0 % / -0.0 % (handover); at +0.5 s within +-2 %.
- P4 (same path, same 4 s end point, constant deceleration < 3 m/s2 steers < 0.1 rad): B4e (a = 2.35 / 1.53 m/s2) steers 0.005 / 0.001 rad; sweep at 23.5 m/s stays <= 0.013 rad up to 4 m/s2, 0.055-0.124 rad at 5, 0.24-0.26 at 6. B5 (only the first step re-spaced): 0.151 rad on 04749bb9, 0.004 rad on acd22430.
- P5 (clean human-like on the menu on at most 3 of 7 replans): 0.1.6 04749bb9 0 of 7, acd22430 1 of 7; 0.1.4b 0 of 7 on both.
- P6 (scorer ranks it outside top 5): the only clean human-like proposal (0.1.6 acd22430, +0.0 s, proposal 23) has scorer rank 9 (score 0.883 vs 0.902 for the pick); held through the controller it steers 0.480 rad (first-0.5 s demand -5.42 m/s2).
- P7 (bd07fb91 +6.0 s): see the section above - chosen plan y = -2.15 m at 2 s, -9.09 m/s2; straight hold-speed proposals on that menu: 0 of 32.

## Files

CSVs: `gate.csv`, `counterfactuals.csv`, `braking_sweep.csv`, `braking_onset.csv`, `menu_proposals.csv`, `menu_summary.csv`, `menu_through_controller.csv`, `plan_anatomy.csv`, `replan_jumps.csv`, `plan_relation.csv`.
Plots: `plot1_gate.png`, `plot2_counterfactuals_04749bb9.png`, `plot3_counterfactuals_acd22430.png`, `plot4_braking_sweep.png`, `plot5_plan_anatomy.png`, `plot6_menu_handover_016.png`, `plot7_menu_handover_014b.png`, `plot8_bd07fb91_menu_topview.png`; camera frame `bd07fb91_front_camera_plus6p0s.png`.
Series for re-plotting: `gate_series.pkl`, `run_series.pkl`; `cache/` = per-run cache of the controller requests read from the rollout logs.
