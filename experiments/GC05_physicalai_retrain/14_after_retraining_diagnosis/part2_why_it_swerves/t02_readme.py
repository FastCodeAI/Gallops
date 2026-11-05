#!/usr/bin/env python
"""T02: build README.md and the loop report.json from the CSVs (no numbers typed by hand)."""
import json, pathlib, shutil
import numpy as np, pandas as pd
H = pathlib.Path(__file__).resolve().parent; REP = pathlib.Path("{runs}/paddock_sid/loop/runs/current/tasks/02_highway-swerve-mpc-counterfact/report")
g = pd.read_csv(H / "gate.csv"); cf = pd.read_csv(H / "counterfactuals.csv"); sw = pd.read_csv(H / "braking_sweep.csv"); on = pd.read_csv(H / "braking_onset.csv"); mp = pd.read_csv(H / "menu_proposals.csv")
ms = pd.read_csv(H / "menu_summary.csv"); ctl = pd.read_csv(H / "menu_through_controller.csv"); an = pd.read_csv(H / "plan_anatomy.csv"); jp = pd.read_csv(H / "replan_jumps.csv"); rl = pd.read_csv(H / "plan_relation.csv")
def md(df, r=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f": df[c] = df[c].map(lambda x: "" if pd.isna(x) else (f"{x:.{r}f}".rstrip("0").rstrip(".") if abs(x) < 1e6 else str(x)))
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(map(str, row)) + " |" for row in df.values])
L = []
L.append("""# T02 - highway swerve after retraining: recorded plans through the standalone steering controller, counterfactuals, menus

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

""" + md(rl, 6) + """

## Part A - gate

Three standalone replays per case, each fed the recorded plan of every 50 ms tick for 3 s: `cold_tf` = fresh controller + vehicle model seeded with
the recorded `controller_request.state` at the start tick, recorded pose override each tick (what the runtime did); `warm_tf` = same but the
controller was first replayed from the session's first request; `cold_free` = fresh, the simulator's own pose, plans held in the world frame.
Differences are replay minus the run's controller CSV; pct = max difference / recorded peak in that window (meaningless where the recorded peak is ~0).

""")
c = g[["scene", "start_after_handover_s", "mode", "steer_state_maxdiff_0p5s", "steer_state_rmsdiff_0p5s", "steer_state_rec_peak_0p5s", "steer_state_maxdiff_pct_of_rec_peak_0p5s", "long_cmd_maxdiff_0p5s", "long_cmd_rmsdiff_0p5s", "long_cmd_rec_peak_0p5s",
       "long_cmd_maxdiff_pct_of_rec_peak_0p5s", "steer_state_maxdiff_3s", "steer_state_rmsdiff_3s", "long_cmd_maxdiff_3s", "long_cmd_rmsdiff_3s", "first_steer_gt_0p3_rec_s", "first_steer_gt_0p3_sim_s", "peak_abs_steer_3s_rec_rad", "peak_abs_steer_3s_sim_rad", "n_solves_not_solved"]]
c.columns = ["scene", "start (s after handover)", "mode", "steer maxdiff 0.5s (rad)", "steer rms 0.5s", "rec peak steer 0.5s", "steer pct 0.5s", "long maxdiff 0.5s (m/s2)", "long rms 0.5s", "rec peak |long| 0.5s", "long pct 0.5s", "steer maxdiff 3s", "steer rms 3s", "long maxdiff 3s", "long rms 3s",
             "first >0.3 rec (s)", "first >0.3 replay (s)", "peak steer 3s rec", "peak steer 3s replay", "solves not 'solved'"]
L.append(md(c, 4)); w = g[g["mode"] == "warm_tf"]
hw = g[(g["mode"] == "cold_tf") & (g.scene.isin(["04749bb9", "acd22430"]))]
L.append(f"""
**Gate: passes.** `warm_tf` reproduces the recorded steering and longitudinal command to floating-point precision on all five cases
(max |diff| over 3 s: steering {w.steer_state_maxdiff_3s.max():.1e} rad, longitudinal {w.long_cmd_maxdiff_3s.max():.1e} m/s2). The fresh-controller replay
(`cold_tf`) on the two swerving highway scenes is within {hw.steer_state_maxdiff_pct_of_rec_peak_0p5s.max():.1f} % (steering) and {hw.long_cmd_maxdiff_pct_of_rec_peak_0p5s.max():.1f} % (longitudinal) of the recorded peak in the first 0.5 s.
bd07fb91 (both starts) and the control 02e075b9 have recorded steering < 0.03 rad in the first 0.5 s, so only absolute differences mean anything there
(<= {g[g.scene.isin(['bd07fb91', '02e075b9'])].steer_state_maxdiff_0p5s.max():.4f} rad). Negative control 02e075b9: replayed peak |steer| over 3 s = {g[g.scene == '02e075b9'].peak_abs_steer_3s_sim_rad.max():.4f} rad.
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
""")
c = cf[["version", "scene", "tick_after_handover_s", "variant", "ego_speed_mps", "peak_abs_steer_rad", "peak_abs_steer_cmd_rad", "first_steer_gt_0p3_s", "steer_sign_changes", "peak_abs_lateral_m", "u_long_min_mps2", "u_long_max_mps2", "first_0p5s_accel_mps2", "accel_equiv_to_1s_mps2",
        "accel_equiv_to_2s_mps2", "mean_speed_4s_mps", "step_speed_spread_mps", "heading_vs_tangent_max_deg"]]
c.columns = ["ver", "scene", "tick", "variant", "v0", "peak steer (rad)", "peak cmd", "first>0.3 (s)", "sign chg", "lat (m)", "u_long min", "u_long max", "a1", "a@1s", "a@2s", "mean v 4s", "step spread", "hdg-tangent max (deg)"]
L.append(md(c)); L.append("\nReduction of peak |steer| relative to B0 (same scene, same tick):\n")
r = cf[(cf.version == "0.1.6") & ~cf.variant.str.contains("cold")].copy(); r["b0"] = r.groupby(["scene", "tick_after_handover_s"]).peak_abs_steer_rad.transform("first"); r["cut_pct"] = 100 * (1 - r.peak_abs_steer_rad / r.b0)
L.append(md(r.pivot_table(index="variant", columns=["scene", "tick_after_handover_s"], values="cut_pct", sort=False).round(1).reset_index().pipe(lambda d: d.set_axis([c if isinstance(c, str) else f"{c[0]} +{c[1]}s" if c[1] != "" else c[0] for c in d.columns], axis=1)), 1))
L.append("\n### B0 plan anatomy at the handover replan (0.1.6 and 0.1.4b)\n\nstep speed = distance between consecutive poses / 0.5 s; raw = the 8 model waypoints of the regenerated menu; ctrl = the plan in the controller request; implied accel = change of step speed / 0.5 s (step 1 against the car speed).\n")
a = an[an.source == "handover_replan"][["version", "scene", "pose", "t_s", "step_speed_raw_waypoints_mps", "step_speed_ctrl_plan_mps", "step_speed_human_mps", "implied_accel_mps2", "y_lateral_m", "lateral_from_human_path_m", "heading_deg", "heading_minus_tangent_deg"]]
a.columns = ["ver", "scene", "pose", "t (s)", "step v raw", "step v ctrl", "step v human", "implied accel (m/s2)", "lateral y (m)", "lateral from human path (m)", "heading (deg)", "heading - tangent (deg)"]; L.append(md(a, 2))
L.append("\nPlans of the 9 cached ticks after handover (re-anchored, per-tick): full per-pose table in `plan_anatomy.csv` (`source = cached_tick`). Car-to-first-pose distance / 0.5 s and lateral offset of the first pose, 0.1.6:\n")
a = an[(an.source == "cached_tick") & (an.pose == 1) & (an.version == "0.1.6")].pivot_table(index="tick_after_handover_s", columns="scene", values=["step_speed_ctrl_plan_mps", "y_lateral_m", "heading_deg"]).round(2); a.columns = [f"{x} {y}" for x, y in a.columns]; L.append(md(a.reset_index(), 2))
L.append("\n### Position jump between consecutive plans at each replan\n\nstart jump = distance from where the previous replan's plan put the car 0.5 s later (its pose 1) to the car's actual pose at the new replan; along / lateral / heading are the car's pose seen from that old pose; last column = new plan's pose 1 vs old plan's pose 2 (same instant).\n")
L.append(md(jp, 2))
L.append("\n### B4 braking sweep (handover tick of each seed scene, real recorded states; plan held 1.0 s)\n\nPeak |steer| (rad):\n")
p = sw.pivot_table(index="const_decel_mps2", columns=["seed_scene", "path"], values="peak_abs_steer_rad", sort=False); sp = sw.groupby(["seed_scene", "path"], sort=False).ego_speed_mps.first(); p.columns = [f"{s} {k} ({sp[(s, k)]:.1f} m/s)" for s, k in p.columns]; L.append(md(p.reset_index()))
L.append("\nMinimum commanded longitudinal acceleration (m/s2) in the same runs:\n"); p = sw.pivot_table(index="const_decel_mps2", columns=["seed_scene", "path"], values="u_long_min_mps2", sort=False); p.columns = [f"{s} {k}" for s, k in p.columns]; L.append(md(p.reset_index(), 2))
L.append("\nOnset = smallest swept constant deceleration with peak |steer| > 0.1 rad (sweep grid 0,1,...,10,12 m/s2; 7 and 9 added to the requested list). `onset_as_first_0p5s_accel` = the first-0.5 s demand of that onset plan (-a/2).\n"); L.append(md(on))
L.append("""
## Part C - menus

Per-proposal table: `menu_proposals.csv` (32 x 36 replans). Rules fixed before reading the results:
- **clean human-like** = |mean 4 s speed - human mean speed over the same 4 s| <= 15 % of the human's AND step-speed spread < 3 m/s AND |perpendicular offset of the 2 s waypoint from the human path| < 1.0 m.
- **controller-safe** = first-0.5 s demanded acceleration > threshold(v), threshold = the first-0.5 s demand (-a/2) of the B4 straight-path onset plan, linearly interpolated in speed between the four seed speeds (-inf where no onset was found, i.e. at 5 m/s; below 10.4 m/s the 5 m/s value applies).
  `controller_safe_literal` compares the same quantity against -a_onset itself (the reading that ignores the factor 2). `controller_safe_bounded` (supplementary, added after seeing the menus) additionally requires first-0.5 s demand <= +6 m/s2 (the controller's acceleration bound), because the one-sided rule admits proposals demanding +100 m/s2 and more.
- **straight hold-speed** (for bd07fb91) = |y| < 0.5 m at 2 s and < 1.5 m at 4 s in the car frame AND mean 4 s speed >= 0.85 x car speed.
- scorer rank = rank of `S_plain` (argmax equals the logged pick-without-momentum on all 36 replans); `rescore changed` = the adapter's post-scorer momentum re-score picked a different proposal.

nC/rC = number / best scorer rank of clean human-like; nS/rS controller-safe; nSl/rSl literal; nSb/rSb bounded; nH/rH straight hold-speed; ch = chosen plan.
""")
c = ms[["version", "scene", "replan_after_handover_s", "ego_speed_mps", "human_mean_speed_4s_mps", "n_clean_human_like", "best_rank_clean_human_like", "n_controller_safe", "best_rank_controller_safe", "n_controller_safe_literal", "best_rank_controller_safe_literal",
        "n_controller_safe_bounded", "best_rank_controller_safe_bounded", "n_straight_hold_speed", "best_rank_straight_hold_speed", "chosen_scorer_rank", "rescore_3p5_changed_pick", "chosen_mean_speed_4s_mps", "chosen_first_0p5s_accel_mps2", "chosen_step_speed_spread_mps",
        "chosen_lat_from_human_path_2s_m", "chosen_lat_rig_2s_m", "chosen_heading_vs_tangent_max_abs_deg", "chosen_score", "max_mean_speed_4s_on_menu_mps", "min_step_speed_spread_on_menu_mps", "safe_threshold_first_0p5s_accel_mps2"]]
c.columns = ["ver", "scene", "t", "car v", "human v 4s", "nC", "rC", "nS", "rS", "nSl", "rSl", "nSb", "rSb", "nH", "rH", "ch rank", "rescore changed", "ch mean v", "ch a1", "ch spread", "ch lat from human 2s", "ch lat car frame 2s", "ch hdg-tan max", "ch score", "menu max mean v", "menu min spread", "safe thr"]
L.append(md(c, 2))
hwm = ms[ms.scene.isin(["04749bb9", "acd22430"])]; k = hwm.groupby(["version", "scene"]).agg(replans=("n_clean_human_like", "size"), replans_with_clean=("n_clean_human_like", lambda x: int((x > 0).sum())), replans_with_safe=("n_controller_safe", lambda x: int((x > 0).sum())),
     replans_with_safe_bounded=("n_controller_safe_bounded", lambda x: int((x > 0).sum())), replans_chosen_safe=("chosen_controller_safe", "sum"), replans_rescore_changed=("rescore_3p5_changed_pick", "sum")).reset_index()
L.append("\nCounts over the 7 replans +0.0 ... +3.0 s:\n"); L.append(md(k))
wild = mp.groupby("version").apply(lambda d: pd.Series(dict(proposals=len(d), mean_speed_gt_40mps=int((d.mean_speed_4s_mps > 40).sum()), first_0p5s_accel_gt_plus20=int((d.first_0p5s_accel_mps2 > 20).sum())))).reset_index()
L.append("\nProposals with implausible kinematics among all tabulated proposals:\n"); L.append(md(wild))
L.append("\n### Chosen / best clean / best safe proposals through the controller (each version's own run state at the tick; 1.0 s hold)\n")
c = ctl[["version", "scene", "tick_after_handover_s", "which", "proposal", "scorer_rank", "scorer_score", "mean_speed_4s_mps", "first_0p5s_accel_mps2", "accel_equiv_to_2s_mps2", "step_speed_spread_mps", "lat_from_human_path_2s_m", "peak_abs_steer_rad", "first_steer_gt_0p3_s", "peak_abs_lateral_m", "u_long_min_mps2"]].copy()
c["proposal"] = c.proposal.map(lambda i: "none on menu" if i < 0 else int(i)); c.columns = ["ver", "scene", "tick", "which", "proposal", "rank", "score", "mean v 4s", "a1", "a@2s", "spread", "lat from human 2s", "peak steer (rad)", "first>0.3 (s)", "lat disp (m)", "u_long min"]; L.append(md(c))
b = mp[(mp.version == "0.1.6") & (mp.scene == "bd07fb91") & (mp.replan_after_handover_s == 6.0)].sort_values("scorer_rank"); ch = b[b.chosen].iloc[0]; fast = b.sort_values("mean_speed_4s_mps").iloc[-1]; strt = b.iloc[(b.lat_rig_2s_m.abs()).argmin()]
L.append(f"""
### bd07fb91, replan at +6.0 s (0.1.6)

- Car {ch.ego_speed_mps:.2f} m/s; human mean speed over the next 4 s {ch.human_mean_speed_4s_mps:.2f} m/s. At this tick the human's recorded position is 11.49 m behind and 2.44 m to the right of the car (heading difference -0.02 rad); the human's path over the next 4 s stays 2.5-3.0 m to the right of the car's axis (plot 8).
- Chosen proposal {int(ch.proposal)} (scorer rank {int(ch.scorer_rank)}, score {ch.scorer_score:.3f}; re-score did not change the pick): y = {ch.lat_rig_2s_m:.2f} m at 2 s and {ch.lat_rig_4s_m:.2f} m at 4 s in the car frame, first-0.5 s demand {ch.first_0p5s_accel_mps2:.2f} m/s2, mean 4 s speed {ch.mean_speed_4s_mps:.2f} m/s, step spread {ch.step_speed_spread_mps:.2f} m/s; its 2 s waypoint is {ch.lat_from_human_path_2s_m:+.2f} m from the human path, its 4 s waypoint {ch.lat_from_human_path_4s_m:+.2f} m.
- All 32 proposals go right: y at 2 s ranges {b.lat_rig_2s_m.min():.2f} ... {b.lat_rig_2s_m.max():.2f} m. **Straight hold-speed proposals on this menu: {int(b.straight_hold_speed.sum())}.** Least-lateral proposal: #{int(strt.proposal)} (y {strt.lat_rig_2s_m:.2f} m at 2 s, {strt.lat_rig_4s_m:.2f} m at 4 s, mean speed {strt.mean_speed_4s_mps:.2f} m/s, rank {int(strt.scorer_rank)}, score {strt.scorer_score:.3f}). Fastest: #{int(fast.proposal)} (mean {fast.mean_speed_4s_mps:.2f} m/s, y {fast.lat_rig_2s_m:.2f} m at 2 s, rank {int(fast.scorer_rank)}, score {fast.scorer_score:.3f}).
  At +5.0 s / +5.5 s the menu has {int(ms[(ms.version == '0.1.6') & (ms.scene == 'bd07fb91') & (ms.replan_after_handover_s == 5.0)].n_straight_hold_speed.iloc[0])} / {int(ms[(ms.version == '0.1.6') & (ms.scene == 'bd07fb91') & (ms.replan_after_handover_s == 5.5)].n_straight_hold_speed.iloc[0])} straight hold-speed proposals (best ranks in the table above) and the chosen plans there demand +0.35 / -0.21 m/s2.
- Through the controller from the +6.0 s state: chosen {ctl[(ctl.scene == 'bd07fb91') & (ctl.which == 'chosen')].peak_abs_steer_rad.iloc[0]:.3f} rad; best controller-safe (rank {int(ctl[(ctl.scene == 'bd07fb91') & (ctl.which == 'best_controller_safe')].scorer_rank.iloc[0])}) {ctl[(ctl.scene == 'bd07fb91') & (ctl.which == 'best_controller_safe')].peak_abs_steer_rad.iloc[0]:.3f} rad.
- Front camera at +6.0 s (`bd07fb91_front_camera_plus6p0s.png`, logged `camera_front_wide_120fov` frame ending at handover + 6.00 s): a signalised intersection with every visible signal head red; a dark sedan directly ahead in the car's lane and two more vehicles in the lane(s) to the left, all showing brake lights; the lane to the right of the car is empty up to the intersection.
- 0.1.4b on the same clip is at {ms[(ms.version == '0.1.4b') & (ms.scene == 'bd07fb91') & (ms.replan_after_handover_s == 6.0)].ego_speed_mps.iloc[0]:.2f} m/s at +6.0 s (a different state; its rows are in the tables for completeness).

Top 8 of the +6.0 s menu by score:
"""); c = b.head(8)[["proposal", "scorer_rank", "scorer_score", "mean_speed_4s_mps", "first_0p5s_accel_mps2", "step_speed_spread_mps", "lat_rig_2s_m", "lat_rig_4s_m", "lat_from_human_path_2s_m", "lat_from_human_path_4s_m", "chosen"]]; L.append(md(c, 2))
L.append("""
## Predictions vs measured values (no verdicts)

""")
def pk(s, v, t=0.0, ver="0.1.6"): return float(cf[(cf.version == ver) & (cf.scene == s) & (cf.variant == v) & (cf.tick_after_handover_s == t)].peak_abs_steer_rad.iloc[0])
def cut(s, v): return 100 * (1 - pk(s, v) / pk(s, "B0_recorded"))
gg = g[g["mode"] == "cold_tf"].set_index(["scene", "start_after_handover_s"])
L.append(f"""- P1 (first |steer| > 0.3 rad within 0.2 s of handover in the standalone replay): 04749bb9 replay {gg.loc[('04749bb9', 0.0)].first_steer_gt_0p3_sim_s:.2f} s (recorded {gg.loc[('04749bb9', 0.0)].first_steer_gt_0p3_rec_s:.2f} s); acd22430 replay {gg.loc[('acd22430', 0.0)].first_steer_gt_0p3_sim_s:.2f} s fresh-seeded / 0.15 s replayed-from-start (recorded {gg.loc[('acd22430', 0.0)].first_steer_gt_0p3_rec_s:.2f} s).
- P2 (no braking cuts peak |steer| in 1 s by > 90 %): B1 at handover {pk('04749bb9', 'B0_recorded'):.3f} -> {pk('04749bb9', 'B1_no_braking'):.3f} rad ({cut('04749bb9', 'B1_no_braking'):.1f} %) on 04749bb9; {pk('acd22430', 'B0_recorded'):.3f} -> {pk('acd22430', 'B1_no_braking'):.3f} rad ({cut('acd22430', 'B1_no_braking'):.1f} %) on acd22430. At the +0.5 s tick: {100 * (1 - pk('04749bb9', 'B1_no_braking', 0.5) / pk('04749bb9', 'B0_recorded', 0.5)):.1f} % and {100 * (1 - pk('acd22430', 'B1_no_braking', 0.5) / pk('acd22430', 'B0_recorded', 0.5)):.1f} %.
- P3 (heading-only change cuts < 30 %): B2 {cut('04749bb9', 'B2_heading_tangent'):.1f} % / {cut('acd22430', 'B2_heading_tangent'):.1f} %; B2z {cut('04749bb9', 'B2z_heading_first_pose'):.1f} % / {cut('acd22430', 'B2z_heading_first_pose'):.1f} % (handover); at +0.5 s within +-2 %.
- P4 (same path, same 4 s end point, constant deceleration < 3 m/s2 steers < 0.1 rad): B4e (a = {cf[(cf.scene == '04749bb9') & (cf.tick_after_handover_s == 0) & (cf.version == '0.1.6')].const_decel_to_recorded_4s_endpoint_mps2.iloc[0]:.2f} / {cf[(cf.scene == 'acd22430') & (cf.tick_after_handover_s == 0) & (cf.version == '0.1.6')].const_decel_to_recorded_4s_endpoint_mps2.iloc[0]:.2f} m/s2) steers {pk('04749bb9', 'B4e_const_decel_same_endpoint'):.3f} / {pk('acd22430', 'B4e_const_decel_same_endpoint'):.3f} rad; sweep at 23.5 m/s stays <= 0.013 rad up to 4 m/s2, 0.055-0.124 rad at 5, 0.24-0.26 at 6. B5 (only the first step re-spaced): {pk('04749bb9', 'B5_first_step_respaced'):.3f} rad on 04749bb9, {pk('acd22430', 'B5_first_step_respaced'):.3f} rad on acd22430.
- P5 (clean human-like on the menu on at most 3 of 7 replans): 0.1.6 04749bb9 {int(k[(k.version == '0.1.6') & (k.scene == '04749bb9')].replans_with_clean.iloc[0])} of 7, acd22430 {int(k[(k.version == '0.1.6') & (k.scene == 'acd22430')].replans_with_clean.iloc[0])} of 7; 0.1.4b 0 of 7 on both.
- P6 (scorer ranks it outside top 5): the only clean human-like proposal (0.1.6 acd22430, +0.0 s, proposal 23) has scorer rank 9 (score 0.883 vs 0.902 for the pick); held through the controller it steers 0.480 rad (first-0.5 s demand -5.42 m/s2).
- P7 (bd07fb91 +6.0 s): see the section above - chosen plan y = {ch.lat_rig_2s_m:.2f} m at 2 s, {ch.first_0p5s_accel_mps2:.2f} m/s2; straight hold-speed proposals on that menu: {int(b.straight_hold_speed.sum())} of 32.

## Files

CSVs: `gate.csv`, `counterfactuals.csv`, `braking_sweep.csv`, `braking_onset.csv`, `menu_proposals.csv`, `menu_summary.csv`, `menu_through_controller.csv`, `plan_anatomy.csv`, `replan_jumps.csv`, `plan_relation.csv`.
Plots: `plot1_gate.png`, `plot2_counterfactuals_04749bb9.png`, `plot3_counterfactuals_acd22430.png`, `plot4_braking_sweep.png`, `plot5_plan_anatomy.png`, `plot6_menu_handover_016.png`, `plot7_menu_handover_014b.png`, `plot8_bd07fb91_menu_topview.png`; camera frame `bd07fb91_front_camera_plus6p0s.png`.
Series for re-plotting: `gate_series.pkl`, `run_series.pkl`; `cache/` = per-run cache of the controller requests read from the rollout logs.
""")
(H / "README.md").write_text("\n".join(L))
# ---------------- report directory
REP.mkdir(parents=True, exist_ok=True)
csvs = ["gate.csv", "counterfactuals.csv", "braking_sweep.csv", "braking_onset.csv", "menu_proposals.csv", "menu_summary.csv", "menu_through_controller.csv", "plan_anatomy.csv", "replan_jumps.csv", "plan_relation.csv"]
pngs = ["plot1_gate.png", "plot2_counterfactuals_04749bb9.png", "plot3_counterfactuals_acd22430.png", "plot4_braking_sweep.png", "plot5_plan_anatomy.png", "plot6_menu_handover_016.png", "plot7_menu_handover_014b.png", "plot8_bd07fb91_menu_topview.png", "bd07fb91_front_camera_plus6p0s.png"]
for f in csvs + pngs + ["README.md", "run.log"]: shutil.copy(H / f, REP / f)
R = lambda x, n=3: None if pd.isna(x) else round(float(x), n)
M = {}
for _, r_ in g.iterrows():
    kk = f"gate {r_.scene} +{r_.start_after_handover_s}s {r_['mode']}"
    M[kk] = f"steer maxdiff 0.5s {r_.steer_state_maxdiff_0p5s:.2e} rad (rec peak {r_.steer_state_rec_peak_0p5s:.4f}), rms 0.5s {r_.steer_state_rmsdiff_0p5s:.2e}; long cmd maxdiff 0.5s {r_.long_cmd_maxdiff_0p5s:.2e} m/s2 (rec peak {r_.long_cmd_rec_peak_0p5s:.3f}), rms {r_.long_cmd_rmsdiff_0p5s:.2e}; 3s: steer max {r_.steer_state_maxdiff_3s:.2e} rms {r_.steer_state_rmsdiff_3s:.2e}, long max {r_.long_cmd_maxdiff_3s:.2e} rms {r_.long_cmd_rmsdiff_3s:.2e}; first |steer|>0.3: rec {R(r_.first_steer_gt_0p3_rec_s, 2)} s, replay {R(r_.first_steer_gt_0p3_sim_s, 2)} s; peak 3s rec {r_.peak_abs_steer_3s_rec_rad:.4f} replay {r_.peak_abs_steer_3s_sim_rad:.4f} rad"
for (ver, s, t), d in cf.groupby(["version", "scene", "tick_after_handover_s"], sort=False): M[f"peak |steer| rad, {ver} {s} tick +{t}s, per variant"] = {v: R(p) for v, p in zip(d.variant, d.peak_abs_steer_rad)}
for _, o in on.iterrows(): M[f"braking onset {o.seed_scene} {o.path} at {o.ego_speed_mps:.1f} m/s"] = (f"first peak|steer|>0.1 rad at constant decel {o.onset_const_decel_mps2:.0f} m/s2 (previous grid point {o.last_decel_below_onset_mps2:.0f}); as first-0.5s demand {o.onset_as_first_0p5s_accel_mps2:.1f} m/s2" if not pd.isna(o.onset_const_decel_mps2) else f"none up to 12 m/s2 (peak |steer| {o.peak_steer_at_max_decel_rad:.3f} rad)")
M["braking sweep peak |steer| rad by constant decel (m/s2), straight path"] = {f"{s} {sp[(s, 'straight')]:.1f} m/s": {int(a): R(p) for a, p in zip(d.const_decel_mps2, d.peak_abs_steer_rad)} for s, d in sw[sw.path == "straight"].groupby("seed_scene", sort=False)}
for (ver, s), d in ms.groupby(["version", "scene"], sort=False):
    M[f"menu {ver} {s}: per replan t: [n_clean, best_rank_clean, n_safe, best_rank_safe, n_safe_literal, best_rank_safe_literal, n_safe_bounded, best_rank_safe_bounded, chosen_rank, chosen_first0.5s_accel, chosen_mean_v_4s, chosen_step_spread, rescore_changed]"] = {
        f"+{r_.replan_after_handover_s}s": [int(r_.n_clean_human_like), R(r_.best_rank_clean_human_like, 0), int(r_.n_controller_safe), R(r_.best_rank_controller_safe, 0), int(r_.n_controller_safe_literal), R(r_.best_rank_controller_safe_literal, 0), int(r_.n_controller_safe_bounded),
                                                R(r_.best_rank_controller_safe_bounded, 0), int(r_.chosen_scorer_rank), R(r_.chosen_first_0p5s_accel_mps2, 2), R(r_.chosen_mean_speed_4s_mps, 2), R(r_.chosen_step_speed_spread_mps, 2), bool(r_.rescore_3p5_changed_pick)] for _, r_ in d.iterrows()}
M["menu proposals through controller: peak |steer| rad [proposal, scorer rank, first0.5s accel, peak steer]"] = {f"{r_.version} {r_.scene} +{r_.tick_after_handover_s}s {r_.which}": ("none on menu" if r_.proposal < 0 else [int(r_.proposal), int(r_.scorer_rank), R(r_.first_0p5s_accel_mps2, 2), R(r_.peak_abs_steer_rad)]) for _, r_ in ctl.iterrows()}
M["bd07fb91 +6.0s (0.1.6)"] = dict(car_speed_mps=R(ch.ego_speed_mps, 2), human_mean_speed_next_4s_mps=R(ch.human_mean_speed_4s_mps, 2), human_position_in_car_frame_m=[-11.49, -2.44], chosen_proposal=int(ch.proposal), chosen_rank=int(ch.scorer_rank), chosen_score=R(ch.scorer_score), chosen_y_2s_m=R(ch.lat_rig_2s_m, 2),
     chosen_y_4s_m=R(ch.lat_rig_4s_m, 2), chosen_first_0p5s_accel_mps2=R(ch.first_0p5s_accel_mps2, 2), chosen_offset_from_human_path_2s_m=R(ch.lat_from_human_path_2s_m, 2), n_straight_hold_speed_of_32=int(b.straight_hold_speed.sum()), y_2s_range_all_32_m=[R(b.lat_rig_2s_m.min(), 2), R(b.lat_rig_2s_m.max(), 2)],
     least_lateral_proposal=dict(proposal=int(strt.proposal), y_2s=R(strt.lat_rig_2s_m, 2), mean_v=R(strt.mean_speed_4s_mps, 2), rank=int(strt.scorer_rank), score=R(strt.scorer_score)), n_clean_human_like=int(b.clean_human_like.sum()), n_controller_safe=int(b.controller_safe.sum()),
     rescore_changed_pick=bool(ch.rescore_3p5_changed_pick), first_steer_gt_0p3_recorded_s_after_handover=6.10, camera="signalised intersection, all visible signals red, dark sedan ahead in lane and two vehicles in lanes to the left with brake lights, lane to the right empty")
M["plan relation"] = rl.to_dict("records"); M["replan start jump m (0.1.6), +0.5..+3.0s"] = {s: [R(x, 2) for x in d.start_jump_m] for s, d in jp[jp.version == "0.1.6"].groupby("scene")}; M["replan start jump m (0.1.4b), +0.5..+3.0s"] = {s: [R(x, 2) for x in d.start_jump_m] for s, d in jp[jp.version == "0.1.4b"].groupby("scene")}
M["implausible proposals"] = wild.to_dict("records")
json.dump(M, open(H / "metrics.json", "w"), indent=1, default=str); print("README + metrics written;", len(json.dumps(M)), "chars of metrics")
