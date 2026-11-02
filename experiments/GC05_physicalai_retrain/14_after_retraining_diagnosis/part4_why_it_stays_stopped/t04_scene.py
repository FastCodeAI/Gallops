#!/usr/bin/env python
"""T04 Part A (human timeline, what is in front, frames, fed inputs) and Part C (controller check). CPU only.
  CUDA_VISIBLE_DEVICES="" {runs}/alpasim_bulk/nurec-venv/bin/python t04_scene.py
"""
import math, pickle, sys
import numpy as np, pandas as pd, cv2
import t04_common as T
from t04_menu import poly
C = T.C
sys.path.insert(0, str(T.HERE.parent / "part3_swerve_fixes"))
import t03_common as T3  # noqa: E402


def part_a():
    tl, fr, inp = [], [], []
    for s in T.STUCK:
        o = T.load("0.1.6", s); ob = T.load("0.1.4b", s); ho = o["ho"]; g0 = T.gt_at(o, [ho])[0]; e0 = o["egot"][ho][1:4]
        for t in np.round(np.arange(-3.0, 16.76, 0.25), 2):
            h = T.gt_at(o, [ho + t * 1e6])[0]; r = C.to_rig(h[None], g0)[0]; row = dict(scene=s, t_after_handover_s=t, human_speed_mps=float(np.interp(t, o["gt_t"], o["gt_v"])), human_forward_m=r[0], human_lateral_left_m=r[1], human_heading_change_rad=r[2])
            for nm, oo in (("016", o), ("014b", ob)):
                k = int(np.argmin(np.abs(oo["ego"][:, 0] - (ho + t * 1e6)))); e = oo["ego"][k]; q = C.to_rig(e[None, 1:4], g0)[0]; row.update({f"ego_{nm}_speed_mps": float(np.hypot(e[4], e[5])), f"ego_{nm}_forward_m": q[0], f"ego_{nm}_lateral_left_m": q[1]})
            tl.append(row)
        # what is in front at handover
        i0 = int(np.argmin(np.abs(o["ap_t"] - ho))); A = o["A"]; ep = poly(np.array([e0[0] + o["rig_to_aabb_x"] * math.cos(e0[2]), e0[1] + o["rig_to_aabb_x"] * math.sin(e0[2])]), e0[2], o["ego_dims"]); front_x = o["rig_to_aabb_x"] + o["ego_dims"][0] / 2
        for j, a in enumerate(o["ids"]):
            if not np.isfinite(A[i0, j, 0]): continue
            p = poly(A[i0, j, :2].astype(float), float(A[i0, j, 2]), o["dims"][j]); cr = C.to_rig(np.c_[np.array(p.exterior.coords)[:4], np.zeros(4)], e0); ctr = C.to_rig(A[i0, j][None].astype(float), e0)[0]; dist = ep.distance(p)
            if dist > 15: continue
            tr = A[:, j]; ok = np.isfinite(tr[:, 0]); tt = (o["ap_t"][ok] - ho) / 1e6; xy = tr[ok, :2]; sp = np.hypot(*np.gradient(xy, tt, axis=0).T) if ok.sum() > 2 else np.zeros(ok.sum())
            in_corr = bool(cr[:, 1].min() < o["ego_dims"][1] / 2 and cr[:, 1].max() > -o["ego_dims"][1] / 2 and cr[:, 0].min() > 0)
            fr.append(dict(scene=s, actor=a, label=o["labels"][j], length_m=o["dims"][j][0], width_m=o["dims"][j][1], centre_x_fwd_m=ctr[0], centre_y_left_m=ctr[1], yaw_rel_rad=ctr[2], nearest_corner_x_m=cr[:, 0].min(), y_min_m=cr[:, 1].min(), y_max_m=cr[:, 1].max(),
                           box_to_box_distance_m=dist, ahead_in_ego_width_corridor=in_corr, gap_from_ego_front_bumper_m=(cr[:, 0].min() - front_x) if in_corr else np.nan,
                           present_from_s=tt[0], present_to_s=tt[-1], displacement_over_clip_m=float(np.hypot(*(xy[-1] - xy[0]))), max_speed_mps=float(sp.max()), moves=bool(np.hypot(*(xy[-1] - xy[0])) > 0.5)))
        # frames
        for ver, oo in (("0.1.6", o), ("0.1.4b", ob)):
            for rel, (fs, fe, b) in sorted(oo["img"].items()):
                im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                if ver == "0.1.6" or rel == 5.0: cv2.imwrite(str(T.HERE / f"frame_{s}_{ver.replace('.', '')}_plus{rel:.0f}s.png"), im)
                cv2.imwrite(f"/tmp/t04_{s}_{ver}_{rel:.0f}.jpg", cv2.resize(im, (im.shape[1] * 720 // im.shape[0] * 0 + 960, 960 * im.shape[0] // im.shape[1])))
        # what the model was fed
        for ver, oo in (("0.1.6", o), ("0.1.4b", ob)):
            F = np.array([(a, b) for a, b, _, _ in oo["frames"]]); H = [h for _, _, h, _ in oo["frames"]]; drv = {int(r[0]): r for r in oo["drv"]}
            for tn, z in T.menus(ver, s).items():
                rel = (tn - ho) / 1e6
                if rel < 0 or tn not in drv: continue
                k = int(np.flatnonzero(F[:, 1] <= tn)[-1]); k2 = int(np.argmin(np.abs(F[:k, 1] - (F[k, 1] - 500_000)))); d = drv[tn]
                inp.append(dict(scene=s, version=ver, t_after_handover_s=rel, model_fed_speed_mps=float(z["ego_v"]), reported_speed_mps=float(np.hypot(d[4], d[5])), reported_accel_x_mps2=d[6], reported_accel_y_mps2=d[7],
                                accel_fed_to_model="zeroed (0.1.6)" if ver == "0.1.6" else "as reported", newest_frame_end_minus_now_s=(F[k, 1] - tn) / 1e6, older_frame_end_minus_now_s=(F[k2, 1] - tn) / 1e6, frame_gap_s=(F[k, 1] - F[k2, 1]) / 1e6,
                                frames_byte_identical=bool(H[k] == H[k2]), human_speed_mps=float(np.interp(rel, o["gt_t"], o["gt_v"])), true_ego_speed_mps=float(np.hypot(*oo["egot"][tn][4:6]))))
    pd.DataFrame(tl).to_csv(T.HERE / "human_timeline.csv", index=False, float_format="%.4f"); F = pd.DataFrame(fr).sort_values(["scene", "box_to_box_distance_m"]); F.to_csv(T.HERE / "front_actors.csv", index=False, float_format="%.4g")
    I = pd.DataFrame(inp); I.to_csv(T.HERE / "model_inputs.csv", index=False, float_format="%.5g")
    print(F[F.box_to_box_distance_m < 8][["scene", "actor", "label", "length_m", "width_m", "centre_x_fwd_m", "centre_y_left_m", "box_to_box_distance_m", "ahead_in_ego_width_corridor", "gap_from_ego_front_bumper_m", "displacement_over_clip_m", "max_speed_mps"]].round(2).to_string())
    print(I.groupby(["scene", "version"]).agg(n=("frames_byte_identical", "size"), identical=("frames_byte_identical", "sum"), fed_max=("model_fed_speed_mps", "max"), fed_med=("model_fed_speed_mps", "median"), err=("reported_speed_mps", lambda x: 0), gap_min=("frame_gap_s", "min"), gap_max=("frame_gap_s", "max")).round(3).to_string())
    print("fed vs true speed max abs diff:", float((I.model_fed_speed_mps - I.true_ego_speed_mps).abs().max()), "fed vs reported:", float((I.model_fed_speed_mps - I.reported_speed_mps).abs().max()))


def part_c():
    R = pd.read_csv(T.HERE / "replans.csv"); out = []; series = {}
    for s in T.STUCK:
        o = T.load("0.1.6", s); d = T3.load(s); ho = d["ho"]; M = T.menus("0.1.6", s); hs = T.human_start(o); hs_tick = math.ceil(hs / 0.5 - 1e-9) * 0.5; sim = C.Sim(d, d["ticks"][0])
        for name, rel in (("handover", 0.0), ("human_start", hs_tick), ("plus5s", 5.0)):
            ts = int(ho + rel * 1e6); z = M[ts]; sim.feed_recorded(ts); ego = o["egot"][ts][1:4]; r = R[(R.scene == s) & (R.version == "0.1.6") & (np.abs(R.t_after_handover_s - rel) < 1e-6)].iloc[0]
            th = min(rel, hs_tick); hw = T.gt_at(o, ho + (th + T.T8) * 1e6); hplan = C.to_rig(hw, hw[0]); off = float(np.hypot(*(hw[0, :2] - ego[:2])))
            plans = [("i_chosen", int(z["chosen"])), ("ii_best_usable", r.best_usable_proposal_final), ("iii_best_leave", r.best_leave_proposal_final), ("iv_best_late_go", r.best_late_go_proposal_final), ("v_human_path", -1)]
            P = pd.read_csv(T.HERE / "proposals.csv"); P = P[(P.scene == s) & (P.version == "0.1.6") & (np.abs(P.t_after_handover_s - rel) < 1e-6)]; pl = P[P.leave & P.prompt]
            plans.append(("vi_best_prompt_leave", pl.proposal[pl.rank_final.idxmin()] if len(pl) else np.nan))
            rec = o["ego"]; k2 = int(np.argmin(np.abs(rec[:, 0] - (ts + 2e6)))); rec2 = float(np.hypot(*(rec[k2, 1:3] - ego[:2])))
            for nm, pi in plans:
                base = dict(scene=s, tick=name, t_after_handover_s=rel, plan=nm, proposal=pi, recorded_run_displacement_2s_m=rec2)
                if isinstance(pi, float) and np.isnan(pi): out.append(dict(**base, note="no such proposal on the menu at this tick")); continue
                pi = int(pi); p9 = hplan if pi < 0 else np.r_[np.zeros((1, 3)), z["proposals"][pi].astype(float)]; pr = P[P.proposal == pi].iloc[0] if pi >= 0 else None
                def fn(sm, p9=p9, ts=ts):
                    sm.hold(C.plan_pb(ts, p9), dur_us=2_000_000); return sm.rows()
                rows = T3.branch(sim, fn); m = rows["timestamp_us"] > ts; t = (rows["timestamp_us"][m] - ts) / 1e6; x, y = rows["x"][m], rows["y"][m]; yaw = 2 * np.arctan2(rows["qz"][m], rows["qw"][m]); w = np.c_[x, y, yaw]
                disp = np.hypot(x - ego[0], y - ego[1]); lat = C.to_rig(w, ego)[:, 1]; st = rows["front_steering_angle"][m]; sp = np.hypot(rows["vx"][m], rows["vy"][m]); i1 = int(np.argmin(np.abs(t - 1.0))); arc = np.hypot(*np.diff(p9[:, :2], axis=0).T).cumsum()
                f0 = T.box_check(o, ts, w, t, 0.0); f3 = T.box_check(o, ts, w, t, 0.3)
                out.append(dict(**base, plan_class=("human" if pi < 0 else ("leave" if pr.leave else "stay") + ("/prompt" if pr.prompt else "/not-prompt") + ("/box-free" if pr.box_free else "/overlap")), plan_rank_final=(np.nan if pi < 0 else pr.rank_final), plan_score_final=(np.nan if pi < 0 else pr.score_final),
                                plan_score_plain=(np.nan if pi < 0 else pr.score_plain), plan_arc_1s_m=arc[1], plan_arc_2s_m=arc[3], plan_arc_4s_m=arc[7], moved_1s_m=disp[i1], moved_2s_m=disp[-1], speed_at_2s_mps=sp[-1], peak_abs_steer_rad=float(np.abs(st).max()),
                                lateral_at_2s_m=lat[-1], peak_abs_lateral_m=float(np.abs(lat).max()), driven_overlap=not f0[0], driven_first_overlap_s=f0[1], driven_overlap_actor=f0[2], driven_overlap_margin0p3=not f3[0], driven_first_overlap_margin0p3_s=f3[1],
                                human_path_time_origin_s=(th if pi < 0 else np.nan), human_vs_ego_start_offset_m=(off if pi < 0 else np.nan), note=""))
                series[(s, name, nm)] = (t, disp)
    D = pd.DataFrame(out); D.to_csv(T.HERE / "controller_check.csv", index=False, float_format="%.4g"); pickle.dump(series, open(T.HERE / "cache" / "controller_series.pkl", "wb"))
    print(D[["scene", "tick", "plan", "proposal", "plan_class", "plan_rank_final", "plan_arc_2s_m", "moved_1s_m", "moved_2s_m", "peak_abs_steer_rad", "lateral_at_2s_m", "driven_overlap", "driven_first_overlap_s", "recorded_run_displacement_2s_m"]].round(2).to_string())


if __name__ == "__main__":
    part_a(); part_c()
