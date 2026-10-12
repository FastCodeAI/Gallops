#!/usr/bin/env python
"""Step 0 build gate (offline, CPU, no simulator): replay the two RECORDED 0.1.7 runs through OraclePlanner, replans in time order,
real model in the 0.1.7 worker, live grader. Method of 07_closed_loop_test/runs_017/s017_menus.py (same replay helpers). Read-only on the runs.

Checks, per scene:
  G1 the 32 plans at every replan equal the menus stored for the 0.1.7 trial (NAS 07_closed_loop_test_017/menus)        -> same generator, same inputs
  G2 the scorer's pick (logged, unused) equals the trial's stored pick without momentum ("shipped")           -> the scorer still runs as in 0.1.7
  G3 the plan handed back is proposals[picked] and the debug payload names the same slot                      -> the one change does what it says
  G4 the car pose given to the grader equals the simulator's own state of that run at the anchor time (< 1 cm)-> right frame, right clock
  G5 the grader's collision-free flag agrees with an independent box-overlap check of the same plan            -> informational, share printed
     (T04's check: plan interpolated linearly, ego box against every recorded box, zero margin; t04_common.py:50-74)
  G6 grading time per replan
Writes NAS gate/<scene>_steps.jsonl (the oracle's picks on the STATIONARY 0.1.7 drive; open loop) and tables/gate_offline.json.

  cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" DRIVEJEPA_MOMENTUM=off alpasim/.venv/bin/python -B analysis_3.0/Retrain-data/13_best_possible_pick/gate_offline.py
"""
from __future__ import annotations
import asyncio, glob, importlib.util, json, math, os, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent; ALP = HERE.parents[2]; NAS = pathlib.Path("{runs}/drivejepa_retrain/13_best_possible_pick"); T017 = pathlib.Path("{runs}/drivejepa_retrain/07_closed_loop_test_017")
assert os.environ.get("DRIVEJEPA_MOMENTUM") == "off" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
spec = importlib.util.spec_from_file_location("serve_oracle", HERE / "draft" / "serve_drivejepa_0_1_7_oracle.py"); L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
planner_mod, divider_mod, K, real_path = L.S017._resolve_planner()          # the frozen 0.1.7 package, checked by 0.1.7's own launcher, BEFORE any other path is added
sys.path.insert(1, str(ALP.parent / "analysis" / "drivejepa" / "mode1")); sys.path.insert(2, str(ALP.parent))
import mode1_common as C  # noqa: E402
import dj014_accel_counterfactual as H  # noqa: E402
import drivejepa.planner as P  # noqa: E402
assert pathlib.Path(os.path.realpath(P.__file__)) == real_path and "runs_017/adapters/drivejepa017/planner.py" in str(real_path), real_path
from drivejepa.bridge import BridgedDriveJEPA  # noqa: E402
from drivejepa017_oracle import oracle_planner as OP  # noqa: E402
SCENES = ["d2715023", "b44a4a44"]; TS = np.round(np.arange(0, 4.0001, 0.1), 1); T8 = np.arange(0, 4.0001, 0.5)


def sat(c1, yaw1, h1, c2, yaw2, h2):                                           # t04_common.py:50-59
    u1 = np.stack([np.c_[np.cos(yaw1), np.sin(yaw1)], np.c_[-np.sin(yaw1), np.cos(yaw1)]], 1); u2 = np.stack([np.c_[np.cos(yaw2), np.sin(yaw2)], np.c_[-np.sin(yaw2), np.cos(yaw2)]], 1)
    d = c2[None] - c1[:, None]; dots = np.abs(np.einsum("mik,njk->mnij", u1, u2)); sep = np.zeros(d.shape[:2], bool)
    for i in range(2):
        sep |= np.abs(np.einsum("mnk,mk->mn", d, u1[:, i])) > h1[i] + dots[:, :, i, 0] * h2[None, :, 0] + dots[:, :, i, 1] * h2[None, :, 1]
        sep |= np.abs(np.einsum("mnk,nk->mn", d, u2[:, i])) > h2[None, :, i] + dots[:, :, 0, i] * h1[0] + dots[:, :, 1, i] * h1[1]
    return ~sep


def box_free(z, t0_us, pose, plan8):
    """True when the plan (rig poses, linear between waypoints) never overlaps a recorded box over the part of 4 s the log covers."""
    x0, y0, yaw0 = pose; c, s = math.cos(yaw0), math.sin(yaw0); p9 = np.vstack([[0, 0, 0], plan8]); d = np.c_[np.interp(TS, T8, p9[:, 0]), np.interp(TS, T8, p9[:, 1]), np.interp(TS, T8, np.unwrap(p9[:, 2]))]
    W = np.c_[x0 + c * d[:, 0] - s * d[:, 1], y0 + s * d[:, 0] + c * d[:, 1], yaw0 + d[:, 2]]; ap_t = z["ap_t"]; h1 = z["ego_dims"] / 2; r2c = float(z["rig_to_aabb_x"])
    for k, t in enumerate(TS):
        tq = t0_us + int(round(t * 1e6)); i = int(np.argmin(np.abs(ap_t - tq)))
        if abs(int(ap_t[i]) - tq) > 26_000: continue
        a = z["A"][i]; ok = np.isfinite(a[:, 0]); cc = np.array([[W[k, 0] + r2c * math.cos(W[k, 2]), W[k, 1] + r2c * math.sin(W[k, 2])]]); near = ok & (np.hypot(a[:, 0] - cc[0, 0], a[:, 1] - cc[0, 1]) < 12.0)
        if near.any() and sat(cc, np.array([W[k, 2]]), h1, a[near, :2], a[near, 2], z["dims"][near] / 2)[0].any(): return False
    return True


def main():
    t0 = time.time(); (NAS / "gate").mkdir(parents=True, exist_ok=True); model = BridgedDriveJEPA(device="cpu", stderr_log=NAS / "gate" / "gate_worker.log"); print(f"model up {time.time() - t0:.0f}s", flush=True); res = {}
    for sc in SCENES:
        t1 = time.time(); run = T017 / "runs" / f"drivejepa_017_{sc}"; actors = NAS / "actors" / f"{sc}.npz"; z = np.load(actors); slog = NAS / "gate" / f"{sc}_steps.jsonl"; slog.unlink(missing_ok=True)
        grader = OP.OracleGrader(str(actors), str(NAS / "gate" / f"{sc}_grader.log")); planner = OP.OraclePlanner(model_fn=model, front_camera_id=C.FRONT, expected_frame_interval_us=50_000, grader=grader, step_log=str(slog))
        assert planner.version_id == "drivejepa-v2-driver-0.1.7-oracle" and not planner._momentum_on
        (asl,) = glob.glob(str(run / "rollouts" / "clipgt-*" / "*" / "rollout.asl")); idx = asyncio.run(C.replay_full(asl, want_ticks=None, collect_all=True)); asks = idx["asks"]; rec = idx["recorded"]; ho = asks[0] + 3_000_000
        replans = [tn for tn in asks if rec.get(tn, {}).get("n") == 9 and (rec[tn].get("debug") or {}).get("replan") is True]; got = asyncio.run(C.replay_full(asl, want_ticks=set(replans), collect_all=False))
        g1 = g4 = 0.0; g2 = g3 = 0; agree = tot = 0; ms = []; ego = z["ego"]; picks = []
        for tn in replans:
            snap = got["snaps"][tn]; view = H.build_view(snap, None); planner._cadence_checked.clear(); pr = planner.plan(view, time_now_us=tn, time_query_us=snap["time_query_us"]); k = asks.index(tn)
            r = json.loads(slog.read_text().splitlines()[-1]); st = np.load(T017 / "menus" / f"d017_{sc}" / f"replan_{k:03d}.npz", allow_pickle=True); Pm = np.asarray(r["proposals"])
            g1 = max(g1, float(np.abs(Pm - st["proposals"]).max())); g2 += int(r["scorer_pick"] != int(st["shipped"]))
            g3 += int(not (np.allclose(pr.positions_in_rig[:, :2], st["proposals"][r["picked"]][:, :2], atol=1e-6) and pr.debug["chosen_index"] == r["picked"] and pr.debug["oracle"] is True))
            i = int(np.argmin(np.abs(ego[:, 0] - r["anchor_time_us"]))); assert abs(ego[i, 0] - r["anchor_time_us"]) < 1000; g4 = max(g4, float(np.hypot(ego[i, 1] - r["car_x"], ego[i, 2] - r["car_y"])), abs(float(ego[i, 3] - r["car_yaw"])))
            for j in range(32):
                bf = box_free(z, r["model_t0_us"], (r["car_x"], r["car_y"], r["car_yaw"]), st["proposals"][j]); agree += int(bf == (r["nc_strict"][j] == 1.0)); tot += 1
            ms.append(r["oracle_ms"]); picks.append(dict(step=r["step"], t=(tn - ho) / 1e6, picked=r["picked"], fallback=r["fallback"], grade=round(max(r["grade"]), 2), wp8=(round(r["picked_wp8_long"], 2), round(r["picked_wp8_lat"], 2)), scorer=r["scorer_pick"], free=r["n_collision_free"], widest=round(r["menu_widest_abs_lat_wp8"], 2)))
        planner.on_session_close(view.session_uuid); grader.close()
        res[sc] = dict(replans=len(replans), G1_menu_max_abs_dev_m=g1, G2_scorer_pick_mismatches=g2, G3_handback_mismatches=g3, G4_car_pose_max_dev=g4, G5_box_check_agreement=agree / tot, G5_plans=tot, G6_oracle_ms_median=float(np.median(ms)), G6_oracle_ms_max=float(max(ms)),
                       fallbacks=int(sum(p["fallback"] for p in picks)), secs=round(time.time() - t1, 1), ok=bool(g1 < 1e-3 and g2 == 0 and g3 == 0 and g4 < 0.01))
        print(sc, json.dumps(res[sc]), flush=True)
        for p in picks: print("   ", p, flush=True)
    model.close(); res["ok"] = all(res[s]["ok"] for s in SCENES); res["planner"] = str(real_path); res["version_id"] = OP.ORACLE_VERSION_ID
    json.dump(res, open(HERE / "tables" / "gate_offline.json", "w"), indent=1); print("GATE", "PASSED" if res["ok"] else "FAILED", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
