#!/usr/bin/env python
"""S5 TRIAL copy of baseline_014b/b014b_menus.py for the 0.1.6 runs (diff: tables/s5_menus_diff.txt -- package drivejepa016, i.e. the S4
candidate weights with the 0.1.6 switches ON as deployed; helper paths). Original docstring: DRIVEJEPA_014B site support: regenerate the 32-proposal menu at every replan of a 0.1.4b run (the rollout log carries the
scores and picks but not the proposals) through the 0.1.4b planner, real model on CPU, momentum on, replans in time order in one
session so the previous plan chains as it did live; check the regenerated pick equals the logged one. Method of s5/s5_menus.py.
Writes menus/<label>/replan_<k>.npz + menus/<label>.json.

  cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python <this file> LABEL=RUN_DIR [...]
"""
from __future__ import annotations

import asyncio, glob, json, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent; ALP = HERE.parents[2]; (HERE / "logs").mkdir(exist_ok=True)
sys.path.insert(0, str(ALP.parent / "analysis" / "drivejepa" / "mode1")); import mode1_common as C  # noqa: E402
import dj014_accel_counterfactual as H  # noqa: E402
sys.path.append(str(ALP / "adapters"))
import drivejepa016.planner as P14B  # noqa: E402   (name kept from the original; this is the 0.1.6 planner)
import drivejepa016.constants as K  # noqa: E402
from drivejepa016.bridge import BridgedDriveJEPA  # noqa: E402
assert (K.ADAPTER_VERSION, K.BONNET_MASK_ROWS, K.ZERO_ACCELERATION, K.ZERO_MOMENTUM_ACCELERATION) == ("0.1.6", 37, True, True)


def main(specs):
    t0 = time.time(); model = BridgedDriveJEPA(device="cpu", stderr_log=HERE / "logs" / "menus_worker.log"); print(f"model up {time.time() - t0:.0f}s", flush=True)

    class Capture:
        def __init__(self, fn): self.fn = fn; self.last = None; self.feat = None
        def __call__(self, f): self.feat = f; self.last = self.fn(f); return self.last
    cap = Capture(model); planner = P14B.DriveJEPAPlanner(model_fn=cap, front_camera_id=C.FRONT, expected_frame_interval_us=50_000, momentum="on")
    for spec in specs:
        label, run = spec.split("="); run = pathlib.Path(run); out = HERE / "menus" / label
        if (HERE / "menus" / f"{label}.json").is_file(): print(f"{label}: done already, skip", flush=True); continue
        out.mkdir(parents=True, exist_ok=True); t1 = time.time()
        (asl,) = glob.glob(str(run / "rollouts" / "clipgt-*" / "*" / "rollout.asl"))
        idx = asyncio.run(C.replay_full(asl, want_ticks=None, collect_all=True)); asks = idx["asks"]; rec = idx["recorded"]; ho = asks[0] + 3_000_000
        replans = [tn for tn in asks if rec.get(tn, {}).get("n") == 9 and (rec[tn].get("debug") or {}).get("replan") is True]
        got = asyncio.run(C.replay_full(asl, want_ticks=set(replans), collect_all=False)); mism = []; worst = 0.0; uuid = None
        for tn in replans:
            snap = got["snaps"][tn]; view = H.build_view(snap, None); uuid = view.session_uuid; planner._cadence_checked.clear()
            pr = planner.plan(view, time_now_us=tn, time_query_us=snap["time_query_us"]); o = cap.last; d = pr.debug; lg = rec[tn]["debug"]
            P = np.asarray(o["proposals"])[0]; es = cap.feat["ego_status"][0, -1]; ego_v = float(np.hypot(es[3], es[4])); step1 = (np.hypot(*P[:, 0, :2].T) / 0.5 - ego_v) / 0.5
            chosen, plain = int(d["pick_with_momentum"]), int(d["pick_without_momentum"])
            rig, _ = H.recorded_rig(rec[tn]); dev = float(np.max(np.abs(P[chosen][:, :2] - rig[:, :2]))); worst = max(worst, dev)
            ok = chosen == lg["chosen_index"] == lg["pick_with_momentum"] and plain == lg["pick_without_momentum"] and d["momentum_code"] == lg["momentum_code"] and dev <= 1e-3
            if not ok: mism.append(dict(t=(tn - ho) / 1e6, chosen=chosen, logged=lg["chosen_index"], plain=plain, logged_plain=lg["pick_without_momentum"], dev=dev))
            k = asks.index(tn)
            np.savez(out / f"replan_{k:03d}.npz", proposals=P, pdm_score=np.asarray(d["calibrated"], dtype=np.float32), S_plain=np.asarray(d["S"], dtype=np.float32),
                     S_c=np.asarray(d["S_c"] if d["S_c"] is not None else [np.nan] * 32, dtype=np.float32), step1_decel=step1, chosen=chosen, shipped=plain, changed=bool(chosen != plain),
                     code=str(d["momentum_code"]), tn=tn, idx=k, t_after_ho=(tn - ho) / 1e6, ego_v=ego_v, gate_dev_m=dev)
        planner.on_session_close(uuid)
        js = dict(label=label, run=str(run), n_replans=len(replans), n_mismatch=len(mism), mismatches=mism, worst_gate_dev_m=worst, handover_tn=ho, secs=round(time.time() - t1, 1))
        json.dump(js, open(HERE / "menus" / f"{label}.json", "w"), indent=1, default=float)
        print(f"{label}: {len(replans)} replans, mismatches {len(mism)}, worst dev {worst:.2e} m ({js['secs']:.0f}s)", flush=True)
    model.close()


if __name__ == "__main__":
    main(sys.argv[1:])
