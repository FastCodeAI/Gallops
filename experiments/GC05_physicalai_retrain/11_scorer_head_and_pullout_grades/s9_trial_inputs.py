#!/usr/bin/env python
"""S9 C.4, EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION. Read-only replay of the S5 trial's two stuck pull-out scenes (d2715023, b44a4a44,
adapter 0.1.6) to recover the model INPUT at every replan, because the trial folder stores the 32 plans and scores but not the 256
features the gate head reads. Method and code path of 07_closed_loop_test/s5_menus.py (same planner, same weights, CPU, momentum on, replans in
time order so the previous plan chains as it did live); the only differences: the input dict handed to the model is saved, and every
output and log goes to 11_scorer_head_and_pullout_grades (nothing is written under 07_closed_loop_test/ or the run folders). The inputs are camera frames of exam clips:
they are written to the NAS scratch folder 11_scorer_head_and_pullout_grades/trial_tmp/, read once by s9_trial_gate.py, and deleted by it.

  cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python <this file>
"""
from __future__ import annotations
import asyncio, glob, json, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ALP = HERE.parents[2]; (HERE / "logs").mkdir(exist_ok=True); TMP = pathlib.Path("{runs}/drivejepa_retrain/11_scorer_head_and_pullout_grades/trial_tmp"); TRIAL = HERE.parent / "07_closed_loop_test" / "menus"
sys.path.insert(0, str(ALP.parent / "analysis" / "drivejepa" / "mode1")); import mode1_common as C  # noqa: E402
import dj014_accel_counterfactual as H  # noqa: E402
sys.path.append(str(ALP / "adapters"))
import drivejepa016.planner as P16  # noqa: E402
import drivejepa016.constants as K  # noqa: E402
from drivejepa016.bridge import BridgedDriveJEPA  # noqa: E402
assert (K.ADAPTER_VERSION, K.BONNET_MASK_ROWS, K.ZERO_ACCELERATION, K.ZERO_MOMENTUM_ACCELERATION) == ("0.1.6", 37, True, True)

def main():
    t0 = time.time(); model = BridgedDriveJEPA(device="cpu", stderr_log=HERE / "logs" / "trial_worker.log"); print(f"model up {time.time() - t0:.0f}s", flush=True)
    class Capture:
        def __init__(self, fn): self.fn = fn; self.last = None; self.feat = None
        def __call__(self, f): self.feat = f; self.last = self.fn(f); return self.last
    cap = Capture(model); planner = P16.DriveJEPAPlanner(model_fn=cap, front_camera_id=C.FRONT, expected_frame_interval_us=50_000, momentum="on"); summ = []
    for sc in ("d2715023", "b44a4a44"):
        js = json.load(open(TRIAL / f"d016_{sc}.json")); run = pathlib.Path(js["run"]); out = TMP / sc; out.mkdir(parents=True, exist_ok=True)
        (asl,) = glob.glob(str(run / "rollouts" / "clipgt-*" / "*" / "rollout.asl"))
        idx = asyncio.run(C.replay_full(asl, want_ticks=None, collect_all=True)); asks = idx["asks"]; rec = idx["recorded"]; ho = asks[0] + 3_000_000; assert ho == js["handover_tn"]
        replans = [tn for tn in asks if rec.get(tn, {}).get("n") == 9 and (rec[tn].get("debug") or {}).get("replan") is True]; assert len(replans) == js["n_replans"]
        got = asyncio.run(C.replay_full(asl, want_ticks=set(replans), collect_all=False)); uuid = None; worst = 0.0; nmis = 0
        for tn in replans:
            snap = got["snaps"][tn]; view = H.build_view(snap, None); uuid = view.session_uuid; planner._cadence_checked.clear()
            pr = planner.plan(view, time_now_us=tn, time_query_us=snap["time_query_us"]); o = cap.last; d = pr.debug; lg = rec[tn]["debug"]; k = asks.index(tn); P = np.asarray(o["proposals"])[0]
            st = np.load(TRIAL / f"d016_{sc}" / f"replan_{k:03d}.npz", allow_pickle=True); dev = float(np.abs(P - st["proposals"]).max()); worst = max(worst, dev); ok = int(d["pick_with_momentum"]) == lg["chosen_index"] == int(st["chosen"]); nmis += (not ok)
            np.savez(out / f"replan_{k:03d}.npz", **{kk: v for kk, v in cap.feat.items() if isinstance(v, np.ndarray)}, proposals=P, S_plain=np.asarray(d["S"], dtype=np.float32), pdm_score=np.asarray(d["calibrated"], dtype=np.float32), chosen=int(d["pick_with_momentum"]), t_after_ho=(tn - ho) / 1e6, idx=k,
                     nonarray_keys=np.array([kk for kk, v in cap.feat.items() if not isinstance(v, np.ndarray)]))
        planner.on_session_close(uuid); summ.append(dict(scene=sc, replans=len(replans), pick_mismatches=nmis, worst_plan_dev_vs_trial_m=worst)); print(summ[-1], flush=True)
    model.close(); json.dump(summ, open(HERE / "c4_trial_replay.json", "w"), indent=1); print("done", f"{time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
