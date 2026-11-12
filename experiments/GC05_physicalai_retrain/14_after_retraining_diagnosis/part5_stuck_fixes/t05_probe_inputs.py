#!/usr/bin/env python
"""T05 Part 2, step 1: recover the model INPUT (camera tensors, ego status, the momentum request) at the probe ticks of d2715023 and
b44a4a44 (adapter 0.1.6). Method and code path of Retrain-data/11_scorer_head_and_pullout_grades/s9_trial_inputs.py (same planner, weights, CPU, momentum on,
replans in time order so the previous plan chains as it did live); differences: only the probe ticks are saved, the whole feature dict
(including the non-array momentum request) is pickled, and outputs go to the NAS scratch folder t05_scratch_20261009 and this directory.

  cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python <this file>
"""
from __future__ import annotations
import asyncio, glob, json, pathlib, pickle, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ALP = HERE.parents[2]; TMP = pathlib.Path("{runs}/drivejepa_retrain/t05_scratch_20261009"); TRIAL = ALP / "analysis_3.0/Retrain-data/07_closed_loop_test/menus"
sys.path.insert(0, str(ALP.parent / "analysis" / "drivejepa" / "mode1")); import mode1_common as C  # noqa: E402
import dj014_accel_counterfactual as H  # noqa: E402
sys.path.append(str(ALP / "adapters"))
import drivejepa016.planner as P16  # noqa: E402
import drivejepa016.constants as K  # noqa: E402
from drivejepa016.bridge import BridgedDriveJEPA  # noqa: E402
assert (K.ADAPTER_VERSION, K.BONNET_MASK_ROWS, K.ZERO_ACCELERATION, K.ZERO_MOMENTUM_ACCELERATION) == ("0.1.6", 37, True, True)
TICKS = {"d2715023": (0.0, 1.0, 5.0), "b44a4a44": (0.0, 3.0, 5.0)}


def main():
    t0 = time.time(); model = BridgedDriveJEPA(device="cpu", stderr_log=HERE / "logs" / "probe_inputs_worker.log"); print(f"model up {time.time() - t0:.0f}s", flush=True)
    class Capture:
        def __init__(self, fn): self.fn = fn; self.last = None; self.feat = None
        def __call__(self, f): self.feat = f; self.last = self.fn(f); return self.last
    cap = Capture(model); planner = P16.DriveJEPAPlanner(model_fn=cap, front_camera_id=C.FRONT, expected_frame_interval_us=50_000, momentum="on"); summ = []
    for sc, want in TICKS.items():
        js = json.load(open(TRIAL / f"d016_{sc}.json")); run = pathlib.Path(js["run"]); out = TMP / sc; out.mkdir(parents=True, exist_ok=True)
        (asl,) = glob.glob(str(run / "rollouts" / "clipgt-*" / "*" / "rollout.asl"))
        idx = asyncio.run(C.replay_full(asl, want_ticks=None, collect_all=True)); asks = idx["asks"]; rec = idx["recorded"]; ho = asks[0] + 3_000_000; assert ho == js["handover_tn"]
        replans = [tn for tn in asks if rec.get(tn, {}).get("n") == 9 and (rec[tn].get("debug") or {}).get("replan") is True]; assert len(replans) == js["n_replans"]
        replans = [tn for tn in replans if (tn - ho) / 1e6 <= max(want) + 1e-6]
        got = asyncio.run(C.replay_full(asl, want_ticks=set(replans), collect_all=False)); uuid = None; worst = 0.0; nmis = 0; saved = []
        for tn in replans:
            snap = got["snaps"][tn]; view = H.build_view(snap, None); uuid = view.session_uuid; planner._cadence_checked.clear()
            pr = planner.plan(view, time_now_us=tn, time_query_us=snap["time_query_us"]); o = cap.last; d = pr.debug; lg = rec[tn]["debug"]; k = asks.index(tn); P = np.asarray(o["proposals"])[0]
            st = np.load(TRIAL / f"d016_{sc}" / f"replan_{k:03d}.npz", allow_pickle=True); dev = float(np.abs(P - st["proposals"]).max()); worst = max(worst, dev); ok = int(d["pick_with_momentum"]) == lg["chosen_index"] == int(st["chosen"]); nmis += (not ok)
            rel = (tn - ho) / 1e6
            if any(abs(rel - w) < 1e-6 for w in want):
                pickle.dump(dict(feat=cap.feat, proposals=P, S_plain=np.asarray(d["S"], dtype=np.float32), pdm_score=np.asarray(d["calibrated"], dtype=np.float32), chosen=int(d["pick_with_momentum"]), t_after_ho=rel, idx=k, tn=tn,
                                 feat_types={kk: type(v).__name__ for kk, v in cap.feat.items()}), open(out / f"replan_{k:03d}.pkl", "wb"), protocol=4); saved.append(rel)
        planner.on_session_close(uuid); summ.append(dict(scene=sc, replans_run=len(replans), saved=saved, pick_mismatches=nmis, worst_plan_dev_vs_trial_m=worst)); print(summ[-1], flush=True)
    model.close(); json.dump(summ, open(HERE / "cache" / "probe_inputs.json", "w"), indent=1); print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
