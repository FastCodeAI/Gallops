#!/usr/bin/env python
"""S5 STEP 1 proof, on recorded 0.1.4b runs (read-only), CPU.

Pass A  "only the weights and the two zeroings changed": the 0.1.6 PLANNER (adapters/drivejepa016/planner.py) with its three 0.1.6
        switches turned off in this process (BONNET_MASK_ROWS = 0, ZERO_ACCELERATION = False, ZERO_MOMENTUM_ACCELERATION = False) and
        the SHIPPED weights behind it (0.1.4b's worker: drivejepa014b.bridge; momentum.py is byte-identical in both packages, asserted)
        replays every replan of the three runs in time order (method of baseline_014b/b014b_menus.py) and must reproduce the recorded
        0.1.4b pick, the pick without momentum, the momentum code and the plan (<= 1 mm) on each.
Pass B  "images in are bit-identical to the S4 training loader": for one tick per scene the frames the planner used go through
        (1) 0.1.6's build_features with the switches ON and (2) the training converter's own composition (s1_convert.sample: the 0.1.3
        base planner's _pipeline_a, then the converter's mask line; then the S4 loader's slim PNG round trip and accel zeroing).

  cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python analysis_3.0/Retrain-data/07_closed_loop_test/step1_proof.py
"""
from __future__ import annotations
import asyncio, glob, hashlib, json, pathlib, sys, time
import numpy as np, cv2, torch

HERE = pathlib.Path(__file__).resolve().parent; ALP = HERE.parents[2]; RUNS = pathlib.Path("{runs}/alpasim_bulk/baseline_014b/runs")
sys.path.insert(0, str(ALP.parent / "analysis" / "drivejepa" / "mode1")); import mode1_common as C  # noqa: E402
import dj014_accel_counterfactual as H  # noqa: E402
sys.path.append(str(ALP / "adapters"))
import drivejepa016.planner as P16, drivejepa016.constants as K16  # noqa: E402
from drivejepa014b.bridge import BridgedDriveJEPA  # noqa: E402   shipped weights
sys.path.insert(0, str(HERE.parent / "04_training_set")); import s3_common as C3; import s1_common as C1; C1.NAS = C3.S1ROOT; import s1_convert as S  # noqa: E402
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
A = ALP / "adapters"; assert sha(A / "drivejepa016/momentum.py") == sha(A / "drivejepa014b/momentum.py") and sha(A / "drivejepa016/divider.py") == sha(A / "drivejepa014b/divider.py")
SCENES = {"02e075b9": 2.0, "16b83346": 5.0, "04749bb9": 0.0}          # scene -> the tick (seconds after handover) reported in the table
assert (K16.ADAPTER_VERSION, K16.BONNET_MASK_ROWS, K16.ZERO_ACCELERATION, K16.ZERO_MOMENTUM_ACCELERATION) == ("0.1.6", 37, True, True)
def switches(on): K16.BONNET_MASK_ROWS = 37 if on else 0; K16.ZERO_ACCELERATION = on; K16.ZERO_MOMENTUM_ACCELERATION = on

t0 = time.time(); model = BridgedDriveJEPA(device="cpu", stderr_log=HERE / "logs" / "step1_worker.log"); print(f"shipped-weights worker up {time.time()-t0:.0f}s", flush=True)
class Capture:
    def __init__(self, fn): self.fn = fn; self.last = None; self.feat = None
    def __call__(self, f): self.feat = f; self.last = self.fn(f); return self.last
cap = Capture(model); planner = P16.DriveJEPAPlanner(model_fn=cap, front_camera_id=C.FRONT, expected_frame_interval_us=50_000, momentum="on")
frames = {}; _bf = planner.build_features
def bf(observations, older, newer): frames["last"] = (observations, older, newer); return _bf(observations, older, newer)
planner.build_features = bf
rows = []; picked = {}
for scene, t_pick in SCENES.items():
    run = RUNS / f"b014b_{scene}_on"; (asl,) = glob.glob(str(run / "rollouts" / "clipgt-*" / "*" / "rollout.asl"))
    idx = asyncio.run(C.replay_full(asl, want_ticks=None, collect_all=True)); asks = idx["asks"]; rec = idx["recorded"]; ho = asks[0] + 3_000_000
    replans = [tn for tn in asks if rec.get(tn, {}).get("n") == 9 and (rec[tn].get("debug") or {}).get("replan") is True]
    got = asyncio.run(C.replay_full(asl, want_ticks=set(replans), collect_all=False)); n_ok = 0; worst = 0.0; uuid = None; switches(False)
    for tn in replans:
        snap = got["snaps"][tn]; view = H.build_view(snap, None); uuid = view.session_uuid; planner._cadence_checked.clear()
        pr = planner.plan(view, time_now_us=tn, time_query_us=snap["time_query_us"]); o = cap.last; d = pr.debug; lg = rec[tn]["debug"]; P = np.asarray(o["proposals"])[0]
        chosen, plain = int(d["pick_with_momentum"]), int(d["pick_without_momentum"]); rig, _ = H.recorded_rig(rec[tn]); dev = float(np.max(np.abs(P[chosen][:, :2] - rig[:, :2]))); worst = max(worst, dev)
        ok = chosen == lg["chosen_index"] == lg["pick_with_momentum"] and plain == lg["pick_without_momentum"] and d["momentum_code"] == lg["momentum_code"] and dev <= 1e-3; n_ok += ok
        t = (tn - ho) / 1e6
        if abs(t - t_pick) < 0.26 and scene not in picked:
            picked[scene] = dict(scene=scene, t_after_handover_s=round(t, 2), tick_us=tn, recorded_pick=int(lg["chosen_index"]), replayed_pick=chosen, recorded_pick_without_momentum=int(lg["pick_without_momentum"]), replayed_pick_without_momentum=plain,
                                 momentum_code=d["momentum_code"], plan_max_abs_diff_m=dev, same_pick=bool(ok), frames=frames["last"])
    planner.on_session_close(uuid); rows.append(dict(scene=scene, replans=len(replans), reproduced=int(n_ok), worst_plan_diff_m=worst)); print(rows[-1], flush=True)
model.close()

# ---------------- Pass B: images and ego status, switches ON
switches(True); out = []
for scene, p in picked.items():
    view, older, newer = p.pop("frames"); feats = _bf(view, older, newer); switches(False); raw = _bf(view, older, newer); switches(True)
    res = {}
    for key, fr in (("camera_feature_1", newer), ("camera_feature_2", older)):
        rgb = P16._decode_rgb(fr[1]); cf = S.P._pipeline_a(rgb).copy(); cf[:, -S.BONNET_MASK_ROWS:, :] = 0.0; cf = cf.astype(np.float32)            # s1_convert.sample: planner pipeline, then the mask line
        u8 = (torch.from_numpy(cf) * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy(); ok_, enc = cv2.imencode(".png", u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])       # dataset_s35._slim_write / _slim_decode
        loader = torch.from_numpy(np.ascontiguousarray(cv2.imdecode(enc, cv2.IMREAD_UNCHANGED))).permute(2, 0, 1).to(torch.float32).div(255.0).numpy()
        a = feats[key][0]; res[key] = bool(a.dtype == loader.dtype and a.shape == loader.shape and np.array_equal(a, loader)); res[key + "_masked_rows_max"] = float(a[:, -37:, :].max())
    es, es_raw = feats["ego_status"][0], raw["ego_status"][0]; z = es_raw.copy(); z[..., 5:7] = 0.0                                                    # dataset_s35: ego_status[..., 5:7] = 0
    res.update(ego_status_equals_loader_rule=bool(np.array_equal(es, z)), accel_reported=[round(float(v), 3) for v in es_raw[-1, 5:7]], accel_fed_to_model=[float(v) for v in es[-1, 5:7]],
               other_feature_keys_unchanged=bool(all(np.array_equal(feats[k], raw[k]) for k in ("lidar2img", "img_shape", "camera_feature"))))
    req = planner._momentum_request(view, feats, int(newer[0])); switches(False); req_off = planner._momentum_request(view, raw, int(newer[0])); switches(True)
    res.update(momentum_accel_016=[float(v) for v in req["ego_acceleration"]], momentum_accel_014b_rule=[round(float(v), 3) for v in req_off["ego_acceleration"]], momentum_velocity_unchanged=bool(np.array_equal(req["ego_velocity"], req_off["ego_velocity"])))
    out.append({**p, **res}); print(json.dumps(out[-1]), flush=True)
ok = all(r["reproduced"] == r["replans"] for r in rows) and all(o["same_pick"] and o["camera_feature_1"] and o["camera_feature_2"] and o["ego_status_equals_loader_rule"] and o["momentum_accel_016"] == [0.0, 0.0] for o in out)
json.dump(dict(pass_A_all_replans=rows, three_ticks=out, ok=bool(ok), momentum_py_sha256=sha(A / "drivejepa016/momentum.py"), checkpoint_shipped="drivejepa014b worker (real_model.DEFAULT_CHECKPOINT)"), open(HERE / "tables" / "step1_proof.json", "w"), indent=1)
print("STEP 1 PROOF", "PASS" if ok else "FAIL")
