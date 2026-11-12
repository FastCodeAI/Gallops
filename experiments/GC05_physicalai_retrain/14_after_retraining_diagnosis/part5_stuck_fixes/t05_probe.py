#!/usr/bin/env python
"""T05 Part 2, step 3: CPU forward passes of the retrained model (adapter 0.1.6 class MomentumDriveJEPA, pass-9 weights) on the saved
inputs of the probe ticks, with the waypoints of ONE slot replaced by an injected path. Nothing under adapters/ is edited: the shared
Traj_refiner instance gets an instance-level forward that repeats traj_refiner.py:24-29 and overwrites one slot's decoded waypoints
  mode "last": in the 4th (last) refinement pass only -- the waypoints the scorer's features are sampled at and that are returned;
  mode "all":  in all 4 passes -- the slot's feature is refined around the injected path from the first pass on.
The image backbone output is computed once per tick and reused (same inputs). Run as the bridge runs its worker:
  cd adapters && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD conda-envs/drive-jepa/bin/python <this file>"""
import os, sys, json, glob, pickle, pathlib, time
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
import numpy as np
import numpy.core as _nc  # the inputs were pickled under numpy 2 (alpasim venv); this env has numpy 1
for _n in ('multiarray', 'numeric', '_multiarray_umath'): sys.modules['numpy._core.' + _n] = getattr(_nc, _n)
sys.modules['numpy._core'] = _nc
HERE = pathlib.Path(__file__).resolve().parent; TMP = pathlib.Path("{runs}/drivejepa_retrain/t05_scratch_20261009")
from drivejepa016 import constants as K
from drivejepa016.momentum import MomentumDriveJEPA
import torch; torch.set_num_threads(6)
model = MomentumDriveJEPA(checkpoint_path=K.CHECKPOINT_PATH, device="cpu"); m = model._agent._pad_model; ref = m._trajectory_head[0]; assert all(r is ref for r in m._trajectory_head) and len(m._trajectory_head) == 4
ST = dict(n=0, slot=None, path=None, mode=None, key=None); CACHE = {}
def fwd(bev, imf):
    props = ref.traj_decoder(bev).reshape(bev.shape[0], -1, ref.poses_num, ref.state_size); ST["n"] += 1
    if ST["slot"] is not None and (ST["mode"] == "all" or ST["n"] == 4):
        props = props.clone(); props[0, ST["slot"]] = torch.as_tensor(ST["path"], dtype=props.dtype)
    return ref.Bev_refiner(props, bev, imf), props
ref.forward = fwd
bb = m._backbone.forward
def backbone(*a, **k):
    if ST["key"] not in CACHE: CACHE.clear(); CACHE[ST["key"]] = bb(*a, **k)
    return CACHE[ST["key"]]
m._backbone.forward = backbone

def run(feat, key, slot=None, path=None, mode=None):
    ST.update(n=0, slot=slot, path=path, mode=mode, key=key); r = model({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in feat.items()}); assert ST["n"] == 4
    return np.asarray(r["proposals"])[0], np.asarray(r["momentum_S"]).reshape(-1), np.asarray(r["pdm_score"])[0], (None if r.get("momentum_Sc") is None else np.asarray(r["momentum_Sc"]).reshape(-1)), int(r["momentum_meta"]["pick_with"])

out = []; gate = []; t0 = time.time()
for spec in json.load(open(HERE / "cache" / "probe_paths.json")):
    s = spec["scene"]; z = pickle.load(open(TMP / s / f"replan_{spec['idx']:03d}.pkl", "rb")); feat = z["feat"]; key = (s, spec["idx"])
    P, S, Sf, Sc, pick = run(feat, key)
    gate.append(dict(scene=s, t_after_handover_s=spec["t_after_handover_s"], max_abs_waypoint_diff_m=float(np.abs(P[..., :2] - z["proposals"][..., :2]).max()), max_abs_all_diff=float(np.abs(P - z["proposals"]).max()), max_abs_plain_score_diff=float(np.abs(S - z["S_plain"]).max()),
                     max_abs_final_score_diff=float(np.abs(Sf - z["pdm_score"]).max()), pick=pick, logged_pick=int(z["chosen"]), momentum_request_present=bool(feat.get(K.MOMENTUM_FEATURE_KEY) is not None)))
    print(gate[-1], flush=True)
    base = dict(scene=s, t_after_handover_s=spec["t_after_handover_s"])
    for i in range(32): out.append(dict(**base, path="control_no_replacement", mode="none", slot=i, slot_role="", score_plain=float(S[i]), score_final=float(Sf[i]), comfort_term=None if Sc is None else float(Sc[i]), slot_orig_plain=float(S[i]), slot_orig_final=float(Sf[i]), pick_after=pick, max_other_slot_plain_change=0.0))
    for role, slot in spec["slots"].items():
        for mode in ("last", "all"):
            for name, path in list(spec["paths"].items()) + [("self_own_waypoints", P[slot].tolist())]:
                P2, S2, Sf2, Sc2, pick2 = run(feat, key, slot, np.asarray(path, dtype=np.float32), mode); oth = np.delete(np.abs(S2 - S), slot).max()
                out.append(dict(**base, path=name, mode=mode, slot=slot, slot_role=role, score_plain=float(S2[slot]), score_final=float(Sf2[slot]), comfort_term=None if Sc2 is None else float(Sc2[slot]), slot_orig_plain=float(S[slot]), slot_orig_final=float(Sf[slot]), pick_after=pick2,
                                max_other_slot_plain_change=float(oth), returned_waypoint_dev_m=float(np.abs(P2[slot] - np.asarray(path)).max())))
    print(s, spec["t_after_handover_s"], "done", f"{time.time() - t0:.0f}s", flush=True)
json.dump(dict(gate=gate, rows=out), open(HERE / "cache" / "probe_raw.json", "w")); print("done")
