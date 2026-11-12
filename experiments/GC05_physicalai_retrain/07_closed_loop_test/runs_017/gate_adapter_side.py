#!/usr/bin/env python
"""0.1.7 build gate, adapter side (alpasim venv): the SAME worker the driver starts (runs_017/adapters/drivejepa017/bridge.py, conda
drive-jepa python, torch 2.1, upstream agent class, merged checkpoint verified by size and sha256, batch fix installed) is fed the
feature dicts saved by gate_train_side.py.
  A  fix on,  batch 1: 16 forwards            -> against the training-side plans (gate: largest difference under 2 cm)
  B  fix off, batch 1: 16 forwards            -> against A (Sid's gate says "differ"; S10 measured 0.0 at batch 1, so this is reported as measured)
  C  fix on,  batch 2: 8 forwards of 2 rows   -> against A (the fix makes a row independent of its batch neighbour)
  D  fix off, batch 2: 8 forwards of 2 rows   -> against A and C (without the fix the rows leak: this is the proof the switch is live)
Writes OUT/gate_017.json. CPU only."""
import os, sys, json, time, pathlib
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; OUT = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(HERE / "adapters"))
from drivejepa017.bridge import BridgedDriveJEPA
import drivejepa017.constants as K
z = np.load(OUT / "train_side.npz"); T = z["plans"]; toks = [str(t) for t in z["tokens"]]; assert str(z["ckpt_sha256"]) == K.CHECKPOINT_SHA256
keys = sorted({k.split("__", 1)[1] for k in z.files if k.startswith("f00__")}); feats = [{k: z[f"f{n:02d}__{k}"] for k in keys} for n in range(16)]
def run(fix, bs):
    os.environ[K.BATCH_FIX_ENV] = "on" if fix else "off"; log = OUT / f"worker_fix_{'on' if fix else 'off'}.log"; t0 = time.time()
    m = BridgedDriveJEPA(device="cpu", stderr_log=log); res = {}
    for b in bs:
        P = []
        for i in range(0, 16, b):
            o = m({k: np.concatenate([feats[j][k] for j in range(i, i + b)]) for k in keys}); P.append(np.asarray(o["proposals"], dtype=np.float32))
        res[b] = np.concatenate(P); assert res[b].shape == T.shape
    m.close(); line = [l for l in open(log, errors="replace") if "batch_fix: installed" in l][-1].strip(); print(f"fix {fix}: {time.time()-t0:.0f}s | {line}", flush=True); return res, line
on, line_on = run(True, (1, 2)); off, line_off = run(False, (1, 2))
def d(a, b):
    xy = np.abs(a[..., :2] - b[..., :2]).reshape(16, -1).max(1); return dict(plan_xy_max_abs_diff_m=float(xy.max()), plan_xy_median_over_examples_m=float(np.median(xy)), heading_max_abs_diff_rad=float(np.abs(a[..., 2] - b[..., 2]).max()), examples_above_2cm=int((xy > 0.02).sum()), per_example_xy_max_m=[round(float(v), 6) for v in xy])
res = dict(checkpoint=K.CHECKPOINT_PATH, checkpoint_sha256=K.CHECKPOINT_SHA256, batch_fix_sha256=K.BATCH_FIX_SHA256, adapter_version=K.ADAPTER_VERSION, examples=16, tokens=toks, slices=[str(s) for s in z["slices"]],
           training_side=f"s35_common.build_agent + load_ckpt, torch {z['torch']}, CPU float32, batch 1, batch fix on", adapter_side="drivejepa017 bridge worker (conda drive-jepa python), CPU float32",
           worker_line_fix_on=line_on, worker_line_fix_off=line_off,
           A_adapter_fix_on_b1_vs_training_side=d(on[1], T), B_adapter_fix_off_b1_vs_fix_on_b1=d(off[1], on[1]), C_adapter_fix_on_b2_vs_fix_on_b1=d(on[2], on[1]), D_adapter_fix_off_b2_vs_fix_on_b1=d(off[2], on[1]),
           D2_adapter_fix_off_b2_vs_fix_on_b2=d(off[2], on[2]))
res["gate_1_under_2cm"] = res["A_adapter_fix_on_b1_vs_training_side"]["plan_xy_max_abs_diff_m"] < 0.02 and res["A_adapter_fix_on_b1_vs_training_side"]["heading_max_abs_diff_rad"] < 0.02
res["gate_2_fix_off_differs_at_batch_1"] = res["B_adapter_fix_off_b1_vs_fix_on_b1"]["plan_xy_max_abs_diff_m"] > 0
res["fix_live_at_batch_2"] = res["D2_adapter_fix_off_b2_vs_fix_on_b2"]["plan_xy_max_abs_diff_m"] > 0.02 and res["C_adapter_fix_on_b2_vs_fix_on_b1"]["plan_xy_max_abs_diff_m"] < 0.02
json.dump(res, open(OUT / "gate_017.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "tokens"}, indent=1))
