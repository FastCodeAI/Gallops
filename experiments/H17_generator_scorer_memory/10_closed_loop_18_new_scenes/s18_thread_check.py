#!/usr/bin/env python
"""18-scene closed loop, before any scene: does pinning the driver to 2 cores (taskset -c 0-1) change what the model outputs?
The 12-scene runs ran the CPU worker unpinned (torch uses as many threads as it may run on); the 18-scene runs are pinned to 2 cores.
A different thread count can change the order of float sums and so the numbers. Both workers are fed fixed recorded inputs:
  p06     the 0.1.8b-p06 worker, the 64 inputs of its equivalence test (single calls, no comfort re-score)
  carry   the memory worker in carry mode (the real checkpoint), picking clip 0238dfe4 as one session (s12_adapter_proofs.clip_inputs)
Usage:  s18_thread_check.py <tag>   -> out/cl18_thread_<tag>.npz (run once unpinned as tag "all", once under taskset -c 0-1 as "two")
        s18_thread_check.py compare -> out/cl18_thread_check.json"""
import os, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE.parent / "out"
sys.path.insert(0, str(HERE))
if sys.argv[1] == "compare":
    import json
    a, b = np.load(OUT / "cl18_thread_all.npz"), np.load(OUT / "cl18_thread_two.npz")
    res = {"cores": {"all": int(a["cores"]), "two": int(b["cores"])}, "secs_per_call": {"all": a["secs"].tolist(), "two": b["secs"].tolist()}}
    for k in a.files:
        if k in ("cores", "secs"):
            continue
        x, y = a[k], b[k]
        r = {"bit_identical": bool(np.array_equal(x, y)), "max_abs_diff": float(np.abs(x - y).max())}
        if k.endswith("pdm_score"):
            r["picks_equal"] = int((x.argmax(-1) == y.argmax(-1)).sum()); r["n"] = int(x.shape[0])
        if k.endswith("proposals"):
            r["plan_xy_max_abs_diff_m"] = float(np.abs(x[..., :2] - y[..., :2]).max())
        res[k] = r
    json.dump(res, open(OUT / "cl18_thread_check.json", "w"), indent=1); print(json.dumps(res, indent=1)); sys.exit(0)

import s12_adapter_proofs as P                                   # sets the env checks (CUDA hidden, batch fix on)
tag = sys.argv[1]; out = {"cores": len(os.sched_getaffinity(0))}; secs = []
toks, feats = P.gate_inputs()
w = P.worker("drivejepa018bp06", f"cl18_thread_{tag}_p06_worker.log"); R = {"proposals": [], "pdm_score": []}; t0 = time.time()
for f in feats:
    o = w(dict(f)); R["proposals"].append(np.asarray(o["proposals"])[0]); R["pdm_score"].append(np.asarray(o["pdm_score"])[0])
w.close(); secs.append((time.time() - t0) / len(feats))
out.update({f"p06_{k}": np.stack(v) for k, v in R.items()})
sids, cfeats, poses, _ = P.clip_inputs()
w = P.worker("drivejepamemory", f"cl18_thread_{tag}_carry_worker.log", P.CL12 / "model/carry"); w.memory_control("start", "T")
R = {"proposals": [], "pdm_score": []}; t0 = time.time()
for n, f in enumerate(cfeats):
    o = w({**f, "memory": P.mem_req("T", n, poses[n])}); R["proposals"].append(np.asarray(o["proposals"])[0]); R["pdm_score"].append(np.asarray(o["pdm_score"])[0])
w.close(); secs.append((time.time() - t0) / len(cfeats))
out.update({f"carry_{k}": np.stack(v) for k, v in R.items()}); out["secs"] = np.array(secs)
np.savez(OUT / f"cl18_thread_{tag}.npz", **out); print(tag, "cores", out["cores"], "secs/call", secs)
