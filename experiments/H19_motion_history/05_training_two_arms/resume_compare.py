"""TRAIN step 1: compare the resume test's two runs (6 straight vs 3 + save + resume + 3): every per-step record, final weights, Adam
states and RNG states must be identical. Output MH/out/resume_test.json."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(sys.argv[1])
a = torch.load(R / "straight" / "final.pt", map_location="cpu", weights_only=False)
b = torch.load(R / "split" / "final.pt", map_location="cpu", weights_only=False)
res = {"per_step": {}, "weights": {}, "optimisers": [], "rng": {}}
ra, rb = a["rec"], b["rec"]
for k in list(ra["losses"]) :
    res["per_step"][f"loss.{k}"] = {"straight": ra["losses"][k], "split": rb["losses"][k], "identical": ra["losses"][k] == rb["losses"][k]}
for k in ("zeroed_targets", "pull_rows", "pull_map_rows", "map_rows", "hidden_rows"):
    res["per_step"][k] = {"identical": ra[k] == rb[k], "straight": ra[k], "split": rb[k]}
n_t = n_eq = 0; maxd = 0.0
for grp in ("generator", "scorer", "new"):
    for n, t in a[grp].items():
        n_t += 1; e = torch.equal(t, b[grp][n]); n_eq += int(e)
        maxd = max(maxd, float((t.float() - b[grp][n].float()).abs().max()))
res["weights"] = {"tensors": n_t, "identical": n_eq, "max_abs_diff": maxd, "hide_gen_state_identical": bool(torch.equal(a["hide_gen_state"], b["hide_gen_state"]))}
for oa, ob in zip(a["optimisers"], b["optimisers"]):
    ok = True
    for i, s in oa["state"].items():
        for kk, v in s.items():
            ok &= bool(torch.equal(v, ob["state"][i][kk])) if torch.is_tensor(v) else v == ob["state"][i][kk]
    res["optimisers"].append(ok)
ga, gb = a["rng"], b["rng"]
res["rng"] = {"torch": bool(torch.equal(ga["torch"], gb["torch"])), "cuda": all(torch.equal(x, y) for x, y in zip(ga["cuda"], gb["cuda"])),
              "numpy": bool(all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(ga["numpy"], gb["numpy"]))),
              "python": ga["python"] == gb["python"]}
res["resumes_in_split"] = rb.get("resumes")
res["identical"] = bool(all(v["identical"] for v in res["per_step"].values()) and n_eq == n_t and res["weights"]["hide_gen_state_identical"]
                        and all(res["optimisers"]) and all(res["rng"].values()))
Path(sys.argv[2]).write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({"identical": res["identical"], "losses_straight": ra["losses"]["loss"], "losses_split": rb["losses"]["loss"],
                  "weights": res["weights"], "optimisers": res["optimisers"], "rng": res["rng"]}))
