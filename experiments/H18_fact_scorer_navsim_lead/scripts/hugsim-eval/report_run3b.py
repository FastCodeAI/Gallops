"""HUGSIM results report with Run 3B v2 (20 Nov; copy of report_run3a.py, 26 Oct, Run 3B added as the first column and
the paired differences taken Run 3B minus each model). Was: HUGSIM results report with Run 3A (26 Oct; copy of report.py, 22 Oct): Run 3A, released Drive-JEPA, the Final, AutoE2E
on the runnable scenarios, paired. Adds paired Run 3A differences with a bootstrap 95% CI (scenario resampling, seed 0).

Scores (x100): HD-Score, PDMS, RC and the PDMS terms, overall / by dataset / by difficulty, on the scenarios every
model finished. How each episode ended, from the simulator's own termination message in the run's log (HUGSimEnv.step prints
exactly one): "Complete" (RC >= 1), "Collision with foreground" (hit a car), "Collision with background" (hit the
scenery), "Far from preset trajectory" (> 10 m off the recorded path); none of them and 400 steps = the time cap.
(22 Oct fix: v1 matched the scorer's lowercase "collision with ..." lines, which it prints for any PLANNED trajectory
that would collide, so plan-level hits were counted as crashes. v1 kept as report_final.v1_endings_wrong.json.)
Excluded for every model: HUGSIM release bugs -- 15 PandaSet scenarios with ConstantPlanner(max_t=...) and waymo
scene-140045460035-hard-00 (start_velo: 1R).
usage: python report_run3b.py [out.json]
"""
import json
import os
import subprocess
import sys

import numpy as np
from collections import Counter, defaultdict

R = "{data.p3_renesas}/hugsim-eval"
MODELS = {"Run 3B": "run3b_v2", "Run 3A": "run3a", "Released Drive-JEPA": "dj_released", "The Final": "final", "AutoE2E": "autoe2e"}
M = ("hdscore", "pdms", "rc", "nc", "dac", "ttc", "c")
bad = {p.split("/")[-1][:-5] for p in subprocess.run("grep -l max_t {data.hugsim_home}/data/scenarios/*/*.yaml",
                                                     shell=True, capture_output=True, text=True).stdout.split()}
bad |= {"scene-140045460035-hard-00"}
# * (user, 26 Oct: use the rear camera where there is one, mark the rest): HUGSIM has no rear data for these two and
# renders CAM_BACK black, so Run 3A drives without its back view there (the frame it gets is black).
NO_REAR = {"kitti360", "waymo"}


def ending(r, label):
    log = f"{R}/{label}/logs/{r['scenario']}.log"
    lines = {x.strip() for x in open(log, errors="ignore")} if os.path.exists(log) else set()
    if "Complete" in lines:
        return "completed route"
    if "Collision with background" in lines:
        return "hit scenery"
    if "Collision with foreground" in lines:
        return "hit a car"
    if "Far from preset trajectory" in lines:
        return "> 10 m off route"
    if (r.get("steps") or 0) >= 400:
        return "400-step cap"
    return "other"


runs = {}
for name, label in MODELS.items():
    rows = [json.loads(l) for l in open(f"{R}/{label}/results.jsonl")]
    runs[name] = {r["scenario"]: r for r in rows if "hdscore" in r and r["scenario"] not in bad}
common = sorted(set.intersection(*(set(v) for v in runs.values())))
out = {"n_paired": len(common), "per_model_n": {k: len(v) for k, v in runs.items()}, "excluded": sorted(bad)}
print(f"paired scenarios: {len(common)} | per model: {out['per_model_n']} | excluded (HUGSIM release bugs): {len(bad)}\n")


def mean(name, ss, m):
    return 100 * sum(runs[name][s][m] for s in ss) / len(ss)


hdr = f"{'':18s}" + "".join(f"{n:>21s}" for n in MODELS)
print("ALL (x100)\n" + hdr)
out["all"] = {}
for m in M:
    vals = {n: round(mean(n, common, m), 1) for n in MODELS}
    out["all"][m] = vals
    print(f"{m:18s}" + "".join(f"{vals[n]:22.1f}" for n in MODELS))
for key, title in (("dataset", "HD-Score by dataset"), ("difficulty", "HD-Score by difficulty")):
    groups = defaultdict(list)
    for s in common:
        groups[runs["The Final"][s][key]].append(s)
    print(f"\n{title}\n" + hdr)
    out[key] = {}
    for g in sorted(groups):
        vals = {n: round(mean(n, groups[g], "hdscore"), 1) for n in MODELS}
        out[key][g] = {"n": len(groups[g]), **vals}
        star = "*" if g in NO_REAR else ""
        if star:
            out[key][g]["no_rear_camera"] = True
        print(f"{g + star + ' (' + str(len(groups[g])) + ')':18s}" + "".join(f"{vals[n]:22.1f}" for n in MODELS))
print("\nHow episodes ended (% of paired scenarios)\n" + hdr)
ends = {n: Counter(ending(runs[n][s], MODELS[n]) for s in common) for n in MODELS}
out["endings"] = {}
for e in ("completed route", "hit a car", "hit scenery", "> 10 m off route", "400-step cap", "other"):
    vals = {n: round(100 * ends[n][e] / len(common), 1) for n in MODELS}
    out["endings"][e] = vals
    print(f"{e:18s}" + "".join(f"{vals[n]:22.1f}" for n in MODELS))
print("\nRun 3B minus each model, paired (x100, bootstrap 95% CI)")
rng = np.random.default_rng(0)
out["diff"] = {}
for n in list(MODELS)[1:]:
    out["diff"][n] = {}
    for m in ("hdscore", "pdms", "rc"):
        d = 100 * np.array([runs["Run 3B"][s][m] - runs[n][s][m] for s in common])
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        w, l = int((d > 0.5).sum()), int((d < -0.5).sum())
        out["diff"][n][m] = {"mean": round(d.mean(), 1), "ci": [round(lo, 1), round(hi, 1)], "better": w, "worse": l,
                             "same": len(d) - w - l}
        print(f"  vs {n:20s} {m:8s} {d.mean():+6.1f} [{lo:+.1f}, {hi:+.1f}]  better {w} / same {len(d) - w - l} / worse {l}")
print("\n* no rear camera on this dataset: HUGSIM renders CAM_BACK black, so Run 3A has no back view there")
out["note"] = "* kitti360, waymo: no rear camera (HUGSIM renders CAM_BACK black); Run 3A has no back view there"
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print("\nwritten", sys.argv[1])
