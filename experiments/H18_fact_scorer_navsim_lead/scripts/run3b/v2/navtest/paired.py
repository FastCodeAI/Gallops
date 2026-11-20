"""Paired navtest comparison: per-scene human-filtered EPDMS (navtest_table.py's recombination), candidate minus each other run,
95 % bootstrap over scenes and over logs (log from metric_cache_v2/<log>/unknown/<token>), 2,000 resamples, seed 0;
better / same / worse counts (|diff| < 1e-9 = same). usage: paired.py cand=csv other=csv [other=csv ...]"""
import csv, math, os, sys
import numpy as np
sys.path.insert(0, "{runs}/run3a/build")
from navtest_table import SUMMARY, filtered  # noqa: E402

MC = "{data.paddock_dump}/navsim/metric_cache_v2"


def load(p):
    return {r["token"]: filtered(r) for r in csv.DictReader(open(p)) if r["token"] not in SUMMARY and r["valid"] == "True"}


log_of = {}
for lg in os.listdir(MC):
    d = os.path.join(MC, lg, "unknown")
    if os.path.isdir(d):
        for t in os.listdir(d):
            log_of[t] = lg
args = [a.split("=", 1) for a in sys.argv[1:]]
(cn, cp), others = args[0], args[1:]
cand = load(cp)
rng = np.random.default_rng(0)
print(f"{cn} minus:  n  mean_diff  95% scenes  95% logs (n_logs)  better/same/worse")
for on, op in others:
    oth = load(op)
    toks = sorted(set(cand) & set(oth))
    d = np.array([cand[t] - oth[t] for t in toks]) * 100
    B = 2000
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    logs = sorted({log_of[t] for t in toks})
    li = {l: i for i, l in enumerate(logs)}
    lid = np.array([li[log_of[t]] for t in toks])
    sums, cnts = np.bincount(lid, d, len(logs)), np.bincount(lid, None, len(logs))
    bl = []
    for _ in range(B):
        k = rng.integers(0, len(logs), len(logs))
        bl.append(sums[k].sum() / cnts[k].sum())
    bl = np.array(bl)
    q = lambda x: f"[{np.percentile(x, 2.5):+.2f}, {np.percentile(x, 97.5):+.2f}]"  # noqa: E731
    print(f"  {on:<14} {len(toks)} {d.mean():+.2f}  {q(bs)}  {q(bl)} ({len(logs)})  "
          f"{(d > 1e-7).sum()}/{(np.abs(d) <= 1e-7).sum()}/{(d < -1e-7).sum()}")
