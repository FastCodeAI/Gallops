"""Bench2Drive 220, one result per route: Run 3B v2 (run3b_v2_b2d) and Run 3A (wajepa_dj_run3a_b2d), both run in one
setup (worktree gate-oct27 = paddock 3daf4b9 + ops adapters, Bench2Drive copy b2d-fix, command fix, 2 Hz), against
released Drive-JEPA's committed paddock record (results/bench2drive/drivejepa/pooled220-86aa45e768a29a64-1507a1, its
22 source runs). Per route: the last attempt whose status is not a crash (sorted by run directory), else the last.
Also Run 3A's own native-50 run of 26 Oct (worktree rerank-native, model wajepa_dj_run3a, another Bench2Drive copy)
against the same 50 routes in the 220's setup. The native-50 routes: released Drive-JEPA's native-50 record's routes_subset (dj50_run.json, copied beside this file).
Paired by route: mean difference, t = mean / (sd / sqrt(n)), higher / lower / tied (|diff| < 1e-9 = tied).
Writes b2d220_paired.json beside this file. usage: paired_b2d220.py"""
import ast, collections, glob, json, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
RUN3B = "{data.p3_renesas}/b2d220_run3b_cmdfix/bench2drive/run3b_v2_b2d"
RUN3A = "{data.p3_renesas}/b2d220_run3a_cmdfix/bench2drive/wajepa_dj_run3a_b2d"
RUN3A_26OCT = "{data.p3_renesas}/paddock-runs/run3a_cmdfix/repa/bench2drive/wajepa_dj_run3a"  # worktree rerank-native
DJ_RUNS = "{data.paddock_dump}/b2d220/bench2drive/drivejepa"
DJ_RECORD = os.path.join(HERE, "dj_pooled220_run.json")  # git show origin/main:results/.../run.json (paddock)
L = lambda v: v if isinstance(v, list) else ast.literal_eval(v)


def pick(files, subset=None):
    per = collections.defaultdict(list)
    for f in sorted(files):
        m = re.match(r"(\d+)-", f.split("/")[-2])
        if not m or (subset and m.group(1) not in subset):
            continue
        try:
            r = json.load(open(f))["_checkpoint"]["records"][0]
        except Exception:
            continue
        per[m.group(1)].append(r)
    out = {}
    for rid, lst in per.items():
        ok = [r for r in lst if "crash" not in str(r.get("status", "")).lower()]
        r = (ok or lst)[-1]
        inf = {k: (len(v) if isinstance(v, list) else v) for k, v in r.get("infractions", {}).items() if v}
        out[rid] = {"ds": r["scores"]["score_composed"], "rc": r["scores"]["score_route"], "status": r["status"],
                    "infractions": inf, "attempts": len(lst)}
    return out


cfg = json.load(open(DJ_RECORD))["config"]
subset = set(L(cfg["routes_subset"]))
dj_files = [f for run in L(cfg["source_runs"]) for f in glob.glob(f"{DJ_RUNS}/{run}/*/results.json")]
M = {"run3b": pick(glob.glob(RUN3B + "/*/*/results.json")), "run3a": pick(glob.glob(RUN3A + "/*/*/results.json")),
     "released_drivejepa": pick(dj_files, subset)}
ids = sorted(M["run3b"])
assert len(ids) == 220 and all(set(m) == set(ids) for m in M.values()), {k: len(v) for k, v in M.items()}
# scenario name from the route directory (<id>-<Town>-<Scenario>), same for every model
scen = {}
for f in glob.glob(RUN3B + "/*/*/results.json"):
    parts = f.split("/")[-2].split("-")
    scen[parts[0]] = parts[-1]
native50 = sorted(L(json.load(open(os.path.join(HERE, "dj50_run.json")))["config"]["routes_subset"]))


def summary(m, sel):
    n = len(sel)
    tot = collections.Counter()
    for i in sel:
        tot.update(m[i]["infractions"])
    return {"n": n, "driving_score": round(sum(m[i]["ds"] for i in sel) / n, 2),
            "route_completion": round(sum(m[i]["rc"] for i in sel) / n, 2),
            "status": dict(collections.Counter(m[i]["status"] for i in sel).most_common()), "infractions": dict(tot)}


def paired(a, b, sel):
    d = [a[i]["ds"] - b[i]["ds"] for i in sel]
    n = len(d); mu = sum(d) / n; sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    return {"n": n, "diff": round(mu, 2), "t": round(mu / (sd / math.sqrt(n)), 2), "higher": sum(x > 1e-9 for x in d),
            "lower": sum(x < -1e-9 for x in d), "tied": sum(abs(x) <= 1e-9 for x in d)}


oct26 = pick(glob.glob(RUN3A_26OCT + "/*/*/results.json"))
assert sorted(oct26) == native50, len(oct26)
res = {"routes": ids, "summary": {k: summary(m, ids) for k, m in M.items()},
       "summary_native50": {k: summary(m, native50) for k, m in M.items()},
       "paired": {"run3b-run3a": paired(M["run3b"], M["run3a"], ids),
                  "run3b-run3a_native50": paired(M["run3b"], M["run3a"], native50),
                  "run3b-released": paired(M["run3b"], M["released_drivejepa"], ids),
                  "run3a-released": paired(M["run3a"], M["released_drivejepa"], ids),
                  "run3a-run3a_26oct_native50": paired(M["run3a"], oct26, native50)},
       "run3a_26oct_native50": summary(oct26, native50),
       "by_scenario": {}, "per_route": {i: {k: M[k][i] for k in M} for i in ids}}
g = collections.defaultdict(list)
for i in ids:
    g[scen[i]].append(i)
for s, sel in sorted(g.items()):
    res["by_scenario"][s] = {"n": len(sel), **{k: round(sum(M[k][i]["ds"] for i in sel) / len(sel), 1) for k in M}}
json.dump(res, open(os.path.join(HERE, "b2d220_paired.json"), "w"), indent=1)
for k, v in res["summary"].items():
    print(k, v["n"], v["driving_score"], v["route_completion"], v["status"])
for k, v in res["paired"].items():
    print(k, v)
