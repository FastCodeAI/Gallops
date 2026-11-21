#!/usr/bin/env python
"""CL20 (reports/CL20_PREREG.md): per-scene table, primary rule on the 20 new scenes, 50-scene secondary, not-deciding numbers.
Score = stepb.score (results-summary.json rollouts[0].score). At-fault = front/side collision INSIDE the score window = the eval's own windowed
collision_at_fault, via s19_fulldrive_rescore.one() (old_collision_at_fault; it asserts the windowed score reproduces the file to 1e-9).
stepb.exit_of (CL18's at-fault, first event over the whole drive) is reported beside it. Read-only on the runs. Writes out/cl20_table.{csv,json}.
    cd <alpasim> && nice -n 19 taskset -c 0,1 .venv/bin/python -B <this>"""
import importlib.util, json, pathlib, statistics as st, sys

HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE.parent / "out"
spec = importlib.util.spec_from_file_location("s19", HERE / "s19_fulldrive_rescore.py"); S19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S19)
G = S19.G; S5 = G / "s5_diagnosis/src"; sys.argv = sys.argv[:1]
src = (S5 / "stepb.py").read_text().split("\nrows = []")[0]
B = {"__file__": str(S5 / "stepb.py")}; exec(compile(src, str(S5 / "stepb.py"), "exec"), B)

N40 = "{local_store}/runs/new40"
S20 = [l.split()[0] for l in open(OUT / "cl20_scenes_20_v2.txt") if l.strip()]
assert len(S20) == 20 and not set(S20) & set(S19.S12 + S19.S18)
ROOTS = {  # the folders the pre-registration names
    "20": {"carry": f"{N40}/runs/memory_carry_{{}}", "018b": f"{N40}/018bp06/runs/drivejepa_018bp06_{{}}"},
    "12": {"carry": f"{S19.CL12}/runs/memory_carry_{{}}", "018b": S19.R018},
    "18": {"carry": f"{S19.CL18}/runs/memory_carry_{{}}", "018b": f"{S19.CL18}/018bp06/runs/drivejepa_018bp06_{{}}"}}
BLOCKS = {"20": S20, "12": S19.S12, "18": S19.S18}


def arm_rec(run):
    if not pathlib.Path(run, "aggregate/results-summary.json").is_file():
        return None
    r = S19.one("x", "x", "x", run)
    assert r["reproduces"], run                     # windowed re-aggregation must equal the run's own score
    score = B["score"](run)["score"]; assert abs(score - r["old_score"]) < 1e-9, run
    ex = B["exit_of"](run)
    return dict(run=run, score=score, atfault_window=bool(r["old_collision_at_fault"] and r["old_collision_at_fault"] > 0),
                offroad_window=bool(r["old_offroad"] and r["old_offroad"] > 0), exit=ex, atfault_exit=ex.startswith("at-fault"),
                window_end_rel=r["old_win_end_rel"], events=[(e["label"], round(e["rel"], 2), e["in_old_window"]) for e in r["events"]])


rows = []
for blk, scenes in BLOCKS.items():
    for s in scenes:
        row = dict(scene=s, block=blk)
        for arm, pat in ROOTS[blk].items():
            row[arm] = arm_rec(pat.format(s))
        rows.append(row)


def summary(R):
    R = [r for r in R if r["carry"] and r["018b"]]
    if not R:
        return dict(n=0)
    diff = {r["scene"]: r["carry"]["score"] - r["018b"]["score"] for r in R}
    af = {a: [r["scene"] for r in R if r[a]["atfault_window"]] for a in ("carry", "018b")}
    afx = {a: [r["scene"] for r in R if r[a]["atfault_exit"]] for a in ("carry", "018b")}
    top2 = sorted(diff, key=lambda s: -diff[s])[:2]
    return dict(n=len(R), mean={a: st.mean(r[a]["score"] for r in R) for a in ("carry", "018b")}, d=st.mean(diff.values()),
                wins=[s for s in diff if diff[s] > 0.1], losses=[s for s in diff if diff[s] < -0.1], ties=sum(abs(v) <= 0.1 for v in diff.values()),
                atfault_window=af, atfault_window_carry_not_018b=[s for s in af["carry"] if s not in af["018b"]],
                atfault_exit=afx, atfault_exit_carry_not_018b=[s for s in afx["carry"] if s not in afx["018b"]],
                both_perfect=sum(r["carry"]["score"] >= 1 - 1e-9 and r["018b"]["score"] >= 1 - 1e-9 for r in R),
                top2=top2, d_wo_top2=st.mean(v for s, v in diff.items() if s not in top2) if len(R) > 2 else None)


P = summary([r for r in rows if r["block"] == "20"])
missing = [(r["scene"], a) for r in rows if r["block"] == "20" for a in ("carry", "018b") if r[a] is None]
if missing:
    verdict = f"NO VERDICT: {len(missing)} of 40 drives missing"
else:
    W, L, added = len(P["wins"]), len(P["losses"]), P["atfault_window_carry_not_018b"]
    if P["d"] <= -0.05 or L > W or added:
        verdict = "AGAINST"
    elif P["d"] >= 0.05 and W > L and not added:
        verdict = "SUPPORTS memory"
    else:
        verdict = "NO DETECTABLE EFFECT"
res = dict(verdict=verdict, missing=missing, primary_20=P, all_50=summary(rows), old_30=summary([r for r in rows if r["block"] != "20"]),
           rule="SUPPORTS if d >= +0.05 AND wins > losses AND no at-fault (in window) crash carry has that 0.1.8b lacks; AGAINST if d <= -0.05 OR "
                "losses > wins OR carry adds such a crash; else NO DETECTABLE EFFECT (20 new scenes)")
json.dump(dict(result=res, rows=rows), open(OUT / "cl20_table.json", "w"), indent=1, default=str)
with open(OUT / "cl20_table.csv", "w") as f:
    f.write("block,scene,score_carry,score_018b,diff,atfault_window_carry,atfault_window_018b,exit_carry,exit_018b\n")
    for r in rows:
        c, b = r["carry"] or {}, r["018b"] or {}
        f.write(",".join(map(str, [r["block"], r["scene"], c.get("score"), b.get("score"),
                                   (c["score"] - b["score"]) if c and b else "", c.get("atfault_window"), b.get("atfault_window"),
                                   json.dumps(c.get("exit")), json.dumps(b.get("exit"))])) + "\n")
print(json.dumps(res, indent=1, default=str))
