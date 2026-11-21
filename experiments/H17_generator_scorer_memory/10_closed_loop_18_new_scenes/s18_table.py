#!/usr/bin/env python
"""18-scene closed loop: per-scene table, primary rule, robustness and the 30-scene table (reports/CL18_MEMORY_PREREG.md).
Score / exit / at-fault: s5_diagnosis/src/stepb.py `score`, `exit_of` imported unchanged, as s12_table.py. The carry run's shadow share is read
from its "memory: call" lines (s12_table.memory_calls).  Writes out/cl18_table.{csv,json}.   alpasim/.venv/bin/python -B s18_table.py"""
import json, pathlib, sys
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE.parent / "out"
G = pathlib.Path("{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim")
S5 = G / "s5_diagnosis/src"; sys.argv = sys.argv[:1]
src = (S5 / "stepb.py").read_text().split("\nrows = []")[0]
B = {"__file__": str(S5 / "stepb.py")}; exec(compile(src, str(S5 / "stepb.py"), "exec"), B)
T12 = (HERE / "s12_table.py").read_text(); MC = {"__file__": str(HERE / "s12_table.py")}; exec(compile(T12.split("\nrows = []")[0], "s12_table.py", "exec"), MC)     # memory_calls
CL12, CL18 = "{experiment_store}/runs/cl12", "{experiment_store}/runs/cl18"
NEW = {"carry": f"{CL18}/runs/memory_carry_{{}}", "018bp06": f"{CL18}/018bp06/runs/drivejepa_018bp06_{{}}", "017": B["R17"]}
OLD = {"carry": f"{CL12}/runs/memory_carry_{{}}", "018bp06": "{retrain_store}/018b/closed_loop/runs/drivejepa_018bp06_{}", "017": B["R17"]}
S18 = [l.split()[0] for l in open(OUT / "cl18_scenes_18.txt") if l.strip()]
S12 = [l.split()[0] for l in open(G / "s10_018b_closed_loop/tables/scenes_12.txt") if l.strip()]
assert len(S18) == 18 and len(S12) == 12 and not set(S18) & set(S12)
atf = lambda e: isinstance(e, str) and e.startswith("at-fault")      # noqa: E731
off = lambda e: isinstance(e, str) and e.startswith("off-road")      # noqa: E731


def row(s, arms, block):
    r = dict(scene=s, block=block)
    for a, root in arms.items():
        run = root.format(s)
        if not pathlib.Path(run, "aggregate/results-summary.json").is_file():
            r[f"score_{a}"], r[f"exit_{a}"] = np.nan, "not run"; continue
        r[f"score_{a}"] = B["score"](run)["score"]; r[f"exit_{a}"] = B["exit_of"](run)
        r[f"atfault_{a}"], r[f"offroad_{a}"] = atf(r[f"exit_{a}"]), off(r[f"exit_{a}"])
        if a == "carry" and block == "18":
            c = MC["memory_calls"](run); assert c and all(d["mode"] == "carry" for d in c), s
            r.update(calls=len(c), chain_resets=c[-1]["chain_resets"], n_differ_final=sum(bool(d["picks_differ_final"]) for d in c),
                     n_differ_scorer=sum(bool(d["picks_differ_scorer"]) for d in c), gate_mean=float(np.mean([d["gate_mean"] for d in c])))
    return r


D = pd.DataFrame([row(s, NEW, "18") for s in S18] + [row(s, OLD, "12") for s in S12])
D.to_csv(OUT / "cl18_table.csv", index=False)
N = D[D.block == "18"].copy(); done = {a: int(N[f"score_{a}"].notna().sum()) for a in ("carry", "018bp06")}
res = {"scenes_18": S18, "done_18": done}
if done["carry"] == 18 and done["018bp06"] == 18:
    N["diff"] = N.score_carry - N.score_018bp06
    d = float(N["diff"].mean()); ca = N[N.atfault_carry == True].scene.tolist(); ba = N[N.atfault_018bp06 == True].scene.tolist()     # noqa: E712
    added = [s for s in ca if s not in ba]
    verdict = "HURTS" if (d <= -0.08 or added) else ("HELPS" if d >= 0.08 else "NO DETECTABLE EFFECT")
    top2 = N.sort_values("diff", ascending=False).scene.tolist()[:2]
    res.update(mean_18={"carry": float(N.score_carry.mean()), "018bp06": float(N.score_018bp06.mean()), "017_earlier_image": float(N.score_017.mean())},
               d=d, atfault_carry=ca, atfault_018bp06=ba, atfault_carry_not_018bp06=added, atfault_018bp06_not_carry=[s for s in ba if s not in ca],
               verdict=verdict,
               robust={"two_most_in_carrys_favour": top2, "d_without_them": float(N[~N.scene.isin(top2)]["diff"].mean()),
                       "wins": N[N["diff"] > 0.1].scene.tolist(), "losses": N[N["diff"] < -0.1].scene.tolist(),
                       "ties": int((N["diff"].abs() <= 0.1).sum()),
                       "offroad_carry": N[N.offroad_carry == True].scene.tolist(), "offroad_018bp06": N[N.offroad_018bp06 == True].scene.tolist()},     # noqa: E712
               shadow={"calls": int(N.calls.sum()), "final_pick_differs_pct": 100.0 * N.n_differ_final.sum() / N.calls.sum(),
                       "scorer_pick_differs_pct": 100.0 * N.n_differ_scorer.sum() / N.calls.sum(), "chain_resets": int(N.chain_resets.sum())})
    A = D.dropna(subset=["score_carry", "score_018bp06"])
    if len(A) == 30:
        res["mean_30"] = {"carry": float(D.score_carry.mean()), "018bp06": float(D.score_018bp06.mean()), "017_earlier_image": float(D.score_017.mean()),
                          "atfault_carry": int(D.atfault_carry.sum()), "atfault_018bp06": int(D.atfault_018bp06.sum()), "atfault_017": int(D.atfault_017.sum()),
                          "offroad_carry": int(D.offroad_carry.sum()), "offroad_018bp06": int(D.offroad_018bp06.sum()), "offroad_017": int(D.offroad_017.sum())}
json.dump(res, open(OUT / "cl18_table.json", "w"), indent=1, default=str)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
print(D[["block", "scene", "score_carry", "score_018bp06", "score_017"]].round(3).to_string())
print(D[["scene", "exit_carry", "exit_018bp06", "exit_017"]].to_string()); print(json.dumps(res, indent=1, default=str))
