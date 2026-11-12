#!/usr/bin/env python
"""Closed-loop check of the memory: the per-scene table and the verdict by the pre-registered rule (reports/CL12_MEMORY_PREREG.md).

Per scene and arm (carry, reset: new runs; 0.1.8b pass 0.6 and 0.1.7: existing runs): score and exit with the lane's own definitions
(LORA Refiner/Generator-Joint-PAI-alpasim/s5_diagnosis/src/stepb.py `score`, `exit_of`, `override`, imported unchanged, as s10's stepd.py
does). For carry and reset also, from the "memory: call {...}" lines the adapter writes to driver.log (one per model call): number of calls,
chain resets, mean memory norm, mean gate, and how often the other mode's memory would have picked differently on the same input (scorer
pick before the comfort re-score; final pick after it), all calls and calls after handover. A call is "after handover" when the divider's
replan line that follows it has sim_time_us > handover_us (launch/dj_extra_gates_memory.py's G-SENT record of the run).
Writes out/cl12_table.csv, out/cl12_table.json.     alpasim/.venv/bin/python -B s12_table.py
"""
import json, pathlib, re, sys
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE.parent / "out"
G = pathlib.Path("{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim")
S5 = G / "s5_diagnosis/src"; sys.argv = sys.argv[:1]
src = (S5 / "stepb.py").read_text().split("\nrows = []")[0]
B = {"__file__": str(S5 / "stepb.py")}; exec(compile(src, str(S5 / "stepb.py"), "exec"), B)
CL12 = "{runs}/generator_scorer_memory/runs/cl12"
ARMS = {"carry": f"{CL12}/runs/memory_carry_{{}}", "reset": f"{CL12}/runs/memory_reset_{{}}",
        "018bp06": "{runs}/generator_joint_pai/018b/closed_loop/runs/drivejepa_018bp06_{}", "017": B["R17"]}
if pathlib.Path(f"{CL12}/runs_018bp06").is_dir():          # only if the image check failed and 0.1.8b was rerun
    ARMS["018bp06_rerun"] = f"{CL12}/runs_018bp06/drivejepa_018bp06_{{}}"
S12 = [l.split()[0] for l in open(G / "s10_018b_closed_loop/tables/scenes_12.txt") if l.strip()]
atf = lambda e: isinstance(e, str) and e.startswith("at-fault")      # noqa: E731
off = lambda e: isinstance(e, str) and e.startswith("off-road")      # noqa: E731


def memory_calls(run):
    run = pathlib.Path(run); calls, pend = [], []
    ho = json.load(open(run / "dj_extra_gates.json"))["gates"]["G-SENT"]["handover_us"] if (run / "dj_extra_gates.json").is_file() else None
    for line in open(run / "driver.log", errors="replace"):
        m = re.search(r"memory: call (\{.*\})", line)
        if m:
            pend.append(json.loads(m.group(1))); continue
        m = re.search(r"divider: replan #(\d+) at sim_time_us=(\d+)", line)
        if m:
            for d in pend:
                d["sim_time_us"] = int(m.group(2)); d["after_handover"] = (None if ho is None else int(m.group(2)) > ho); calls.append(d)
            pend = []
    return calls + pend


rows = []
for s in S12:
    r = dict(scene=s)
    for a, root in ARMS.items():
        run = root.format(s)
        if not pathlib.Path(run, "aggregate/results-summary.json").is_file():
            r[f"score_{a}"], r[f"exit_{a}"] = np.nan, "not run"; continue
        r[f"score_{a}"] = B["score"](run)["score"]; r[f"exit_{a}"] = B["exit_of"](run)
        r[f"atfault_{a}"], r[f"offroad_{a}"] = atf(r[f"exit_{a}"]), off(r[f"exit_{a}"])
        r[f"override_{a}"], _ = B["override"](run)
        if a in ("carry", "reset"):
            c = memory_calls(run); ah = [d for d in c if d.get("after_handover")]
            assert c and all(d["mode"] == a for d in c), (a, s)
            pct = lambda L, k: (100.0 * sum(bool(d.get(k)) for d in L) / len(L)) if L else np.nan     # noqa: E731
            r.update({f"calls_{a}": len(c), f"calls_after_handover_{a}": len(ah), f"chain_resets_{a}": c[-1]["chain_resets"],
                      f"memory_norm_{a}": float(np.mean([d["read_norm"] for d in c])), f"gate_mean_{a}": float(np.mean([d["gate_mean"] for d in c])),
                      f"other_mode_memory_norm_{a}": float(np.mean([d["shadow_read_norm"] for d in c])),
                      f"n_differ_scorer_{a}": sum(bool(d["picks_differ_scorer"]) for d in c), f"n_differ_final_{a}": sum(bool(d["picks_differ_final"]) for d in c),
                      f"n_differ_scorer_ah_{a}": sum(bool(d["picks_differ_scorer"]) for d in ah), f"n_differ_final_ah_{a}": sum(bool(d["picks_differ_final"]) for d in ah),
                      f"pct_differ_scorer_{a}": pct(c, "picks_differ_scorer"), f"pct_differ_final_{a}": pct(c, "picks_differ_final"),
                      f"pct_differ_scorer_ah_{a}": pct(ah, "picks_differ_scorer"), f"pct_differ_final_ah_{a}": pct(ah, "picks_differ_final"),
                      f"picked_plan_gap_median_m_{a}": float(np.median([d["picked_plan_gap_m"] for d in c])),
                      f"menu_gap_median_m_{a}": float(np.median([d["menu_gap_m"] for d in c]))})
    rows.append(r)
D = pd.DataFrame(rows); D.to_csv(OUT / "cl12_table.csv", index=False)
done = {a: int(D[f"score_{a}"].notna().sum()) for a in ARMS}; means = {a: float(D[f"score_{a}"].mean()) for a in ARMS}
res = {"scenes_run": done, "means": means, "offline_pick_grade": {"build5_carried": 0.7313, "build5_reset": 0.7355, "018b_pass_0.6": 0.7685}}
if done["carry"] == 12 and done["reset"] == 12:
    d = means["carry"] - means["reset"]
    c_atf = D[D.atfault_carry == True].scene.tolist(); r_atf = D[D.atfault_reset == True].scene.tolist()      # noqa: E712
    added = [s for s in c_atf if s not in r_atf]
    verdict = "memory hurts" if (d <= -0.08 or added) else ("memory helps in closed loop" if d >= 0.08 else "no detectable effect")
    pool = lambda a, k, n: 100.0 * D[f"{k}_{a}"].sum() / D[f"{n}_{a}"].sum()      # noqa: E731
    res.update({"carry_minus_reset": d, "atfault_carry": c_atf, "atfault_reset": r_atf, "atfault_carry_not_reset": added,
                "atfault_reset_not_carry": [s for s in r_atf if s not in c_atf],
                "offroad_carry": D[D.offroad_carry == True].scene.tolist(), "offroad_reset": D[D.offroad_reset == True].scene.tolist(),      # noqa: E712
                "scenes_differ_gt_0.1": [(s, round(a, 3), round(b, 3)) for s, a, b in zip(D.scene, D.score_carry, D.score_reset) if abs(a - b) > 0.1],
                "picked_differently_pct": {f"{a}_run": {"scorer_pick_all_calls": pool(a, "n_differ_scorer", "calls"), "final_pick_all_calls": pool(a, "n_differ_final", "calls"),
                                                         "scorer_pick_after_handover": pool(a, "n_differ_scorer_ah", "calls_after_handover"),
                                                         "final_pick_after_handover": pool(a, "n_differ_final_ah", "calls_after_handover"),
                                                         "calls": int(D[f"calls_{a}"].sum()), "calls_after_handover": int(D[f"calls_after_handover_{a}"].sum())} for a in ("carry", "reset")},
                "chain_resets_total": {a: int(D[f"chain_resets_{a}"].sum()) for a in ("carry", "reset")}, "verdict": verdict})
json.dump(res, open(OUT / "cl12_table.json", "w"), indent=1, default=str)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60)
print(D[["scene"] + [f"score_{a}" for a in ARMS]].round(3).to_string())
print(D[["scene"] + [f"exit_{a}" for a in ARMS]].to_string())
cols = [c for c in D.columns if c.startswith(("calls_", "chain_resets_", "memory_norm_", "gate_mean_", "pct_differ_"))]
print(D[["scene"] + cols].round(2).to_string()); print(json.dumps(res, indent=1, default=str))
