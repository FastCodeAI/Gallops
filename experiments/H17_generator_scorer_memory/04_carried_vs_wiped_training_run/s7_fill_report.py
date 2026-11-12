"""Fills reports/PILOT_SEED0.md from out/pilot_final.json and the two run.json files (every number from a file). Usage: python3 -I s7_fill_report.py [extra.json]"""
import json, re, sys, time
from pathlib import Path
import numpy as np
M = Path("{working_folder}"); R = Path("{runs}/generator_scorer_memory/runs/pilot_seed0")
F = json.load(open(M / "out/pilot_final.json")); recs = {a: json.load(open(R / a / "run.json")) for a in ("carry", "wiped")}
comp = F["comparison"]; A = F["arms"]; V = {}
f = lambda x, d=4: "—" if x is None else f"{x:.{d}f}"
pct = lambda x: "—" if x is None else f"{100 * x:.1f} %"
yn = lambda b: "yes" if b else "no"
ch = {a: next(c for c in recs[a]["checks"] if c["progress"] == recs[a]["chosen"]["progress"]) for a in recs}
gt = lambda c: f"{c['hard_picks_above10']} / {pct(c['go_rate_pull_away'])} / {c['closest_of_32_ade3_all']:.3f} m → " + ("pass" if comp["gates"]["carry" if c is ch["carry"] else "wiped"]["pass"] else "FAIL")
for a, k in (("carry", "C"), ("wiped", "W")):
    c = ch[a]; arm = A[a]
    V[f"{k}_CHOSEN"] = f"pass {c['progress']:.3f}, step {c['step']:,}"
    V[f"{k}_STOP"] = f"{arm['stopped']}; {arm['passes_trained']:.2f} passes; {arm['wall_h']:.1f} h"
    V[f"{k}_GRADE"] = f(c["pick_grade_mean_map"]); V[f"{k}_EV"] = f(c["event"]["pick_grade_event"])
    V[f"{k}_GATES"] = gt(c); V[f"{k}_HARD"] = str(c["hard_picks_above10"]); V[f"{k}_GO"] = pct(c["go_rate_pull_away"]); V[f"{k}_ADE"] = f"{c['closest_of_32_ade3_all']:.3f} m"
    V[f"{k}_TIMES"] = f"{arm['started'][5:16]} / {arm['ended'][5:16]} ({arm['wall_h']:.1f} h)"; V[f"{k}_STEPS"] = f"{arm['steps']:,} steps, {arm['passes_trained']:.2f} passes"
    V[f"{k}_NCHK"] = str(arm["checks"]); V[f"{k}_STOPR"] = arm["stopped"]; V[f"{k}_RES"] = str(arm["resumes"])
    st = recs[a]["steps"]; V[f"{k}_STEPT"] = f"{np.mean([s['step_s'] for s in st]):.2f} / {np.mean([s['grade_wait_s'] for s in st]):.2f} / {np.mean([s['data_wait_s'] for s in st]):.2f}"
    V[f"{k}_NAN"] = "none" if all(np.isfinite(recs[a]["losses"]["loss"])) else "YES"
    V[f"{k}_FOLLOW"] = pct(c["follows_previous"]["rate"]); V[f"{k}_PADE"] = f"{c['pick_ade3_all']:.3f} / {f(c['event']['pick_ade3_event'], 3)} m"
    V[f"{k}_COS"] = f"{f(c['slots']['cos_read_mean'])} / {f(c['slots']['cos_new_mean'])}"; V[f"{k}_COSNEW"] = f(c["slots"]["cos_new_mean"])
    V[f"{k}_GATEM"] = f(float(np.mean(c["slots"]["gate_mean_per_slot"])), 3) if c["slots"]["gate_mean_per_slot"] else "—"
    V[f"{k}_NORMS"] = f"{c['memory_use']['generator_read_out_proj_weight_norm']:.3f} / {c['memory_use']['rating_head_new_columns_norm']:.3f}"
    rows = []
    for x in F["curves"][a]:
        rows.append(f"| {x['progress']:.3f} | {x['step']:,} | {x['pick_grade']:.4f} | {f(x['event_pick_grade'])} | {x['hard']} | {pct(x['go_pull'])} | {x['ade32']:.3f} | {x['pick_ade3']:.3f} / {f(x['pick_ade3_event'], 3)} | {pct(x['follows_prev'])} | {f(x['cos_read'])} / {f(x['cos_new'])} | {f(x['gate_mean'], 3)} | {x['gen_read_norm']:.3f} | {x['new_cols_norm']:.3f} | {'pass' if x['gate_ok'] else 'FAIL'} |")
    V[f"{k}_CURVE"] = "\n".join(rows)
V["C_CKPT"] = recs["carry"]["chosen"]["ckpt"]
b0 = recs["carry"]["checks"][0]; V["B_EV"] = f(b0["event"]["pick_grade_event"]); V["B_PADE"] = f"{b0['pick_ade3_all']:.3f} / {f(b0['event']['pick_ade3_event'], 3)} m"; V["B_COSNEW"] = f(b0["slots"]["cos_new_mean"])
V["N_EV"] = str(comp["event"]["n"]); V["GAP_ALL"] = f"{comp['overall']['gap']:+.4f}"; V["GAP_EV"] = f"{comp['event']['gap']:+.4f}" if comp["event"]["gap"] is not None else "—"
z = comp["carry_memory_zeroed"]; V["Z_GRADE"] = f(z["pick_grade"]); V["Z_DROP"] = f"{z['drop']:+.4f}"; V["Z_RELIES"] = yn(z["relies_on_memory"]); V["Z_EV"] = f(z["event_pick_grade"])
V["Z_GATES"] = f"{z['gates']['hard_picks_above10']} / {pct(z['gates']['go_rate_pull_away'])} / {z['gates']['closest_of_32_ade3']:.3f} m → {'pass' if z['gates']['pass'] else 'FAIL'}"; V["Z_FOLLOW"] = pct(z["follows_previous"])
V["Z_TEXT"] = ("The pick grade drops by more than 0.002 when the memory is zeroed: the carry model relies on its memory." if z["relies_on_memory"] else
               ("The pick grade is within 0.002 of the carry model's own: the carry model ignores its memory (CLOSE flag)." if z["within_0.002"] else "The pick grade RISES when the memory is zeroed: the memory hurts the pick."))
V["R1"] = yn(comp["overall"]["carry"] >= comp["overall"]["wiped"]); V["R2"] = yn(comp["event"]["gap"] is not None and comp["event"]["gap"] >= 0.01); V["R3"] = yn(comp["gates"]["carry"]["pass"] and comp["gates"]["wiped"]["pass"])
V["GOSTOP"] = "GO" if comp["go"] else "STOP"; cf = comp["close_flags"]; V["CL1"], V["CL2"], V["CL3"] = yn(cf["overall_gap_under_0.005"]), yn(cf["event_gap_in_0_to_0.02"]), yn(cf["zeroed_within_0.002"])
V["CLOSE"] = "CLOSE" if comp["close"] else "not close"; V["VERDICT"] = comp["verdict"]
bs = comp["bootstrap"]; fb = lambda b: "—" if b is None else f"{b['point']:+.4f} [95 % {b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] ({b['moments']} moments, {b['clips']} clips)"
V["BS_ALL"], V["BS_EV"], V["BS_Z"] = fb(bs["overall"]), fb(bs["event"]), fb(bs["carry_minus_zeroed_overall"])
V["STEPS_PASS"] = f"{recs['carry']['passes'][0]['steps']:,}"; V["END"] = F["ended"][11:16]
V["GPU_LAUNCH"] = "76.0 GB"
extra = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
V.update(extra)
txt = (M / "reports/PILOT_SEED0.md").read_text()
for k, v in V.items():
    txt = txt.replace("{{" + k + "}}", str(v))
(M / "reports/PILOT_SEED0.md").write_text(txt)
print("left:", re.findall(r"\{\{[A-Z0-9_]+\}\}", txt))
