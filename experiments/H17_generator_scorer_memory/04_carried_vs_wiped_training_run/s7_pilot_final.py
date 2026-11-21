"""Pilot step 5 (measuring only, no training): the two arms' chosen checkpoints, the pre-registered comparison (reports/PILOT_PREREG.md),
GO / STOP / CLOSE by the rule as written, the carry arm's chosen checkpoint with its memory ZEROED at check time, clip-clustered paired
bootstrap intervals (information), learning curves. Usage: run_model_pilot.sh <abs>/s7_pilot_final.py   (CORES=0-1; writes out/pilot_final.json)"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import s6_env as SE
import pai_common as C
C.setup_env()
import pai_model_b as MB
SE.patch_grader(MB)
import s6_model as SM, s6_data as SD, s6_train as ST
import pai_train as T
import s7_pilot as P

OUT = ST.OUT; RUNS = P.RUNS; ADE_REF, ADE_TOL, GO_MIN = 0.238, 0.02, 0.66
B_DRAWS, B_SEED = 2000, 0


def wall_hours(rec: dict) -> float:
    t = lambda s: time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
    h = (t(rec["ended"]) - t(rec["started"])) / 3600
    for r in rec.get("resumes", []):                        # time between a death and its resume is not training time; we cannot know it exactly -> reported as is
        pass
    return h


def chosen_check(rec: dict) -> dict:
    ch = rec["chosen"]; assert ch, f"{rec['arm']}: no choosable check"
    return next(c for c in rec["checks"] if c["progress"] == ch["progress"])


def gates(c: dict) -> dict:
    g = {"hard_picks_above10": c["hard_picks_above10"], "hard_ok": c["hard_picks_above10"] == 0, "go_rate_pull_away": c["go_rate_pull_away"],
         "go_ok": c["go_rate_pull_away"] is not None and c["go_rate_pull_away"] >= GO_MIN, "closest_of_32_ade3": c["closest_of_32_ade3_all"],
         "ade_ok": c["closest_of_32_ade3_all"] <= ADE_REF + ADE_TOL}
    g["pass"] = bool(g["hard_ok"] and g["go_ok"] and g["ade_ok"])
    return g


def bootstrap(a: dict, b: dict, clip_of: dict, subset: set | None):
    """clip-clustered paired bootstrap of mean(a) - mean(b) over the moments (keys of a and b), clusters = clips; 95 % percentile interval."""
    keys = [k for k in a if k in b and (subset is None or k in subset)]
    if not keys:
        return None
    clips = sorted({clip_of[k] for k in keys}); by = {c: [] for c in clips}
    for k in keys:
        by[clip_of[k]].append(a[k] - b[k])
    arr = [np.array(by[c]) for c in clips]; rng = np.random.RandomState(B_SEED); diffs = []
    for _ in range(B_DRAWS):
        pick = rng.randint(0, len(arr), size=len(arr)); v = np.concatenate([arr[i] for i in pick]); diffs.append(v.mean())
    d = np.array(diffs); point = float(np.mean(np.concatenate(arr)))
    return {"moments": len(keys), "clips": len(clips), "point": point, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "draws": B_DRAWS}


def main():
    R = {"started": time.strftime("%F %T")}
    recs = {arm: json.loads((RUNS / arm / "run.json").read_text()) for arm in SM.MODES}
    for arm, rec in recs.items():
        assert rec.get("ended"), f"{arm} has not ended"
    ch = {arm: chosen_check(rec) for arm, rec in recs.items()}
    R["arms"] = {arm: {"chosen": recs[arm]["chosen"], "stopped": recs[arm]["stopped"], "passes_trained": max(c["progress"] for c in recs[arm]["checks"]),
                       "wall_h": wall_hours(recs[arm]), "checks": len(recs[arm]["checks"]) - 1, "resumes": len(recs[arm].get("resumes", [])),
                       "started": recs[arm]["started"], "ended": recs[arm]["ended"], "steps": len(recs[arm]["losses"]["loss"])} for arm in SM.MODES}
    # ---- pre-registered comparison --------------------------------------------------------------------------------------------
    cg, wg = ch["carry"]["pick_grade_mean_map"], ch["wiped"]["pick_grade_mean_map"]
    ce, we = ch["carry"]["event"]["pick_grade_event"], ch["wiped"]["event"]["pick_grade_event"]
    G = {arm: gates(ch[arm]) for arm in SM.MODES}
    comp = {"overall": {"carry": cg, "wiped": wg, "gap": cg - wg, "ref_018b": recs["carry"]["checks"][0]["pick_grade_mean_map"]},
            "event": {"carry": ce, "wiped": we, "gap": (ce - we) if (ce is not None and we is not None) else None, "ref_018b": recs["carry"]["checks"][0]["event"]["pick_grade_event"],
                      "n": ch["carry"]["event"]["graded_event_moments"], "too_small_to_read": ch["carry"]["event"]["graded_event_moments"] < 50},
            "gates": G}
    comp["go"] = bool(cg >= wg and comp["event"]["gap"] is not None and comp["event"]["gap"] >= 0.01 and G["carry"]["pass"] and G["wiped"]["pass"])
    # ---- the carry arm with its memory zeroed at check time ------------------------------------------------------------------------
    agent, info = SM.load_agent(); R["agent"] = info
    model, meta = SM.MemoryModel.load_checkpoint(agent, recs["carry"]["chosen"]["ckpt"]); assert meta["mode"] == "carry"
    model.mode = "wiped"                                                               # zero before every moment, same weights
    store_pk, pk = SD.ClipStore("picking"), ST.picking_frame(); ev = P.event_sets(store_pk.idx)
    grader = MB._GRADER = MB.Grader(ST.POOL)
    ref = recs["carry"]["checks"][0]
    mz, dump_z, grades_z = P.run_check(model, store_pk, pk, grader, ref, ev)
    T.save_dump(RUNS / "carry" / "dumps" / "chosen_memory_zeroed.npz", dump_z); np.save(RUNS / "carry" / "dumps" / "chosen_memory_zeroed_grades.npy", grades_z)
    zero = {"pick_grade": mz["pick_grade_mean_map"], "event_pick_grade": mz["event"]["pick_grade_event"], "carry_own_pick_grade": cg, "drop": cg - mz["pick_grade_mean_map"],
            "relies_on_memory": (cg - mz["pick_grade_mean_map"]) > 0.002, "within_0.002": abs(cg - mz["pick_grade_mean_map"]) <= 0.002,
            "gates": gates(mz), "follows_previous": mz["follows_previous"]["rate"], "pick_ade3_all": mz["pick_ade3_all"]}
    comp["carry_memory_zeroed"] = zero
    comp["close_flags"] = {"overall_gap_under_0.005": abs(comp["overall"]["gap"]) < 0.005,
                           "event_gap_in_0_to_0.02": comp["event"]["gap"] is not None and 0 <= comp["event"]["gap"] <= 0.02,
                           "zeroed_within_0.002": zero["within_0.002"]}
    comp["close"] = any(comp["close_flags"].values())
    comp["verdict"] = ("GO" if comp["go"] else "STOP") + (" + CLOSE" if comp["close"] else "")
    # ---- clip-clustered paired bootstrap (information) ------------------------------------------------------------------------------
    clip_of = dict(zip(store_pk.idx.sample_id, store_pk.idx["clip"]))
    pa, pb = ch["carry"]["pick_grade_per_moment"], ch["wiped"]["pick_grade_per_moment"]
    comp["bootstrap"] = {"overall": bootstrap(pa, pb, clip_of, None), "event": bootstrap(pa, pb, clip_of, ev["event_graded"]),
                         "carry_minus_zeroed_overall": bootstrap(pa, mz["pick_grade_per_moment"], clip_of, None)}
    R["comparison"] = comp
    # ---- learning curves ---------------------------------------------------------------------------------------------------------
    R["curves"] = {arm: [{"progress": c["progress"], "step": c["step"], "pick_grade": c["pick_grade_mean_map"], "event_pick_grade": c["event"]["pick_grade_event"],
                          "hard": c["hard_picks_above10"], "go_pull": c["go_rate_pull_away"], "ade32": c["closest_of_32_ade3_all"], "pick_ade3": c["pick_ade3_all"],
                          "pick_ade3_event": c["event"]["pick_ade3_event"], "follows_prev": c["follows_previous"]["rate"], "cos_read": c["slots"]["cos_read_mean"],
                          "cos_new": c["slots"]["cos_new_mean"], "gate_mean": float(np.mean(c["slots"]["gate_mean_per_slot"])) if c["slots"]["gate_mean_per_slot"] else None,
                          "gen_read_norm": c["memory_use"]["generator_read_out_proj_weight_norm"], "new_cols_norm": c["memory_use"]["rating_head_new_columns_norm"],
                          "gate_ok": c["gate_ok"], "check_wall_s": c.get("check_wall_s")} for c in recs[arm]["checks"]] for arm in SM.MODES}
    R["ended"] = time.strftime("%F %T"); C.wj(OUT / "pilot_final.json", R); grader.close()
    print(json.dumps({k: v for k, v in comp.items() if k not in ("bootstrap",)}, indent=1, default=str)); print(json.dumps(comp["bootstrap"], indent=1))


if __name__ == "__main__":
    main(); sys.stdout.flush(); os._exit(0)
