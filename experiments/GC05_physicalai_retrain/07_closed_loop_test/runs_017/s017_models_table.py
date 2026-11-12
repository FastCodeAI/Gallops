#!/usr/bin/env python3
"""All-models score comparison for the review page (tab "Models"): final score per scene, 20 Hz referee, pilot10 and the 30 scenes where run.
Every number is read from a run artifact: aggregate/results-summary.json (rollouts[].score) or the per-scene tables this page already uses.
Writes tables/models_compare.json.  Run: cd paddock-sid/Alpasim && alpasim/.venv/bin/python analysis_3.0/Retrain-data/07_closed_loop_test/runs_017/s017_models_table.py"""
import glob, json, pathlib
import pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; ALP = HERE.parents[3]; R = ALP / "runs"; RR = HERE.parents[2] / "Re-ranker"
def summ(*pats, skip=""):
    out = {}
    for p in pats:
        for g in sorted(glob.glob(str(R / p))):
            if skip and skip in g: continue
            for r in json.load(open(g))["rollouts"]:
                out[r["clipgt_id"][7:15]] = dict(score=r["score"], coll=bool(r["metrics"].get("collision_any")), at_fault=bool(r["metrics"].get("collision_at_fault")))
    return out
def table(path, col, cany, cfault):
    d = pd.read_csv(path); return {r["scene"]: dict(score=float(r[col]), coll=bool(r[cany]), at_fault=bool(r[cfault])) for _, r in d.iterrows()}
P17 = HERE / "tables" / "per_scene.csv"; P15 = RR / "s5_closed_loop" / "s5" / "tables" / "per_scene.csv"
c15 = [c for c in pd.read_csv(P15).columns if c in ("new_scene_score", "new_score")][0]
M = [
 ("CV floor 0.2.3", "constant velocity, no cameras", "runs/baselines/cv_023_20hz_{pilot10,ext20}", summ("baselines/cv_023_20hz_pilot10/aggregate/results-summary.json", "baselines/cv_023_20hz_ext20/cv_023_20hz_ext20_*/aggregate/results-summary.json")),
 ("OpenPilot 0.2.0", "supercombo, 20 Hz query", "runs/openpilot/openpilot_020_{pilot10,ext20}", summ("openpilot/openpilot_020_pilot10/aggregate/results-summary.json", "openpilot/openpilot_020_ext20/*/aggregate/results-summary.json")),
 ("VisionPilot 2.0.0", "10 Hz core, set speed given (oracle)", "runs/visionpilot/visionpilot_v2_{smoke_pilot10_handover_blend,ext20}", summ("visionpilot/visionpilot_v2_smoke_pilot10_handover_blend/aggregate/results-summary.json", "visionpilot/visionpilot_v2_ext20/*/aggregate/results-summary.json")),
 ("Alpamayo 1.5", "proxy 0.1.0, 4 cameras", "runs/alpamayo/alpamayo_proxy_{referee_*,ext20}", summ("alpamayo/alpamayo_proxy_referee_????????/aggregate/results-summary.json", "alpamayo/alpamayo_proxy_ext20/*/aggregate/results-summary.json")),
 ("METEOR 0.1.1 (L1)", "spliced model + Orin selector", "runs/meteor/meteor_011_*_p10", summ("meteor/meteor_011_????????_p10/aggregate/results-summary.json")),
 ("Drive-JEPA 0.1.4", "as published", "runs/drivejepa/drivejepa_014_*", summ("drivejepa/drivejepa_014_????????/aggregate/results-summary.json", "drivejepa/drivejepa_014_ext20/drivejepa_014_ext20_*/aggregate/results-summary.json")),
 ("Drive-JEPA 0.1.4b", "0.1.4 + §3.5 momentum", "Re-ranker/baseline_014b (tables/per_scene.csv 014b_*)", table(P17, "014b_scene_score", "014b_collision_any", "014b_collision_at_fault")),
 ("Drive-JEPA 0.1.5", "learned re-ranker + veto", "Re-ranker/s5_closed_loop/s5/tables/per_scene.csv", table(P15, c15, "new_collision_any", "new_collision_at_fault")),
 ("Drive-JEPA 0.1.6", "first retraining (generator + scorer, pass 9)", "07_closed_loop_test (tables/per_scene.csv 016_*)", table(P17, "016_scene_score", "016_collision_any", "016_collision_at_fault")),
 ("Drive-JEPA 0.1.7", "S10 generator + batch fix, same scorer", "07_closed_loop_test/runs_017 (tables/per_scene.csv 017_*)", table(P17, "017_scene_score", "017_collision_any", "017_collision_at_fault")),
]
def ids(csv): return [l.split(",")[0][7:15] for l in open(csv).read().splitlines()[1:] if l.startswith("clipgt-")]
SS = ALP.parent / "scene_sets"; EXT = ids(SS / "ext20_v1.csv")
PILOT = sorted(s for s in M[0][3] if s not in EXT); assert len(PILOT) == 10 and len(EXT) == 20, (len(PILOT), len(EXT))
mean = lambda xs: (sum(xs) / len(xs)) if xs else None
rows = []
for name, note, src, d in M:
    p = [s for s in PILOT if s in d]; e = [s for s in EXT if s in d]
    rows.append(dict(name=name, note=note, src=src, n_pilot=len(p), n_ext=len(e),
        pilot=mean([d[s]["score"] for s in p]) if len(p) == 10 else None, ext=mean([d[s]["score"] for s in e]) if e else None,
        all30=mean([d[s]["score"] for s in p + e]) if len(p) == 10 and len(e) == 20 else None,
        coll=sum(d[s]["coll"] for s in p + e), at_fault=sum(d[s]["at_fault"] for s in p + e), scores={s: d[s]["score"] for s in p + e}))
    r = rows[-1]; print(f"{name:20s} pilot {r['n_pilot']:2d} {r['pilot'] if r['pilot'] is None else round(r['pilot'],4)}  ext {r['n_ext']:2d} {r['ext'] if r['ext'] is None else round(r['ext'],4)}  all30 {r['all30'] if r['all30'] is None else round(r['all30'],4)}  coll {r['coll']} at-fault {r['at_fault']}")
json.dump(dict(pilot=PILOT, ext=EXT, rows=rows), open(HERE / "tables" / "models_compare.json", "w"), indent=1)
