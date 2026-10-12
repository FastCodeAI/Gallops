"""Independent check of the review agent's vocabulary with gallops-cache fastgrade (tested against grader)."""
import importlib.util, json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, "{gallops}/src")
from gallops.expert import fastgrade, grader
from gallops.expert.drivable import DrivableArea
from gallops.expert.lanes import load_references, reference_at
import swerve_vocab as sv
spec = importlib.util.spec_from_file_location("v", "vocab_review/vocab.py"); v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
R = Path("{runs}")
index = sv.dump_index()
tot = {"rescue": 0, "native": 0, "both": 0, "vocab": 0}; acc = []; rad = []
for name, run in [(a, int(b)) for a, b in (x.split(":") for x in sys.argv[1:])]:
    ep = R / "collect" / name / f"run_{run}"
    rows = sv.dump_rows(ep, index)
    labels = {r["timestamp"]: r for r in map(json.loads, (R / "labelled_round0_v5" / f"{name}__run_{run}.jsonl").read_text().splitlines())}
    poses = json.loads((ep / "ego_poses.json").read_text())
    actors = [a for a in json.loads((ep / "actors.json").read_text()) if a["dims"][1] >= 1.2]
    area = DrivableArea.load(R / "drivable" / f"scene-{name.split('-')[1]}.npz")
    logged, lanes = load_references(ep)
    c = {"rescue": 0, "native": 0, "both": 0, "vocab": 0}
    for ts, lab in labels.items():
        if ts not in rows: continue
        e2w = np.asarray(poses[str(ts)]); natives = np.asarray(rows[ts]["proposals"]); pi = int(np.argmax(rows[ts]["pdm_score"]))
        paths, meta = v.vocabulary(natives, pi, float(rows[ts]["speed"]), float(rows[ts]["accel"]))
        acc += [m["peak_lat_accel"] for m in meta]; rad += [m["min_radius_m"] for m in meta]
        ref, _ = reference_at(ts, logged, lanes)
        g = fastgrade.grade_many(np.concatenate([natives, paths]), e2w, ts, actors, ref, area)
        safe = np.isnan(g["hit_at"]) & np.isnan(g["off_road_at"])
        if np.isnan(g["hit_at"][pi]) or not lab.get("hazard_in_frame"): continue
        c["rescue"] += 1; c["native"] += safe[:32].any(); c["both"] += safe.any(); c["vocab"] += safe[32:].any()
    print(name, run, c, flush=True)
    for k in tot: tot[k] += c[k]
print("TOTAL", tot, "max peak_lat_accel", max(acc), "min radius", min(rad))
