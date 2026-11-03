#!/usr/bin/env python
"""Generator-only trial: write tables/review_groups.json from tables/per_scene.csv. Group membership by score change 0.1.6 -> 0.1.7 (|delta| < 0.02 = same), buckets of the unchanged group
by what both runs do, one generated sentence per scene from the table's numbers. Leads / points / header come from tables/insights_017.json when it exists (hand-written at the end of the run);
until then they are generated counts marked PROVISIONAL."""
import json, pathlib, numpy as np, pandas as pd
T = pathlib.Path(__file__).resolve().parent / "tables"; d = pd.read_csv(T / "per_scene.csv").set_index("scene"); dl = d["017_scene_score"] - d["016_scene_score"]; N = len(d)
def ev(a, s):
    r = d.loc[s]; k = str(r[f"{a}_exit_kind"]); t = r[f"{a}_exit_s"]
    if k == "fail:collision_at_fault": return f"at-fault collision at +{t:.1f} s"
    if k == "fail:offroad": return f"off the road at +{t:.1f} s"
    if k == "end_of_clip": return "runs to the end of the clip"
    c = r[f"{a}_first_collision_any_s"]
    return f"scoring ends at +{t:.1f} s" + (f" (collision, not at fault, at +{c:.1f} s)" if np.isfinite(c) and r[f"{a}_collision_any"] > 0 else "")
why = {s: f"0.1.6: {ev('016', s)}. 0.1.7: {ev('017', s)}. Mean speed {d.loc[s, '016_mean_speed_after_ho']:.1f} → {d.loc[s, '017_mean_speed_after_ho']:.1f} m/s." for s in d.index}
imp = list(dl[dl >= 0.02].sort_values(ascending=False).index); dec = list(dl[dl <= -0.02].sort_values().index); same = [s for s in d.index if abs(dl[s]) < 0.02]
full = [s for s in same if d.loc[s, "016_scene_score"] >= 0.999 and d.loc[s, "017_scene_score"] >= 0.999]; zero = [s for s in same if d.loc[s, "016_scene_score"] <= 0.05 and d.loc[s, "017_scene_score"] <= 0.05]; oth = [s for s in same if s not in full + zero]
B = [dict(title="Full score under both", scenes=full), dict(title="Scores zero (or nearly) under both", scenes=zero)] + ([dict(title="Same partial score", scenes=oth)] if oth else [])
m = lambda a: float(d[f"{a}_scene_score"].mean()); af = lambda a: int((d[f"{a}_collision_at_fault"] > 0).sum())
G = dict(header=f"PROVISIONAL, {N} of 30 scenes finished. Generator only: 0.1.7 = 0.1.6 + the S10 generator weights + the batch fix on; scorer unchanged. Mean score over these {N}: {m('016'):.3f} (0.1.6) to {m('017'):.3f} (0.1.7); 0.1.4b {m('014b'):.3f}. {len(imp)} improved, {len(dec)} decreased, {len(same)} unchanged; at-fault collisions {af('016')} to {af('017')} (0.1.4b {af('014b')}).",
         improved=dict(title="Score improved", scenes=imp, lead=f"PROVISIONAL: {len(imp)} of the {N} finished scenes score higher under 0.1.7.", points=["Written insights follow when all 30 scenes are finished; the table and the videos are final for the scenes shown."]),
         decreased=dict(title="Score decreased", scenes=dec, lead=f"PROVISIONAL: {len(dec)} of the {N} finished scenes score lower under 0.1.7.", points=[]),
         same=dict(title="No change in score", lead=f"PROVISIONAL: {len(same)} of the {N} finished scenes score the same (within 0.02).", points=[], buckets=B), why=why)
ins = T / "insights_017.json"
if ins.is_file():
    I = json.load(open(ins)); G["header"] = I["header"]; G["key"] = I.get("key") or None
    for g in ("improved", "decreased", "same"): G[g]["lead"] = I[g]["lead"]; G[g]["points"] = I[g]["points"]
    G["why"].update(I.get("why", {}))
    for b in G["same"]["buckets"]: b["note"] = I.get("bucket_notes", {}).get(b["title"], "")
    if not G.get("key"): G.pop("key", None)
json.dump(G, open(T / "review_groups.json", "w"), indent=1, ensure_ascii=False); print({k: len(v) for k, v in dict(improved=imp, decreased=dec, same=same, full=full, zero=zero, other=oth).items()})
