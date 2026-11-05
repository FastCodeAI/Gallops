"""D-C: raw-photo availability, metadata only. Match key: NuRec folder uuid (= clip uuid in scene id clipgt-<uuid>) ==
clip_id index of PhysicalAI-Autonomous-Vehicles clip_index.parquet / metadata/feature_presence.parquet."""
import os, json, common
import pandas as pd
from huggingface_hub import HfApi
nurec, raw, pool = common.load_views()
SNAP = "{runs}/drivejepa_retrain/hf_home/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/33f9bf447ed3bcb7d545ce13f4226f824214fafb"
di = HfApi().dataset_info("nvidia/PhysicalAI-Autonomous-Vehicles")
print("remote sha", di.sha, "last_modified", di.last_modified, "| local snapshot", SNAP.split("/")[-1], "| same:", di.sha == SNAP.split("/")[-1])
ci = pd.read_parquet(f"{SNAP}/clip_index.parquet"); fp = pd.read_parquet(f"{SNAP}/metadata/feature_presence.parquet"); dc = pd.read_parquet(f"{SNAP}/metadata/data_collection.parquet")
for n, d in [("clip_index", ci), ("feature_presence", fp), ("data_collection", dc)]:
    print(n, d.shape, "index:", d.index.name, "cols:", list(d.columns)[:12])
def idx(d): return d if d.index.name == "clip_id" else d.set_index("clip_id")
ci, fp, dc = idx(ci), idx(fp), idx(dc)
P = sorted(pool); inci = [u for u in P if u in ci.index]
print(f"pool {len(P)}: in clip_index {len(inci)} ({100*len(inci)/len(P):.1f}%)")
col = [c for c in fp.columns if "front_wide" in c]; print("front-wide presence columns:", col)
fw = fp.loc[[u for u in inci if u in fp.index], col[0]]
print("in feature_presence", len(fw), "front_wide True", int(fw.sum()))
ok = set(fw[fw].index)
m = ci.loc[inci]; print("split:", m["split"].value_counts().to_dict(), "| valid:", m["clip_is_valid"].value_counts().to_dict(), "| chunks:", m["chunk"].nunique())
d = dc.loc[[u for u in inci if u in dc.index]]
for c in dc.columns:
    if dc[c].nunique() < 30: print(" ", c, d[c].value_counts().to_dict())
# near-miss check for unmatched: same first 8 hex present?
miss = [u for u in P if u not in ci.index]
pref = {i[:8] for i in ci.index}; print("unmatched", len(miss), "of which 8-hex prefix exists in clip_index:", sum(u[:8] in pref for u in miss), "(chance collisions expected ~", round(len(miss)*len(ci)/16**8, 2), ")")
# for reference only, uuid membership (no data read): exam view
print("exam nurec_2604 view (uuid lookup only): in clip_index", sum(u in ci.index for u in nurec), "of", len(nurec))
out = pd.DataFrame({"uuid": P}); out["in_clip_index"] = out.uuid.isin(ci.index); out["front_wide"] = out.uuid.isin(ok)
out["split"] = out.uuid.map(ci["split"]); out["chunk"] = out.uuid.map(ci["chunk"])
out.to_csv(common.HERE / "dc_raw_match.csv", index=False)
print(f"P-C: matched with front_wide {len(ok)}/{len(P)} = {100*len(ok)/len(P):.1f}%")
