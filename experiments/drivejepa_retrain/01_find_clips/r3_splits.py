"""Run 3: write split views nurec_train / nurec_val / raw_train / raw_val."""
import pandas as pd, common
nurec, raw, pool = common.load_views()
ci = pd.read_parquet(common.RAW_SNAP / "clip_index.parquet")
assert ci.index.name == "clip_id" and ci.clip_is_valid.all()
sp = ci.split
H = "# Run 3, 2026-10-03 (Sid): split from PhysicalAI-Autonomous-Vehicles clip_index.parquet; test excluded everywhere; exam (raw_exclusions.txt, 913) excluded on top.\n"
rows = []
for name, base, s in [("nurec_train", pool, "train"), ("nurec_val", pool, "val"), ("raw_train", set(ci.index), "train"), ("raw_val", set(ci.index), "val")]:
    ids = {u for u in base if sp.get(u) == s} - raw
    open(common.HERE / f"{name}.txt", "w").write(H + "\n".join(sorted(ids)) + "\n")
    rows.append((name, len(ids), ci.loc[sorted(ids)].chunk.nunique()))
print("raw split sizes", sp.value_counts().to_dict())
print("exam (913) in clip_index:", sum(u in ci.index for u in raw), "by split:", sp.reindex(sorted(raw)).value_counts(dropna=False).to_dict())
print("pool by split:", sp.reindex(sorted(pool)).value_counts(dropna=False).to_dict())
for r in rows: print(r)
V = common.load_split_views(); print({k: len(v) for k, v in V.items()})
ex = ci.loc[[u for u in raw if u in ci.index]]
print("train chunks containing >=1 exam clip:", ex[ex.split == "train"].chunk.nunique(), "| val chunks:", ex[ex.split == "val"].chunk.nunique())
fp = pd.read_parquet(common.RAW_SNAP / "metadata/feature_presence.parquet").loc[sorted(V["raw_train"])]
print("raw_train feature presence:", {c: int(fp[c].sum()) for c in ["egomotion", "egomotion.offline", "obstacle.offline", "vehicle_dimensions", "camera_front_wide_120fov"]})
