"""3a cost probe: 3 seeded raw_train chunks (none containing an exam clip): egomotion.offline + obstacle.offline + vehicle_dimensions."""
import json, random, zipfile, pandas as pd, common, r3_rawio as io
V = common.load_split_views(); _, exam, _ = common.load_views()
ci = pd.read_parquet(common.RAW_SNAP / "clip_index.parquet")
tr = ci.loc[sorted(V["raw_train"])]; exam_chunks = set(ci.loc[[u for u in exam if u in ci.index]].chunk)
cands = sorted(set(tr.chunk) - exam_chunks); rng = random.Random(20261003); pick = sorted(rng.sample(cands, 3))
print("seed 20261003 probe chunks", pick)
sizes = json.load(open(common.HERE / "r3_raw_tree_sizes.json")); res = []
for c in pick:
    for f, d, ext in [("egomotion.offline", "labels", "zip"), ("obstacle.offline", "labels", "zip"), ("vehicle_dimensions", "calibration", "parquet")]:
        p = f"{d}/{f}/{f}.chunk_{c:04d}.{ext}"; n, s = io.download(p, f"chunk_{c:04d}")
        nclip = len(zipfile.ZipFile(common.RAW_NAS / p).infolist()) if ext == "zip" else len(pd.read_parquet(common.RAW_NAS / p))
        res.append(dict(chunk=c, feature=f, bytes=n, secs=round(s, 2), clips=nclip, MBps=round(n / 1e6 / s, 1) if s else None)); print(res[-1])
r = pd.DataFrame(res); r.to_csv(common.HERE / "r3_raw_probe.csv", index=False)
tot = r.bytes.sum(); secs = r.secs.sum(); clips = r[r.feature == "obstacle.offline"].clips.sum()
print(f"3 packages: {tot/1e6:.1f} MB, {secs:.1f} s, {tot/1e6/secs:.1f} MB/s, {tot/clips/1e3:.0f} KB per clip ({clips} clips)")
chunks = sorted(set(tr.chunk)); T = {f: sum(sizes[f][f"{d}/{f}/{f}.chunk_{c:04d}.{e}"] for c in chunks) for f, d, e in [("egomotion.offline", "labels", "zip"), ("obstacle.offline", "labels", "zip"), ("vehicle_dimensions", "calibration", "parquet"), ("egomotion", "labels", "zip")]}
print("raw_train remote bytes (from tree listing):", {k: round(v / 1e9, 2) for k, v in T.items()}, "GB over", len(chunks), "chunks")
full = T["egomotion.offline"] + T["obstacle.offline"] + T["vehicle_dimensions"]
print(f"projection, full packages serial at probe rate: {full/1e9:.1f} GB -> {full/(tot/secs)/3600:.2f} h")
