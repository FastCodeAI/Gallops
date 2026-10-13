"""S3.1 V1: small validation pick (seed 20261007). Run with S3_SPLIT=val. 8 nurec_val scenes by S0 Run 3 flags (2 high-speed, 3 close-pass,
3 ordinary) + 12 raw_val clips. raw_val was never scanned for pull-outs (S0 Run 3 scanned raw_train only), so all 12 are ordinary.
Also fetches, for the val chunks used, the egomotion.offline package and vehicle_dimensions file (S0 raw-cache layout)."""
import io, zipfile, random, hashlib, requests, numpy as np, pandas as pd
import s3_common as C; import r3_core as K
assert C.SPLIT == "val"; V = C.views(); VV, exam, ci = V["V"], V["exam"], V["ci"]; tok = C.token()
fp = pd.read_parquet(C.S0C.RAW_SNAP / "metadata/feature_presence.parquet"); dc = pd.read_parquet(C.S0C.RAW_SNAP / "metadata/data_collection.parquet")
need = [C.FRONT, "camera_intrinsics.offline", "sensor_extrinsics.offline", "egomotion.offline", "obstacle.offline", "vehicle_dimensions"]
ok = set(fp.index[fp[need].all(axis=1)]); exam_chunks = set(ci.loc[[u for u in exam if u in ci.index]].chunk)
elig = lambda us: sorted(u for u in us if u in VV["raw_val"] and u in ok and ci.loc[u, "chunk"] not in exam_chunks)
an = pd.read_csv(C.S0 / "d3_anchors.csv", low_memory=False); ar = pd.read_csv(C.S0 / "d3_around.csv"); mo = pd.read_csv(C.S0 / "d3_moveoffs.csv")
A = set(an[(an.A == True) & (an.split == "val")].uuid); P = set(ar[(ar.split == "val") & (ar.clearance <= 1.0)].uuid) - A; O = VV["nurec_val"] - A - set(ar[ar.split == "val"].uuid) - set(mo[mo.split == "val"].uuid)
rng = random.Random(20261007); rows = [dict(uuid=u, source="nurec", type=t) for t, pool, n in (("A", A, 2), ("B_pass", P, 3), ("ordinary", O, 3)) for u in rng.sample(elig(pool), n)]
def fetch(path):
    p = C.S0C.RAW_NAS / path
    if not p.exists():
        r = requests.get(f"https://huggingface.co/datasets/{C.RAW_REPO}/resolve/main/{path}", headers={"Authorization": f"Bearer {tok}"}, timeout=300); r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(r.content); C.manifest_add(path.split("/")[-1].split(".")[-2], path, len(r.content), str(p), hashlib.sha256(r.content).hexdigest())
    return p
def chunk_files(c):
    assert c not in exam_chunks and ci[ci.chunk == c].split.iloc[0] == "val"
    return fetch(f"labels/egomotion.offline/egomotion.offline.chunk_{c:04d}.zip"), fetch(f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{c:04d}.parquet")
for r in rows: chunk_files(int(ci.loc[r["uuid"], "chunk"]))
used = {int(ci.loc[r["uuid"], "chunk"]) for r in rows}; cand = sorted(set(ci[ci.split == "val"].chunk) - exam_chunks - used); picked = 0
for c in rng.sample(cand, len(cand)):
    if picked == 12: break
    z = zipfile.ZipFile(chunk_files(c)[0]); good = []
    for m in z.infolist():
        u = m.filename.split(".")[0]
        if u not in ok or u not in VV["raw_val"] or u in VV["nurec_val"]: continue
        d = pd.read_parquet(io.BytesIO(z.read(m))); S = K.build_ego(d["timestamp"].to_numpy() / 1e6, d[["x", "y"]].to_numpy(), K.yaw_q(d.qx.to_numpy(), d.qy.to_numpy(), d.qz.to_numpy(), d.qw.to_numpy()))
        sp = S["speed"]; held = max((sp[i:i + 46].min() for i in range(0, len(sp) - 45)), default=0.0)
        if not K.moveoffs(S, u) and held < 16.0 and sp.max() >= 3.0: good.append(u)
    if good: rows.append(dict(uuid=rng.choice(sorted(good)), source="raw", type="ordinary")); picked += 1
df = pd.DataFrame(rows); df["chunk"] = df.uuid.map(ci.chunk); df["split"] = df.uuid.map(ci.split); df["country"] = df.uuid.map(dc.country); assert df.uuid.is_unique and len(df) == 20 and df.chunk.is_unique
C.assert_train(df.uuid); C.assert_train(df[df.source == "nurec"].uuid, nurec=True); tr = pd.read_csv(C.CODE / "clips.csv"); assert not (set(df.chunk) & set(tr.chunk)) and not (set(df.uuid) & set(tr.uuid))
df.to_csv(C.HERE / "clips.csv", index=False); print("eligible nurec_val: A", len(elig(A)), "close-pass", len(elig(P)), "ordinary", len(elig(O))); print(df.to_string())
