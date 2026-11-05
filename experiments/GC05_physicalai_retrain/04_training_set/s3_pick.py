"""A1/A3 clip lists. Types (per clip, fixed precedence): A = nurec_train scene holding >= 16 m/s for a 4.5 s window (the 208 of S0 Run 3);
B_pass = nurec_train scene with an around-still-obstacle event (S0 Run 3 Part 2), not A; B_raw = the 285 raw pull-out clips;
ordinary = remaining nurec_train scenes + a seeded sample of 3,000 raw_train clips with no A/B event, spread over countries."""
import io, zipfile, random, multiprocessing as mp, numpy as np, pandas as pd
import s3_common as C; import r3_core as K
V = C.views(); VV, exam, ci = V["V"], V["exam"], V["ci"]
fp = pd.read_parquet(C.S0C.RAW_SNAP / "metadata/feature_presence.parquet"); dc = pd.read_parquet(C.S0C.RAW_SNAP / "metadata/data_collection.parquet")
need = [C.FRONT, "camera_intrinsics.offline", "sensor_extrinsics.offline", "egomotion.offline", "vehicle_dimensions"]
ok = set(fp.index[fp[need].all(axis=1)]); exam_chunks = set(ci.loc[[u for u in exam if u in ci.index]].chunk)
elig = lambda us: {u for u in us if u in VV["raw_train"] and u in ok and ci.loc[u, "chunk"] not in exam_chunks}
def scan(c):
    out = []; z = zipfile.ZipFile(C.S0C.RAW_NAS / f"labels/egomotion.offline/egomotion.offline.chunk_{c:04d}.zip")
    for m in z.infolist():
        u = m.filename.split(".")[0]
        if u in exam: continue
        d = pd.read_parquet(io.BytesIO(z.read(m))); S = K.build_ego(d["timestamp"].to_numpy() / 1e6, d[["x", "y"]].to_numpy(), K.yaw_q(d.qx.to_numpy(), d.qy.to_numpy(), d.qz.to_numpy(), d.qw.to_numpy()))
        sp = S["speed"]; held = max((sp[i:i + 46].min() for i in range(0, len(sp) - 45)), default=0.0); out.append((u, float(held)))
    return out
if __name__ == "__main__":
    d3a = pd.read_csv(C.S0 / "d3_anchors.csv", low_memory=False); ar = pd.read_csv(C.S0 / "d3_around.csv"); r = pd.read_csv(C.S0 / "r3_raw_moveoffs.csv"); rc = pd.read_csv(C.S0 / "r3_raw_clips.csv").set_index("uuid")
    A_all = set(d3a[(d3a.A == True) & (d3a.split == "train")].uuid); assert len(A_all) == 208
    P_all = set(ar[ar.split == "train"].uuid) - A_all; Braw_all = set(r[(r.actor_bum == True) & (r.dyaw4_deg.abs() <= 45)].uuid); assert len(Braw_all) == 285
    nurec = elig(VV["nurec_train"]); lost_nurec = VV["nurec_train"] - nurec
    rows = [dict(uuid=u, source="nurec", type="A" if u in A_all else ("B_pass" if u in P_all else "ordinary")) for u in sorted(nurec)]
    braw = elig(Braw_all) - nurec; rows += [dict(uuid=u, source="raw", type="B_raw") for u in sorted(braw)]
    # ordinary raw pool: no move-off, moves at all (vmax >= 3 m/s), and never holds >= 16 m/s for 4.5 s (ego-only scan of the S0 egomotion cache)
    cand = sorted(elig(set(rc.index[(rc.status == "ok") & (rc.moveoffs == 0) & (rc.vmax >= 3.0)])) - nurec - Braw_all)
    need_scan = sorted(set(ci.loc[[u for u in cand if rc.loc[u, "vmax"] >= 16.0]].chunk))
    with mp.Pool(8) as p: held = dict(x for res in p.map(scan, need_scan, chunksize=4) for x in res)
    pool = [u for u in cand if held.get(u, 0.0) < 16.0]; pc = pd.Series(dc.loc[pool, "country"].values, index=pool)
    n = 3000; cnt = pc.value_counts(); alloc = (cnt / cnt.sum() * n).round().astype(int).clip(lower=np.minimum(100, cnt)); alloc = (alloc * n / alloc.sum()).round().astype(int)
    while alloc.sum() != n: alloc[alloc.idxmax()] += n - alloc.sum()
    rng = random.Random(20261006); pick = []
    for country in sorted(alloc.index): pick += rng.sample(sorted(pc.index[pc == country]), int(min(alloc[country], cnt[country])))
    rows += [dict(uuid=u, source="raw", type="ordinary") for u in sorted(pick)]
    df = pd.DataFrame(rows); df["chunk"] = df.uuid.map(ci.chunk); df["split"] = df.uuid.map(ci.split); df["country"] = df.uuid.map(dc.country); assert df.uuid.is_unique
    C.assert_train(df.uuid); C.assert_train(df[df.source == "nurec"].uuid, nurec=True); df.to_csv(C.HERE / "clips.csv", index=False)
    print("seed 20261006 | ordinary raw pool", len(pool), "of candidates", len(cand), "| nurec_train usable", len(nurec), "lost (feature/exam-chunk)", len(lost_nurec), "| B_raw usable", len(braw), "of 285 (in nurec:", len(Braw_all & VV["nurec_train"]), ")")
    print(df.groupby(["source", "type"]).size().to_string()); print("ordinary raw by country:", df[(df.source == "raw") & (df.type == "ordinary")].country.value_counts().to_dict()); print("clips", len(df), "chunks", df.chunk.nunique())
