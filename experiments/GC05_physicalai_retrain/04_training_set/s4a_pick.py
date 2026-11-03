"""S4 Part A1: full validation pick (seed 20261008). Run with S3_SPLIT=val_full.
nurec: EVERY eligible nurec_val scene (same eligibility as s31_pick_val.py), typed like s3_pick.py does for train (A = S0 Run 3 high-speed flag;
B_pass = any around-still-obstacle event, not A; else ordinary).
raw: a seeded 400 of the eligible raw_val clips that are not nurec_val, drawn BEFORE typing; then scanned with the S0 Run 3 detector
(r3_core.moveoffs + attach_actor through r3_raw_actors.chunk, the code that produced the 285 train pull-outs): B_raw = a move-off with a
stationary actor ahead of the bumper (actor_bum) and |heading change| <= 45 deg; else A if the clip holds >= 16 m/s for a 4.5 s window
(s3_pick.scan's rule); else ordinary. Fetches the egomotion package + vehicle-dimension file of every val chunk used."""
import io, zipfile, random, hashlib, requests, concurrent.futures as cf, numpy as np, pandas as pd
import s3_common as C; import r3_core as K
assert C.SPLIT == "val" and C.VAL_DIR == "val_full"; V = C.views(); VV, exam, ci = V["V"], V["exam"], V["ci"]; tok = C.token()
C.S0C.manifest_add = C.manifest_add                       # the S0 detector's obstacle-member downloads are logged in the val_full manifest
import r3_raw_actors as RA
fp = pd.read_parquet(C.S0C.RAW_SNAP / "metadata/feature_presence.parquet"); dc = pd.read_parquet(C.S0C.RAW_SNAP / "metadata/data_collection.parquet")
need = [C.FRONT, "camera_intrinsics.offline", "sensor_extrinsics.offline", "egomotion.offline", "obstacle.offline", "vehicle_dimensions"]
ok = set(fp.index[fp[need].all(axis=1)]); exam_chunks = set(ci.loc[[u for u in exam if u in ci.index]].chunk); tr = pd.read_csv(C.CODE / "clips.csv"); train_chunks = set(tr.chunk)
elig = lambda us: sorted(u for u in us if u in VV["raw_val"] and u in ok and ci.loc[u, "chunk"] not in exam_chunks and ci.loc[u, "chunk"] not in train_chunks)
an = pd.read_csv(C.S0 / "d3_anchors.csv", low_memory=False); ar = pd.read_csv(C.S0 / "d3_around.csv")
A = set(an[(an.A == True) & (an.split == "val")].uuid); P = set(ar[ar.split == "val"].uuid) - A
nurec = elig(VV["nurec_val"]); rows = [dict(uuid=u, source="nurec", type="A" if u in A else ("B_pass" if u in P else "ordinary")) for u in nurec]
pool = elig(VV["raw_val"] - VV["nurec_val"]); rng = random.Random(20261008); raw = sorted(rng.sample(pool, 400))
def fetch(path):
    p = C.S0C.RAW_NAS / path
    if not p.exists():
        for a in range(4):
            try: r = requests.get(f"https://huggingface.co/datasets/{C.RAW_REPO}/resolve/main/{path}", headers={"Authorization": f"Bearer {tok}"}, timeout=300); r.raise_for_status(); break
            except Exception:
                if a == 3: raise
        p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(p.suffix + ".part"); tmp.write_bytes(r.content); tmp.rename(p); C.manifest_add(path.split("/")[-1].split(".")[-2], path, len(r.content), str(p), hashlib.sha256(r.content).hexdigest())
    return p
def chunk_files(c):
    assert c not in exam_chunks and c not in train_chunks and ci[ci.chunk == c].split.iloc[0] == "val"
    return fetch(f"labels/egomotion.offline/egomotion.offline.chunk_{c:04d}.zip"), fetch(f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{c:04d}.parquet")
chunks = sorted({int(ci.loc[u, "chunk"]) for u in nurec + raw})
with cf.ThreadPoolExecutor(8) as ex: list(ex.map(chunk_files, chunks))
print("chunks", len(chunks), "egomotion + vehicle dimensions present", flush=True)
# ---- scan the 400 raw clips
clips, evs = [], []
for c, us in pd.Series(raw).groupby(ci.loc[raw, "chunk"].values):
    z = zipfile.ZipFile(C.S0C.RAW_NAS / f"labels/egomotion.offline/egomotion.offline.chunk_{int(c):04d}.zip")
    for u in us:
        assert u not in exam
        d = pd.read_parquet(io.BytesIO(z.read(f"{u}.egomotion.offline.parquet"))); S = K.build_ego(d["timestamp"].to_numpy() / 1e6, d[["x", "y"]].to_numpy(), K.yaw_q(d.qx.to_numpy(), d.qy.to_numpy(), d.qz.to_numpy(), d.qw.to_numpy()))
        sp = S["speed"]; held = max((sp[i:i + 46].min() for i in range(0, len(sp) - 45)), default=0.0); ev = K.moveoffs(S, u)
        for e in ev: e["chunk"] = int(c); e.pop("stop_xy_yaw", None)
        clips.append(dict(uuid=u, chunk=int(c), dur=S["dur"], vmax=float(sp.max()), held=float(held), moveoffs=len(ev))); evs += ev
cl = pd.DataFrame(clips).set_index("uuid"); ev = pd.DataFrame(evs); cand = ev[ev.lat4 >= 1.0] if len(ev) else ev
print("raw clips scanned", len(cl), "| with a move-off", int((cl.moveoffs > 0).sum()), "| move-offs", len(ev), "| lateral >= 1 m", len(cand), "in", cand.uuid.nunique() if len(cand) else 0, "clips", flush=True)
res = []
if len(cand):
    with cf.ThreadPoolExecutor(8) as ex:
        for r in ex.map(RA.chunk, [(c, g) for c, g in cand.groupby("chunk")]): res += r
mo = pd.DataFrame(res).drop(columns=["_i", "_j", "_k"], errors="ignore"); mo.to_csv(C.HERE / "raw_moveoffs.csv", index=False); cl.reset_index().to_csv(C.HERE / "raw_scan.csv", index=False)
pull = set(mo[(mo.actor_bum == True) & (mo.dyaw4_deg.abs() <= 45)].uuid) if len(mo) else set()
print("actor attachment:", mo.actor_status.value_counts().to_dict() if len(mo) else {}, "| pull-outs (actor_bum, |dyaw| <= 45):", len(pull), "| actor_bum before the turn filter:", int((mo.actor_bum == True).sum()) if len(mo) else 0, flush=True)
rows += [dict(uuid=u, source="raw", type="B_raw" if u in pull else ("A" if cl.loc[u, "held"] >= 16.0 else "ordinary")) for u in raw]
df = pd.DataFrame(rows); df["chunk"] = df.uuid.map(ci.chunk); df["split"] = df.uuid.map(ci.split); df["country"] = df.uuid.map(dc.country); assert df.uuid.is_unique and len(df) == len(nurec) + 400
C.assert_train(df.uuid); C.assert_train(df[df.source == "nurec"].uuid, nurec=True); assert not (set(df.chunk) & train_chunks) and not (set(df.uuid) & set(tr.uuid)) and not (set(df.uuid) & exam) and not (set(df.uuid) & V["test"])
df.to_csv(C.HERE / "clips.csv", index=False)
sm = pd.read_csv(C.CODE / "val/clips.csv"); print("seed 20261008 | eligible nurec_val", len(nurec), "| raw_val pool", len(pool), "| chunks", df.chunk.nunique(), "| clips also in the smoke val set:", int(df.uuid.isin(sm.uuid).sum()))
print(df.groupby(["source", "type"]).size().to_string()); print("raw: still the whole clip (vmax < 3 m/s):", int((cl.vmax < 3).sum()), "| countries", df.country.nunique())
