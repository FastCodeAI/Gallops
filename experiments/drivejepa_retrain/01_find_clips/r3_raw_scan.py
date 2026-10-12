"""3b/3c stage 2a: ego-only move-off scan over every raw_train clip with egomotion.offline. -> r3_raw_moveoffs_ego.csv, r3_raw_clips.csv"""
import io, zipfile, multiprocessing as mp, numpy as np, pandas as pd, pyarrow.parquet as pq, common, r3_core as K
V = common.load_split_views(); _, EXAM, _ = common.load_views()
CI = pd.read_parquet(common.RAW_SNAP / "clip_index.parquet")
def ego_from_bytes(b):
    d = pq.read_table(io.BytesIO(b)).to_pandas()
    return K.build_ego(d["timestamp"].to_numpy() / 1e6, d[["x", "y"]].to_numpy(), K.yaw_q(d.qx.to_numpy(), d.qy.to_numpy(), d.qz.to_numpy(), d.qw.to_numpy()))
def members(c):
    p = common.RAW_NAS / f"labels/egomotion.offline/egomotion.offline.chunk_{c:04d}.zip"
    if p.exists():
        z = zipfile.ZipFile(p)
        for m in z.infolist():
            u = m.filename.split(".")[0]
            if u in EXAM: continue
            yield u, z.read(m)
    else:
        d = common.RAW_NAS / f"labels/egomotion.offline/egomotion.offline.chunk_{c:04d}.zip.members"
        for f in sorted(d.glob("*.parquet")): yield f.name.split(".")[0], f.read_bytes()
def chunk(c):
    clips, evs = [], []
    for u, b in members(c):
        assert u not in EXAM
        if u not in V["raw_train"]: clips.append(dict(uuid=u, chunk=c, status="not_raw_train")); continue
        try: S = ego_from_bytes(b)
        except Exception as e: clips.append(dict(uuid=u, chunk=c, status="error:" + repr(e)[:60])); continue
        ev = K.moveoffs(S, u)
        for e in ev: e["chunk"] = c; e["stop_xy_yaw"] = None
        clips.append(dict(uuid=u, chunk=c, status="ok", dur=S["dur"], vmax=float(S["speed"].max()), moveoffs=len(ev))); evs += ev
    return clips, evs
if __name__ == "__main__":
    chunks = sorted(set(CI.loc[sorted(V["raw_train"])].chunk))
    with mp.Pool(12) as p: res = p.map(chunk, chunks, chunksize=4)
    cl = pd.DataFrame([x for r in res for x in r[0]]); ev = pd.DataFrame([x for r in res for x in r[1]]).drop(columns=["stop_xy_yaw"])
    cl.to_csv(common.HERE / "r3_raw_clips.csv", index=False); ev.to_csv(common.HERE / "r3_raw_moveoffs_ego.csv", index=False)
    print("chunks", len(chunks), "clips", len(cl), cl.status.str[:5].value_counts().to_dict(), "move-offs", len(ev), "lat>=1", int((ev.lat4 >= 1.0).sum()))
