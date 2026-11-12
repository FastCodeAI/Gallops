"""Range-read the usdz members the map/agents need (never the full usdz): every clipgt/*.parquet not already in the S0 cache,
map.xodr, and for pilot scenes rig_trajectories.json + sequence_tracks.json. NAS only, sha256 manifest."""
import hashlib, time, concurrent.futures as cf, pandas as pd, s2_common as C
sc = pd.read_csv(C.HERE / "scenes.csv"); C.assert_nurec_train(sc.uuid); tok = C.token()
S0_HAVE = {"clipgt/egomotion_estimate.parquet", "clipgt/obstacle.parquet", "clipgt/clip.parquet"}     # in S0 NAS pool/<uuid>/
def one(r):
    u = r.uuid; d = C.NAS / "scenes" / u; d.mkdir(parents=True, exist_ok=True)
    if (d / "DONE").exists(): return u, 0, 0.0
    t = time.time(); f, zf = C.hfrange.open_zip(f"https://huggingface.co/datasets/{C.NUREC_REPO}/resolve/main/{C.RELEASE}/{u}/{u}.usdz", tok)
    want = [m.filename for m in zf.infolist() if (m.filename.startswith("clipgt/") and m.filename not in S0_HAVE) or m.filename == "map.xodr"]
    if r.pilot: want += ["rig_trajectories.json", "sequence_tracks.json"]
    for n in want:
        data, _ = C.hfrange.fetch_member(zf, f, n); p = d / n.replace("/", "__"); p.write_bytes(data)
        C.manifest_add(u, n, len(data), str(p), hashlib.sha256(data).hexdigest())
    (d / "DONE").write_text(str(f.bytes_fetched)); return u, f.bytes_fetched, time.time() - t
with cf.ThreadPoolExecutor(6) as ex: res = list(ex.map(one, sc.itertuples()))
print("scenes", len(res), "MB", sum(r[1] for r in res) / 1e6, "mean s", sum(r[2] for r in res) / max(len(res), 1))
