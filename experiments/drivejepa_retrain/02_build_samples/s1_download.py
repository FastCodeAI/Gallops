"""STEP 2b: per pilot clip, range-read ONLY <clip>.camera_front_wide_120fov.mp4 + .timestamps.parquet from the camera chunk zip
(features.csv: camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_NNNN.zip), plus the chunk's
calibration/camera_intrinsics.offline and sensor_extrinsics.offline parquets. NAS only, sha256 manifest."""
import time, hashlib, requests, pandas as pd, s1_common as C
df = pd.read_csv(C.HERE / "pilot_clips.csv"); C.assert_train(df.uuid); tok = C.token(); rows = []
url = lambda p: f"https://huggingface.co/datasets/{C.RAW_REPO}/resolve/main/{p}"
for r in df.itertuples():
    d = C.NAS / "clips" / r.uuid; d.mkdir(parents=True, exist_ok=True); t = time.time(); nb = {}
    zp = f"camera/{C.FRONT}/{C.FRONT}.chunk_{r.chunk:04d}.zip"; f = zf = None
    for suf in ("mp4", "timestamps.parquet"):
        name = f"{r.uuid}.{C.FRONT}.{suf}"; p = d / name
        if not p.exists():
            if zf is None: f, zf = C.hfrange.open_zip(url(zp), tok)
            data, zi = C.hfrange.fetch_member(zf, f, name); p.write_bytes(data)
            C.manifest_add(r.uuid, f"{zp}::{name}", len(data), str(p), hashlib.sha256(data).hexdigest())
        nb[suf] = p.stat().st_size
    for feat in ("camera_intrinsics.offline", "sensor_extrinsics.offline"):
        rp = f"calibration/{feat}/{feat}.chunk_{r.chunk:04d}.parquet"; p = C.NAS / rp
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True); b = requests.get(url(rp), headers={"Authorization": f"Bearer {tok}"}, timeout=120); b.raise_for_status()
            p.write_bytes(b.content); C.manifest_add(f"chunk_{r.chunk:04d}", rp, len(b.content), str(p), hashlib.sha256(b.content).hexdigest())
        nb[feat] = p.stat().st_size
    rows.append(dict(type=r.type, uuid=r.uuid, chunk=r.chunk, mp4_MB=nb["mp4"] / 1e6, ts_KB=nb["timestamps.parquet"] / 1e3, calib_KB=(nb["camera_intrinsics.offline"] + nb["sensor_extrinsics.offline"]) / 1e3,
                     zip_GB=(f.size / 1e9 if f else None), secs=round(time.time() - t, 2), fetched_MB=(f.bytes_fetched / 1e6 if f else 0), reqs=(f.requests if f else 0)))
    print(rows[-1], flush=True)
o = pd.DataFrame(rows); o.to_csv(C.HERE / "download_stats.csv", index=False)
mb = o.mp4_MB.mean(); sec = o.secs.mean(); print(f"mean {mb:.1f} MB/clip (min {o.mp4_MB.min():.1f}, max {o.mp4_MB.max():.1f}); {sec:.1f} s/clip serial; {o.fetched_MB.sum()/o.secs.sum():.1f} MB/s")
for name, n in [("285 pull-outs", 285), ("208 A scenes", 208), ("5,000 ordinary", 5000)]:
    print(f"budget {name}: {n*mb/1e3:.1f} GB, {n*sec/3600:.2f} h serial, {n*sec/3600/6:.2f} h with 6 workers")
