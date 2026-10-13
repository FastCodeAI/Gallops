"""A1 download, S1 path (same members and range-read mechanism as s1_download.py, run in parallel and resumable): per clip
<clip>.camera_front_wide_120fov.mp4 + .timestamps.parquet from the camera chunk zip; per chunk the two offline calibration parquets."""
import sys, time, hashlib, requests, concurrent.futures as cf, pandas as pd, s3_common as C
df = pd.read_csv(C.HERE / "clips.csv"); C.assert_train(df.uuid); tok = C.token()
url = lambda p: f"https://huggingface.co/datasets/{C.RAW_REPO}/resolve/main/{p}"
def chunk_job(item):
    ch, g = item; out = []; zp = f"camera/{C.FRONT}/{C.FRONT}.chunk_{ch:04d}.zip"; f = zf = None
    for feat in ("camera_intrinsics.offline", "sensor_extrinsics.offline"):
        rp = f"calibration/{feat}/{feat}.chunk_{ch:04d}.parquet"; p = C.S1ROOT / rp
        if not p.exists():
            for a in range(4):
                try: b = requests.get(url(rp), headers={"Authorization": f"Bearer {tok}"}, timeout=120); b.raise_for_status(); break
                except Exception:
                    if a == 3: raise
                    time.sleep(5 * (a + 1))
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b.content); C.manifest_add(f"chunk_{ch:04d}", rp, len(b.content), str(p), hashlib.sha256(b.content).hexdigest())
    for u in g.uuid:
        d = C.S1ROOT / "clips" / u; t = time.time(); nb = 0
        if (d / "DONE").exists(): continue
        d.mkdir(parents=True, exist_ok=True)
        for suf in ("mp4", "timestamps.parquet"):
            name = f"{u}.{C.FRONT}.{suf}"; p = d / name
            for a in range(4):
                try:
                    if zf is None: f, zf = C.hfrange.open_zip(url(zp), tok)
                    data, _ = C.hfrange.fetch_member(zf, f, name); break
                except Exception as e:
                    zf = None
                    if a == 3: print("FAILED", u, repr(e)[:100], flush=True); data = None
                    else: time.sleep(5 * (a + 1))
            if data is None: break
            p.write_bytes(data); nb += len(data); C.manifest_add(u, f"{zp}::{name}", len(data), str(p), hashlib.sha256(data).hexdigest())
        else:
            (d / "DONE").write_text(f"{nb},{time.time() - t:.2f}"); out.append((u, nb, time.time() - t))
    return out
t0 = time.time(); n = tot = 0; jobs = list(df.groupby("chunk"))
with cf.ThreadPoolExecutor(6) as ex:
    for res in ex.map(chunk_job, jobs):
        for u, nb, s in res: n += 1; tot += nb
        if n and n % 200 < len(res): print(f"{n} clips {tot/1e9:.2f} GB {(time.time()-t0)/60:.1f} min", flush=True)
done = sum((C.S1ROOT / "clips" / u / "DONE").exists() for u in df.uuid)
print(f"FINISHED this run: {n} clips, {tot/1e9:.2f} GB, {(time.time()-t0)/60:.1f} min wall, 6 workers; total clips with DONE {done}/{len(df)}", flush=True)
