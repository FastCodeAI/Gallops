"""NuRec map members for every nurec scene in clips.csv (same members as s2_fetch.py; the 40 S2 scenes are copied from the S2 NAS root)."""
import hashlib, time, shutil, concurrent.futures as cf, pandas as pd, s3_common as C
df = pd.read_csv(C.HERE / "clips.csv"); df = df[df.source == "nurec"]; C.assert_train(df.uuid, nurec=True); tok = C.token()
S2SC = pathlib = None
import pathlib
S2SC = pathlib.Path("{runs}/drivejepa_retrain/03_grade_candidates/scenes"); HAVE = {"clipgt/egomotion_estimate.parquet", "clipgt/obstacle.parquet", "clipgt/clip.parquet"}
def one(u):
    d = C.SCENES / u
    if (d / "DONE").exists(): return 0
    if (S2SC / u / "DONE").exists():
        shutil.copytree(S2SC / u, d, dirs_exist_ok=True); return 0
    d.mkdir(parents=True, exist_ok=True)
    for a in range(4):
        try:
            f, zf = C.hfrange.open_zip(f"https://huggingface.co/datasets/{C.S0C.REPO}/resolve/main/{C.S0C.RELEASE}/{u}/{u}.usdz", tok)
            for n in [m.filename for m in zf.infolist() if m.filename.startswith("clipgt/") and m.filename not in HAVE]:
                p = d / n.replace("/", "__")
                if p.exists(): continue
                data, _ = C.hfrange.fetch_member(zf, f, n); p.write_bytes(data); C.manifest_add(u, n, len(data), str(p), hashlib.sha256(data).hexdigest())
            (d / "DONE").write_text(str(f.bytes_fetched)); return f.bytes_fetched
        except Exception as e:
            if a == 3: print("FAILED", u, repr(e)[:100], flush=True); return -1
            time.sleep(5 * (a + 1))
t0 = time.time()
with cf.ThreadPoolExecutor(4) as ex: res = list(ex.map(one, df.uuid))
print(f"scenes {len(res)} failed {sum(r < 0 for r in res)} fetched {sum(max(r, 0) for r in res)/1e6:.1f} MB {(time.time()-t0)/60:.1f} min", flush=True)
