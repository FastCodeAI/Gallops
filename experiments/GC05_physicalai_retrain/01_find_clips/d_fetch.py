"""STEP D fetch stage (NOT RUN beyond the 3 pilot scenes until the STEP B gate is resolved).
Range-reads egomotion/obstacle/lane parquet + metadata.yaml + data_info.json for every pool scene into NAS/pool/<uuid>/.
Usage: d_fetch.py [--limit N] [--workers K]"""
import os, sys, json, time, hashlib, argparse, concurrent.futures as cf, common, hfrange
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--pool", default=str(common.HERE/"training_pool_candA.txt"))
a = ap.parse_args()
nurec_excl, excl, _pool = common.load_views()
pool = sorted(_pool)
assert not (set(pool) & excl), "pool overlaps exam exclusions"
if a.limit: pool = pool[:a.limit]
WANT = ["clipgt/egomotion_estimate.parquet", "clipgt/obstacle.parquet", "clipgt/clip.parquet", "metadata.yaml", "data_info.json"]
tok = os.environ["HF_TOKEN"]
def one(u):
    for attempt in range(4):
        try: return _one(u)
        except AssertionError: raise
        except Exception as e:
            err = e; time.sleep(5 * (attempt + 1))
    print("FAILED", u, repr(err), flush=True); open(common.HERE / "d_fetch_failed.txt", "a").write(u + "\n"); return u, 0, 0, 0.0
def _one(u):
    assert u not in excl, f"{u} is exam"
    d = common.NAS / "pool" / u
    if (d / "DONE").exists(): return u, 0, 0, 0.0
    d.mkdir(parents=True, exist_ok=True); t0 = time.time()
    url = f"https://huggingface.co/datasets/{common.REPO}/resolve/main/{common.RELEASE}/{u}/{u}.usdz"
    f, zf = hfrange.open_zip(url, tok, readahead=64*1024)
    names = {m.filename for m in zf.infolist()}
    for w in WANT:
        if w not in names: continue
        data, zi = hfrange.fetch_member(zf, f, w)
        p = d / w.replace("/", "__"); p.write_bytes(data)
        common.manifest_add(u, w, len(data), str(p), hashlib.sha256(data).hexdigest())
    (d / "DONE").write_text(json.dumps(dict(usdz_size=f.size, bytes=f.bytes_fetched, requests=f.requests, secs=round(time.time()-t0,1))))
    return u, f.bytes_fetched, f.requests, time.time()-t0
t0 = time.time(); tot = 0
with cf.ThreadPoolExecutor(a.workers) as ex:
    for i, (u, b, r, s) in enumerate(ex.map(one, pool)):
        tot += b
        if i % 25 == 0 or b: print(f"{i+1}/{len(pool)} {u[:8]} {b/1e6:.2f} MB {r} req {s:.1f} s | cum {tot/1e9:.2f} GB {(time.time()-t0)/60:.1f} min", flush=True)
