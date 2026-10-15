"""3b stage 1: egomotion.offline zip + vehicle_dimensions parquet for every raw_train chunk (to NAS, manifest with sha256).
The one train chunk that contains an exam clip is read member-by-member (range reads), skipping the exam member; its
vehicle_dimensions parquet is NOT downloaded (it carries a row for the exam clip)."""
import os, sys, time, hashlib, concurrent.futures as cf, pandas as pd, common, hfrange, r3_rawio as io
V = common.load_split_views(); _, exam, _ = common.load_views()
ci = pd.read_parquet(common.RAW_SNAP / "clip_index.parquet")
chunks = sorted(set(ci.loc[sorted(V["raw_train"])].chunk)); exam_chunks = set(ci.loc[[u for u in exam if u in ci.index]].chunk)
assert all(ci[ci.chunk == c].split.iloc[0] == "train" for c in chunks[:50])
def one(c):
    pe = f"labels/egomotion.offline/egomotion.offline.chunk_{c:04d}.zip"; pv = f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{c:04d}.parquet"
    if c in exam_chunks:
        d = common.RAW_NAS / (pe + ".members")
        if (d / "DONE").exists(): return c, 0
        d.mkdir(parents=True, exist_ok=True); f, zf = hfrange.open_zip(io.url(pe), os.environ["HF_TOKEN"]); n = 0
        for m in zf.infolist():
            u = m.filename.split(".")[0]
            if u in exam: continue                       # exam member never read
            assert u in V["raw_train"]
            data, _ = hfrange.fetch_member(zf, f, m.filename); (d / m.filename).write_bytes(data); n += len(data)
            common.manifest_add(u, f"{pe}::{m.filename}", len(data), str(d / m.filename), hashlib.sha256(data).hexdigest())
        (d / "DONE").write_text(str(n)); return c, n
    n1, _ = io.download(pe, f"chunk_{c:04d}"); n2, _ = io.download(pv, f"chunk_{c:04d}"); return c, n1 + n2
t = time.time(); tot = 0
with cf.ThreadPoolExecutor(8) as ex:
    for i, (c, n) in enumerate(ex.map(one, chunks)):
        tot += n
        if i % 100 == 0 or i == len(chunks) - 1: print(f"{i+1}/{len(chunks)} chunk {c} cum {tot/1e9:.2f} GB {(time.time()-t)/60:.1f} min", flush=True)
print("exam chunks handled member-wise:", sorted(exam_chunks & set(chunks)))
