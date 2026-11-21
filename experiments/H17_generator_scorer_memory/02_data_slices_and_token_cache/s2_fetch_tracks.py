"""Build step 2c: obstacle.offline members (agent tracks) for the fresh raw held-out's 300 clips, with the census range-reader
(s0_census/hfrange.py open_zip / fetch_member, r3_rawio.url) exactly as b4b_fetch.py:29-44 / r3_raw_actors.py:25-38.
Writes only under {experiment_store}/raw_heldout/raw/labels/obstacle.offline.members/chunk_NNNN/ + download_manifest_tracks.csv."""
import os, sys, csv, time, hashlib, threading, concurrent.futures as cf, pandas as pd
sys.path.insert(0, "{paddock_sid}/Alpasim/analysis_3.0/Retrain-data/s0_census"); import hfrange, r3_rawio as rio
DST="{experiment_store}/raw_heldout"; MAN=f"{DST}/download_manifest_tracks.csv"; LOCK=threading.Lock(); PER_CLIP_KB=561
df=pd.read_csv(f"{DST}/heldout_clips.csv"); print(f"clips {len(df)} chunks {df.chunk.nunique()}; planned {len(df)*PER_CLIP_KB/1e6:.2f} GB (census per-clip figure)")
def dst(u,ch): return f"{DST}/raw/labels/obstacle.offline.members/chunk_{ch:04d}/{u}.obstacle.offline.parquet"
def man_add(u, member, n, p, sha):
    with LOCK:
        new=not os.path.exists(MAN)
        with open(MAN,"a",newline="") as f:
            w=csv.writer(f)
            if new: w.writerow(["uuid","file","bytes","path","sha256","ts"])
            w.writerow([u, member, n, p, sha, time.strftime("%Y-%m-%dT%H:%M:%S")])
def chunk(args):
    ch, uuids = args; path=f"labels/obstacle.offline/obstacle.offline.chunk_{ch:04d}.zip"; zf=f=None; res=[]
    os.makedirs(os.path.dirname(dst(uuids[0],ch)), exist_ok=True)
    for u in uuids:
        name=f"{u}.obstacle.offline.parquet"; p=dst(u,ch); data=None; err=""
        if os.path.exists(p): res.append(dict(clip_uuid=u, chunk=ch, status="present", bytes=os.path.getsize(p), error="")); continue
        for a in range(4):
            try:
                if zf is None: f, zf = hfrange.open_zip(rio.url(path), os.environ["HF_TOKEN"])
                data,_=hfrange.fetch_member(zf, f, name); break
            except KeyError as e: err=f"member not in zip: {e!r}"; break
            except Exception as e: err=repr(e)[:200]; zf=None; time.sleep(3*(a+1))
        if data is None: res.append(dict(clip_uuid=u, chunk=ch, status="FAIL", bytes=0, error=err)); continue
        tmp=p+".part"; open(tmp,"wb").write(data); os.replace(tmp,p); man_add(u, f"{path}::{name}", len(data), p, hashlib.sha256(data).hexdigest())
        res.append(dict(clip_uuid=u, chunk=ch, status="ok", bytes=len(data), error=""))
    return res
jobs=[(int(ch), list(g.uuid)) for ch,g in df.groupby("chunk")]; t0=time.time(); rows=[]
with cf.ThreadPoolExecutor(8) as ex:
    for r in ex.map(chunk, jobs): rows+=r
out=pd.DataFrame(rows); out.to_csv(f"{DST}/tracks_fetch_status.csv", index=False)
print(out.status.value_counts().to_dict(), f"{out.bytes.sum()/1e6:.1f} MB in {time.time()-t0:.0f}s"); print(out[out.status=="FAIL"].head())
