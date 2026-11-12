"""B4b: fetch obstacle.offline members (agent tracks) for the 3,000 ordinary raw training clips of the slice.
Reuses the census range-reader (s0_census/hfrange.py open_zip/fetch_member, r3_rawio.url) exactly as r3_raw_actors.py:25-38 does.
Writes ONLY under {dataset_store}/approach3_tracks/ (same layout as the census: obstacle.offline.members/chunk_NNNN/<uuid>.obstacle.offline.parquet)
plus its own download_manifest.csv there. Never calls common.manifest_add (that would write into the read-only census folder).
Usage: b4b_fetch.py plan|fetch"""
import os, sys, csv, time, json, hashlib, threading, concurrent.futures as cf, pandas as pd, numpy as np
sys.path.insert(0, "{paddock_sid}/Alpasim/analysis_3.0/Retrain-data/s0_census"); import common, hfrange, r3_rawio as rio
RD="{paddock_sid}/Alpasim/analysis_3.0/Retrain-data"; HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","out")
DST="{dataset_store}/approach3_tracks"; MAN=f"{DST}/download_manifest.csv"; LOCK=threading.Lock()
PER_CLIP_KB=561  # census probe, RD/s0_census/REPORT.md:527
mode=sys.argv[1]
sl=json.load(open(f"{OUT}/slice_clips.json")); tr=pd.read_parquet(f"{RD}/s3_assembly/manifest.parquet"); tr['chunk']=tr.group.str.replace('chunk_','').astype(int)
c=tr.drop_duplicates('clip_uuid'); assert set(c.clip_uuid)==set(sl['train']), "manifest clips != slice_clips.json train"
ordn=c[(c.source=='raw')&(c.type=='ordinary')][['clip_uuid','chunk']].sort_values(['chunk','clip_uuid']).reset_index(drop=True)
_, EXAM, _ = common.load_views(); assert not (set(ordn.clip_uuid) & EXAM), "exam clip in the ordinary list"
def dst(u,ch): return f"{DST}/obstacle.offline.members/chunk_{ch:04d}/{u}.obstacle.offline.parquet"
ordn['have']=[os.path.exists(dst(u,ch)) for u,ch in zip(ordn.clip_uuid, ordn.chunk)]
todo=ordn[~ordn.have]; print(f"ordinary raw training clips: {len(ordn)} in {ordn.chunk.nunique()} chunks; already fetched: {int(ordn.have.sum())}; to fetch: {len(todo)} clips in {todo.chunk.nunique()} chunks")
print(f"planned download: {len(todo)} clips x {PER_CLIP_KB} KB = {len(todo)*PER_CLIP_KB/1e6:.2f} GB (census per-clip figure); target df:"); os.system(f"df -h {DST} | tail -1")
if len(todo)>3000 or len(todo)*PER_CLIP_KB/1e6>5.0: print("STOP: plan exceeds 5 GB or 3,000 clips"); sys.exit(2)
if mode=="plan": sys.exit(0)
def man_add(u, member, n, p, sha):
    with LOCK:
        new=not os.path.exists(MAN)
        with open(MAN,"a",newline="") as f:
            w=csv.writer(f); 
            if new: w.writerow(["uuid","file","bytes","path","sha256","ts"])
            w.writerow([u, member, n, p, sha, time.strftime("%Y-%m-%dT%H:%M:%S")])
def chunk(args):
    ch, uuids = args; path=f"labels/obstacle.offline/obstacle.offline.chunk_{ch:04d}.zip"; zf=f=None; res=[]
    os.makedirs(f"{DST}/obstacle.offline.members/chunk_{ch:04d}", exist_ok=True)
    for u in uuids:
        assert u not in EXAM; name=f"{u}.obstacle.offline.parquet"; p=dst(u,ch); data=None; err=""
        for a in range(4):
            try:
                if zf is None: f, zf = hfrange.open_zip(rio.url(path), os.environ["HF_TOKEN"])
                data,_=hfrange.fetch_member(zf, f, name); break
            except KeyError as e: err=f"member not in zip: {e!r}"; break
            except Exception as e:
                err=repr(e)[:200]; zf=None; time.sleep(3*(a+1))
        if data is None: res.append(dict(clip_uuid=u, chunk=ch, status="FAIL", bytes=0, error=err)); continue
        tmp=p+".part"; open(tmp,"wb").write(data); os.replace(tmp,p); man_add(u, f"{path}::{name}", len(data), p, hashlib.sha256(data).hexdigest())
        res.append(dict(clip_uuid=u, chunk=ch, status="ok", bytes=len(data), error=""))
    return res
jobs=[(ch, list(g.clip_uuid)) for ch,g in todo.groupby('chunk')]; t0=time.time(); rows=[]; nb=0
with cf.ThreadPoolExecutor(8) as ex:
    for i,r in enumerate(ex.map(chunk, jobs)):
        rows+=r; nb+=sum(x['bytes'] for x in r)
        if (i+1)%100==0: print(f"{i+1}/{len(jobs)} chunks, {len(rows)} clips, {nb/1e6:.0f} MB, {time.time()-t0:.0f}s", flush=True)
df=pd.DataFrame(rows); df.to_csv(f"{OUT}/b4b_fetch_status.csv", index=False)
print(f"DONE fetch: {int((df.status=='ok').sum())} ok, {int((df.status=='FAIL').sum())} failed, {nb/1e9:.3f} GB in {time.time()-t0:.0f}s")
if (df.status=='FAIL').any(): print(df[df.status=='FAIL'].to_string())
