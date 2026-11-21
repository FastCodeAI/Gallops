import pyarrow.parquet as pq, pandas as pd, numpy as np, zipfile, io, time, os, glob
RD="{paddock_sid}/Alpasim/analysis_3.0/Retrain-data/s3_assembly"
P="{dataset_store}"
tr=pd.read_parquet(f"{RD}/manifest.parquet")
clips=tr.drop_duplicates('clip_uuid')[['clip_uuid','group','source','type']].copy()
clips['chunk']=clips.group.str.replace('chunk_','').astype(int) if clips.group.dtype==object else clips.group
print("group sample:", clips.group.head(3).tolist())
sub=clips.sample(20, random_state=0)
t0=time.time(); rows=[]
for r in sub.itertuples():
    z=zipfile.ZipFile(f"{P}/s0_census/raw/labels/egomotion.offline/egomotion.offline.chunk_{r.chunk:04d}.zip")
    df=pd.read_parquet(io.BytesIO(z.read(f"{r.clip_uuid}.egomotion.offline.parquet")))
    ts=np.sort(df.timestamp.values.astype(np.int64)); dt=np.diff(ts)/1e6
    rows.append(dict(uuid=r.clip_uuid[:8], n=len(ts), t0=int(ts[0]), span=(ts[-1]-ts[0])/1e6, dt_med=float(np.median(dt)), dt_max=float(dt.max())))
print("egomotion cols:", list(df.columns)); print(df.head(2).to_string())
print(pd.DataFrame(rows).to_string()); print(f"20 egomotion member reads: {time.time()-t0:.1f}s")
t0=time.time(); rows=[]
for r in sub.itertuples():
    p=f"{P}/s3_assembly/s1/clips/{r.clip_uuid}/{r.clip_uuid}.camera_front_wide_120fov.timestamps.parquet"
    df=pd.read_parquet(p); ts=np.sort(df.timestamp.values.astype(np.int64)); dt=np.diff(ts)/1e6
    rows.append(dict(uuid=r.clip_uuid[:8], n=len(ts), t0=int(ts[0]), span=(ts[-1]-ts[0])/1e6, dt_med=float(np.median(dt)), dt_max=float(dt.max())))
print("frame ts cols:", list(df.columns)); print(pd.DataFrame(rows).to_string()); print(f"20 frame-timestamp reads: {time.time()-t0:.1f}s")
# nurec pool obstacle
nu=clips[clips.source=='nurec'].iloc[0]
d=f"{P}/s0_census/pool/{nu.clip_uuid}"; print("pool dir exists", os.path.isdir(d), os.listdir(d) if os.path.isdir(d) else '')
t0=time.time(); ob=pd.read_parquet(f"{d}/clipgt__obstacle.parquet"); eg=pd.read_parquet(f"{d}/clipgt__egomotion_estimate.parquet")
print(f"pool obstacle read {time.time()-t0:.2f}s rows {len(ob)} cols {list(ob.columns)}"); print(ob.head(3).to_string()[:1500])
print("pool ego cols", list(eg.columns), len(eg)); print(eg.head(2).to_string()[:800])
# B_raw obstacle member
br=clips[clips.type=='B_raw'].iloc[0]
m=f"{P}/s0_census/raw/labels/obstacle.offline.members/chunk_{br.chunk:04d}/{br.clip_uuid}.obstacle.offline.parquet"
print("B_raw member exists", os.path.exists(m))
if os.path.exists(m):
    t0=time.time(); ob2=pd.read_parquet(m); print(f"raw obstacle read {time.time()-t0:.2f}s rows {len(ob2)} cols {list(ob2.columns)}"); print(ob2.head(3).to_string()[:1500])
# how many train clips have an obstacle member on NAS (by type)
cnt={}
for r in clips.itertuples():
    m=f"{P}/s0_census/raw/labels/obstacle.offline.members/chunk_{r.chunk:04d}/{r.clip_uuid}.obstacle.offline.parquet"
    p=f"{P}/s0_census/pool/{r.clip_uuid}/clipgt__obstacle.parquet"
    k=(r.source,r.type); cnt.setdefault(k,[0,0,0]); cnt[k][0]+=1; cnt[k][1]+=os.path.exists(m); cnt[k][2]+=os.path.exists(p)
print("train clips: (source,type): [n, raw obstacle member on NAS, pool clipgt obstacle on NAS]"); print(cnt)
# usdz
us=sorted(glob.glob("{alpasim_store}/nre-artifacts/all-usdzs/*.usdz"))[:1]
t0=time.time(); z=zipfile.ZipFile(us[0]); names=[n for n in z.namelist() if 'clipgt' in n or n.endswith('.json') or n.endswith('.yaml')]; print("usdz members (clipgt/json/yaml):", names[:30])
e=pd.read_parquet(io.BytesIO(z.read('clipgt/egomotion_estimate.parquet'))); print(f"usdz ego read {time.time()-t0:.2f}s cols {list(e.columns)} n {len(e)}"); print(e.head(2).to_string()[:600])
print("metadata.yaml:", z.read('metadata.yaml').decode()[:600])
