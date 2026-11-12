import pandas as pd, numpy as np, pyarrow.parquet as pq, os, json
RD="{paddock_sid}/Alpasim/analysis_3.0/Retrain-data"; P="{dataset_store}"; OUT=os.environ["OUT"]
tr=pd.read_parquet(f"{RD}/s3_assembly/manifest.parquet").sort_values(['clip_uuid','anchor_t'])
# hard braking from manifest speeds (all 4,015 clips): decel between consecutive anchors 0.5 s apart
tr['dv']=tr.groupby('clip_uuid').speed.diff(); tr['dt']=tr.groupby('clip_uuid').anchor_t.diff(); ok=(tr.dt.round(2)==0.5)
dec=-tr.dv/0.5; prev=tr.groupby('clip_uuid').speed.shift()
res={}
for thr in (3,4,6):
    m=ok&(dec>=thr)&(prev>=2); res[f'anchor_pairs_decel_ge_{thr}']=int(m.sum()); res[f'clips_with_decel_ge_{thr}']=int(tr[m].clip_uuid.nunique())
res['anchor_pairs_checked']=int(ok.sum()); res['clips']=int(tr.clip_uuid.nunique()); res['hours']=float(tr.clip_uuid.nunique()*20/3600)
for thr in (3,4,6): res[f'rate_per_hour_ge_{thr}']=res[f'anchor_pairs_decel_ge_{thr}']/res['hours']
print("manifest-speed hard braking (0.5 s anchor pairs, prev speed >= 2 m/s):", json.dumps(res, indent=1))
json.dump(res, open(f"{OUT}/b4_manifest_brake.json","w"), indent=1)
# track gap distribution on 20 NuRec clips: max gap between consecutive observations per track id
clips=tr[tr.source=='nurec'].drop_duplicates('clip_uuid').clip_uuid.head(20); gaps=[]; nt=0
for u in clips:
    o=pq.read_table(f"{P}/s0_census/pool/{u}/clipgt__obstacle.parquet").flatten().flatten()
    df=pd.DataFrame({"tid": o.column("obstacle.trackline_id").to_pylist(), "ts": o.column("key.timestamp_micros").to_numpy(zero_copy_only=False).astype(np.int64)/1e6})
    for tid,r in df.groupby('tid'):
        a=np.sort(r.ts.values); nt+=1
        if len(a)>1: gaps.append(np.diff(a).max())
g=np.array(gaps); print(f"20 NuRec clips: {nt} tracks; max gap per track: median {np.median(g):.3f} s, p99 {np.percentile(g,99):.3f}, max {g.max():.3f}; tracks with max gap >= 0.5 s: {(g>=0.5).sum()}, >= 1.0 s: {(g>=1.0).sum()}")
