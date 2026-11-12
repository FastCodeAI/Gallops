import pandas as pd, numpy as np, json, os, glob, zipfile, yaml, time
RD="{runs}/paddock_sid/Alpasim/analysis_3.0/Retrain-data"; P="{runs}/physicalai"; NAS="{runs}/generator_joint_pai"
OUT=os.environ["OUT"]; SNAP=glob.glob(f"{P}/hf_home/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/*/")[0]
dc=pd.read_parquet(f"{SNAP}/metadata/data_collection.parquet").reset_index().set_index('clip_id')
ci=pd.read_parquet(f"{SNAP}/clip_index.parquet").reset_index().set_index('clip_id')
our=json.load(open(f"{OUT}/our30_usdz.json")); t0=time.time(); test=[]
for clip, art, ok in our:
    z=zipfile.ZipFile(f"{runs}/alpasim/nre-artifacts/all-usdzs/{art}.usdz"); m=yaml.safe_load(z.read('metadata.yaml'))
    di=json.loads(z.read('data_info.json')) if 'data_info.json' in z.namelist() else {}
    test.append(dict(clip=clip, start=m['time_range']['start'], end=m['time_range']['end'], scene_id=m.get('scene_id'), in_raw=clip in ci.index, country=dc.country.get(clip), month=dc.month.get(clip), hour=dc.hour_of_day.get(clip)))
test=pd.DataFrame(test); print(f"30 usdz metadata read {time.time()-t0:.1f}s"); print(test.to_string())
# NuRec-sourced clips in train / val_full / picking: pool metadata.yaml time ranges
tr=pd.read_parquet(f"{RD}/s3_assembly/manifest.parquet"); vf=pd.read_parquet(f"{RD}/s3_assembly/val_full/manifest_val_full.parquet"); pk=pd.read_parquet(f"{NAS}/picking/picking_set.parquet")
nu=sorted(set(tr[tr.source=='nurec'].clip_uuid)|set(vf[vf.source=='nurec'].clip_uuid)); print("NuRec-sourced clips (train ∪ val_full):", len(nu))
sl={u:'train' for u in tr[tr.source=='nurec'].clip_uuid}; sl.update({u:'val_full' for u in vf[vf.source=='nurec'].clip_uuid}); pkset=set(pk.clip_uuid)
t0=time.time(); rows=[]
for u in nu:
    p=f"{P}/s0_census/pool/{u}/metadata.yaml"
    if not os.path.exists(p): rows.append(dict(clip=u, missing=True)); continue
    m=yaml.safe_load(open(p)); rows.append(dict(clip=u, start=m['time_range']['start'], end=m['time_range']['end'], slice=sl[u], picking=u in pkset, country=dc.country.get(u)))
nr=pd.DataFrame(rows); print(f"pool metadata read {time.time()-t0:.1f}s; missing {int(nr.get('missing',pd.Series(dtype=bool)).fillna(False).sum())}")
nr=nr[nr.start.notna()]; print("NuRec clip time spans (s):", ((nr.end-nr.start)/1e6).describe().to_dict())
print("clock magnitude test scenes start (s):", (test.start/1e6).describe()[['min','max']].to_dict(), " NuRec train/val_full starts:", (nr.start/1e6).describe()[['min','max']].to_dict())
MARG=600e6  # 10 min either side
hits=[]
for t in test.itertuples():
    o=nr[(nr.end>=t.start-MARG)&(nr.start<=t.end+MARG)]
    for r in o.itertuples(): hits.append(dict(test=t.clip[:8], test_country=t.country, clip=r.clip, slice=r.slice, picking=r.picking, country=r.country, gap_s=float(max(r.start-t.end, t.start-r.end, 0)/1e6)))
hits=pd.DataFrame(hits); print(f"time-window hits within ±10 min (NuRec-sourced clips only): {len(hits)}"); 
if len(hits): print(hits.sort_values('gap_s').to_string())
print("hits with gap 0 (overlap):", int((hits.gap_s==0).sum()) if len(hits) else 0, " within 60 s:", int((hits.gap_s<=60).sum()) if len(hits) else 0)
# raw clips: only country/month/hour available
raw_tr=tr[tr.source=='raw'].drop_duplicates('clip_uuid').clip_uuid
meta=dc.reindex(raw_tr); cnt=[]
for t in test.itertuples():
    k=((meta.country==t.country)&(meta.month==t.month)&(meta.hour_of_day==t.hour)).sum(); cnt.append(dict(test=t.clip[:8], same_country_month_hour=int(k)))
print("raw training clips sharing (country, month, hour_of_day) with each test scene (weak proxy, no clock available):"); print(pd.DataFrame(cnt).to_string())
test.to_csv(f"{OUT}/test30_timeranges.csv", index=False); nr.to_csv(f"{OUT}/nurec_timeranges.csv", index=False); hits.to_csv(f"{OUT}/timerange_hits.csv", index=False)
