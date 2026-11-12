"""B4b fetch verification: every ordinary clip has its member; sha256 matches the manifest; parquet reads; row counts sane."""
import pandas as pd, numpy as np, os, json, hashlib
RD="{runs}/paddock_sid/Alpasim/analysis_3.0/Retrain-data"; DST="{runs}/physicalai/approach3_tracks"; OUT=os.environ["OUT"]
tr=pd.read_parquet(f"{RD}/s3_assembly/manifest.parquet"); tr['chunk']=tr.group.str.replace('chunk_','').astype(int)
ordn=tr[(tr.source=='raw')&(tr.type=='ordinary')].drop_duplicates('clip_uuid')[['clip_uuid','chunk']]
man=pd.read_csv(f"{DST}/download_manifest.csv"); print("manifest rows", len(man), "distinct uuids", man.uuid.nunique(), "all ordinary:", set(man.uuid)==set(ordn.clip_uuid))
rows=[]
for r in ordn.itertuples():
    p=f"{DST}/obstacle.offline.members/chunk_{r.chunk:04d}/{r.clip_uuid}.obstacle.offline.parquet"; d=dict(clip_uuid=r.clip_uuid, chunk=r.chunk, exists=os.path.exists(p))
    if not d['exists']: rows.append(d); continue
    b=open(p,'rb').read(); d['bytes']=len(b); d['sha_ok']=hashlib.sha256(b).hexdigest()==man.loc[man.uuid==r.clip_uuid,'sha256'].iloc[0]
    try:
        df=pd.read_parquet(p); d['rows']=len(df); d['tracks']=df.track_id.nunique(); d['rig']=bool((df.reference_frame=='rig').all())
        t=df.timestamp_us.to_numpy().astype(np.float64)/1e6; d['span_s']=float(t.max()-t.min()) if len(t) else 0.0; d['read_ok']=True
        d['cols_ok']=set(['track_id','timestamp_us','reference_frame','reference_frame_timestamp_us','center_x','center_y','size_x','size_y','orientation_x','orientation_y','orientation_z','orientation_w','label_class'])<=set(df.columns)
    except Exception as e: d['read_ok']=False; d['err']=repr(e)[:120]
    rows.append(d)
V=pd.DataFrame(rows); V.to_csv(f"{OUT}/b4b_fetch_verify.csv", index=False)
print("clips", len(V), "exists", int(V.exists.sum()), "sha_ok", int(V.sha_ok.sum()), "read_ok", int(V.read_ok.sum()), "cols_ok", int(V.cols_ok.sum()), "rig frame", int(V.rig.sum()))
print("bytes: min %d median %d mean %d max %d; total %.3f GB"%(V.bytes.min(), V.bytes.median(), V.bytes.mean(), V.bytes.max(), V.bytes.sum()/1e9))
print("rows per clip: min %d p5 %d median %d p95 %d max %d; clips with 0 rows: %d"%(V.rows.min(), V.rows.quantile(.05), V.rows.median(), V.rows.quantile(.95), V.rows.max(), int((V.rows==0).sum())))
print("tracks per clip: min %d median %d max %d; span s: min %.1f median %.1f max %.1f"%(V.tracks.min(), V.tracks.median(), V.tracks.max(), V.span_s.min(), V.span_s.median(), V.span_s.max()))
bad=V[(~V.exists)|(~V.sha_ok.fillna(False))|(~V.read_ok.fillna(False))|(~V.cols_ok.fillna(False))|(V.rows.fillna(0)<50)|(V.span_s.fillna(0)<15)]
print("flagged clips (missing / bad hash / unreadable / < 50 rows / span < 15 s):", len(bad)); print(bad.to_string() if len(bad) else "")
# compare with the census's pull-out members as a sanity reference
bo=tr[(tr.source=='raw')&(tr.type=='B_raw')].drop_duplicates('clip_uuid'); ref=[len(pd.read_parquet(f"{runs}/physicalai/s0_census/raw/labels/obstacle.offline.members/chunk_{r.chunk:04d}/{r.clip_uuid}.obstacle.offline.parquet")) for r in bo.head(100).itertuples()]
print("reference: rows per clip on 100 census pull-out members: median %d, min %d, max %d"%(np.median(ref), min(ref), max(ref)))
