"""Build step 3: memory-moment lists for the two approach-3 slices, with b4b_moments.py's rule unchanged (a moment is memory-relevant
if anchor_t in (t_event, t_event + 5 s]; one row per (moment, event type), nearest preceding event). Usage: s3_moments.py graded_heldout|raw_heldout2|picking (pilot: the 84 picking clips' val_full rows -> out/memory_moments_picking.parquet)"""
import pandas as pd, numpy as np, json, os, sys
OUT=os.environ["OUT"]; which=sys.argv[1]; A3="{experiment_store}"
if which=="picking":
    vf=pd.read_parquet("{paddock_sid}/Alpasim/analysis_3.0/Retrain-data/s3_assembly/val_full/manifest_val_full.parquet"); pk=pd.read_parquet("{retrain_store}/picking/picking_set.parquet")
    man=vf[vf.clip_uuid.isin(set(pk.clip_uuid))].copy()
else:
    man=pd.read_parquet(f"{A3}/slices/manifest_{'graded_heldout' if which=='graded_heldout' else 'raw_heldout'}.parquet")
ev=pd.read_csv(f"{OUT}/b4b_{which}_events.csv"); cl=pd.read_csv(f"{OUT}/b4b_{which}_clips.csv")
def moments(man, ev):                       # = b4b_moments.py:8-15
    rows=[]
    for (clip, e), g in ev.groupby(['clip','event']):
        an=man.loc[man.clip_uuid==clip, 'anchor_t'].to_numpy(); te=np.sort(g.t_event.to_numpy())
        for a in an:
            d=a-te; d=d[(d>0)&(d<=5)]
            if len(d): rows.append(dict(clip_id=clip, anchor_t=float(a), event_type=e, s_since_event=float(d.min())))
    return pd.DataFrame(rows, columns=['clip_id','anchor_t','event_type','s_since_event'])
M=moments(man, ev).sort_values(['clip_id','anchor_t','event_type']); M.to_parquet(f"{OUT}/memory_moments_{'graded_heldout' if which=='graded_heldout' else ('picking' if which=='picking' else 'raw_heldout')}.parquet", index=False)
dist=M.drop_duplicates(['clip_id','anchor_t']); n=len(man); nt=int(man.clip_uuid.isin(cl[cl.has_tracks]["clip"]).sum())
r=dict(which=which, moments=n, moments_in_clips_with_tracks=nt, clips=int(cl.shape[0]), clips_with_tracks=int(cl.has_tracks.sum()), clip_hours=float(cl.dur.sum()/3600),
       events={k:int((ev.event==k).sum()) for k in ['hb3','hb4','hb6','cut_in','crossing','oov']},
       moments_per_type={k:int((M.event_type==k).sum()) for k in ['hb3','hb4','hb6','cut_in','crossing','oov']},
       share_per_type_pct={k: round(100*(M.event_type==k).sum()/(n if k.startswith('hb') else nt),2) for k in ['hb3','hb4','hb6','cut_in','crossing','oov']},
       any_event_moments=len(dist), any_event_share_pct=round(100*len(dist)/n,2), rows=len(M),
       by_type=M.merge(man[['clip_uuid','type']].drop_duplicates(), left_on='clip_id', right_on='clip_uuid').groupby('type').size().to_dict())
json.dump(r, open(f"{OUT}/s3_moments_{which}.json","w"), indent=1); print(json.dumps(r, indent=1))
