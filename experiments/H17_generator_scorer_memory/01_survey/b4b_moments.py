"""B4b step 3: moment lists and the per-type summary table.
Reads out/b4b_<mode>_events.csv + the manifests; a moment is memory-relevant if anchor_t in (t_event, t_event + 5 s] (rule unchanged from B4).
One row per (moment, event type); for a type with several qualifying events the nearest preceding one is kept.
Hard braking appears as hb3 / hb4 / hb6 (nested thresholds)."""
import pandas as pd, numpy as np, json, os, sys
RD="{runs}/paddock_sid/Alpasim/analysis_3.0/Retrain-data"; HO="{runs}/generator_joint_pai/heldout"; OUT=os.environ["OUT"]
EV=['hb3','hb4','hb6','cut_in','crossing','oov']; CT={'nurec':'reconstructed','braw':'pull-out','ordinary':'ordinary'}
def moments(man, ev):
    rows=[]
    for (clip, e), g in ev.groupby(['clip','event']):
        an=man.loc[man.clip_uuid==clip, 'anchor_t'].to_numpy(); te=np.sort(g.t_event.to_numpy())
        for a in an:
            d=a-te; d=d[(d>0)&(d<=5)]
            if len(d): rows.append(dict(clip_id=clip, anchor_t=float(a), event_type=e, s_since_event=float(d.min())))
    return pd.DataFrame(rows, columns=['clip_id','anchor_t','event_type','s_since_event'])
def summary(man, clipsdf, ev, split_col):
    out=[]
    for ct, mc in list(clipsdf.groupby(split_col))+[('ALL', clipsdf)]:
        m=man[man.clip_uuid.isin(mc["clip"])]; e=ev[ev["clip"].isin(mc["clip"])]; mm=moments(m, e)
        hrs=mc.dur.sum()/3600; hrs_t=mc[mc.has_tracks].dur.sum()/3600; nm=len(m); nm_t=int(m.clip_uuid.isin(mc[mc.has_tracks]["clip"]).sum())
        r=dict(clip_type=ct, clips=len(mc), clips_with_tracks=int(mc.has_tracks.sum()), clip_hours=round(hrs,2), clip_hours_with_tracks=round(hrs_t,2), moments=nm, moments_with_tracks=nm_t)
        for k in EV:
            n=int((e.event==k).sum()); h=hrs if k.startswith('hb') else hrs_t; nmk=nm if k.startswith('hb') else nm_t
            r[f'{k}_events']=n; r[f'{k}_per_hour']=round(n/h,2) if h else None; r[f'{k}_clips']=int(mc[mc[k]>0].shape[0])
            r[f'{k}_moments']=int((mm.event_type==k).sum()); r[f'{k}_moment_share_pct']=round(100*(mm.event_type==k).sum()/nmk,2) if nmk else None
        any_m=mm.drop_duplicates(['clip_id','anchor_t']); r['any_event_moments']=len(any_m); r['any_event_share_pct']=round(100*len(any_m)/nm,2)
        anyt=any_m[any_m.clip_id.isin(mc[mc.has_tracks]["clip"])]; r['any_event_share_pct_tracked']=round(100*len(anyt)/nm_t,2) if nm_t else None
        out.append(r)
    return pd.DataFrame(out)
# training
man=pd.read_parquet(f"{RD}/s3_assembly/manifest.parquet")
cl=pd.concat([pd.read_csv(f"{OUT}/b4b_{m}_clips.csv").assign(clip_type=CT[m]) for m in CT]); ev=pd.concat([pd.read_csv(f"{OUT}/b4b_{m}_events.csv") for m in CT])
assert cl["clip"].is_unique and len(cl)==4015, len(cl)
S=summary(man, cl, ev, 'clip_type'); S.to_csv(f"{OUT}/b4b_summary_train.csv", index=False); print("TRAIN"); print(S.T.to_string())
M=moments(man, ev).sort_values(['clip_id','anchor_t','event_type']); M.to_parquet(f"{OUT}/memory_moments_train.parquet", index=False)
print("train moment rows", len(M), "distinct moments", len(M.drop_duplicates(['clip_id','anchor_t'])), "of", len(man))
# held-out
hman=pd.read_parquet(f"{HO}/manifest_heldout.parquet"); hcl=pd.read_csv(f"{OUT}/b4b_heldout_clips.csv"); hev=pd.read_csv(f"{OUT}/b4b_heldout_events.csv")
hcl['clip_type']=hcl.type.map({'A':'reconstructed-type (raw A)','B_raw':'pull-out','ordinary':'ordinary'})
HS=summary(hman, hcl, hev, 'clip_type'); HS.to_csv(f"{OUT}/b4b_summary_heldout.csv", index=False); print("HELD-OUT"); print(HS.T.to_string())
HM=moments(hman, hev).sort_values(['clip_id','anchor_t','event_type']); HM.to_parquet(f"{OUT}/memory_moments_heldout.parquet", index=False)
print("heldout moment rows", len(HM), "distinct moments", len(HM.drop_duplicates(['clip_id','anchor_t'])), "of", len(hman))
