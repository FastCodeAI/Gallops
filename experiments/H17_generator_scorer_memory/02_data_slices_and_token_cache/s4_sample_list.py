"""Build step 4a: the 1,000-moment cache sample = whole picking-set clips in time order (every val_full moment of those clips, so
consecutive moments exist; the picking set itself holds only ~24 of a clip's 32 moments). Clips are taken in picking-set order
(sorted by sample_id) until >= 1,000 moments. Writes out/s4_sample_moments.parquet (val_full rows + in_picking flag)."""
import pandas as pd, os, json
RD="{runs}/paddock_sid/Alpasim/analysis_3.0/Retrain-data"; NAS="{runs}/generator_joint_pai"; OUT=os.environ["OUT"]
vf=pd.read_parquet(f"{RD}/s3_assembly/val_full/manifest_val_full.parquet"); pk=pd.read_parquet(f"{NAS}/picking/picking_set.parquet")
assert set(pk.sample_id)<=set(vf.sample_id)
clips=sorted(set(pk.clip_uuid)); rows=[]; n=0
for u in clips:
    m=vf[vf.clip_uuid==u].sort_values("anchor_t"); rows.append(m); n+=len(m)
    if n>=1000: break
s=pd.concat(rows).reset_index(drop=True); s["in_picking"]=s.sample_id.isin(set(pk.sample_id))
s.to_parquet(f"{OUT}/s4_sample_moments.parquet", index=False)
r=dict(clips=int(s.clip_uuid.nunique()), moments=len(s), picking_moments=int(s.in_picking.sum()), graded=int(s.has_label2.sum()), graded_in_picking=int((s.in_picking&s.has_label2).sum()),
       by_source=s.source.value_counts().to_dict(), by_type=s.type.value_counts().to_dict(), anchor_steps_consecutive=float((s.groupby("clip_uuid").anchor_t.diff().dropna()==0.5).mean()), moments_per_clip=s.groupby("clip_uuid").size().describe()[["min","50%","max"]].to_dict())
json.dump(r, open(f"{OUT}/s4_sample_moments.json","w"), indent=1); print(json.dumps(r, indent=1))
