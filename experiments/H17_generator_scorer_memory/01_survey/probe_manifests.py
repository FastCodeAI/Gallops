import pyarrow.parquet as pq, pandas as pd, json, time, sys
RD="{runs}/paddock_sid/Alpasim/analysis_3.0/Retrain-data/s3_assembly"
NAS="{runs}/generator_joint_pai"
def info(name, path):
    t0=time.time(); t=pq.read_table(path); df=t.to_pandas(); 
    print(f"== {name}: {path}\n   rows {len(df)}  cols {list(df.columns)}  read {time.time()-t0:.1f}s")
    for c in ('clip_uuid','clip_id','uuid'):
        if c in df: print(f"   clips {df[c].nunique()}"); break
    if 'anchor_t' in df: print(f"   anchor_t min/median/max {df.anchor_t.min()} {df.anchor_t.median()} {df.anchor_t.max()}  dtype {df.anchor_t.dtype}")
    if 'split' in df: print("   split", df.split.value_counts().to_dict())
    if 'source' in df: print("   source", df.source.value_counts().to_dict())
    if 'type' in df: print("   type", df.type.value_counts().to_dict())
    if 'has_label2' in df: print("   has_label2", int(df.has_label2.sum()))
    if 'group' in df: print("   groups(chunks)", df.group.nunique())
    print(df.head(2).T.to_string()[:1500])
    return df
tr=info("train", f"{RD}/manifest.parquet")
import os
for p in [f"{RD}/val/manifest_val.parquet", f"{RD}/val_full/manifest_val_full.parquet"]:
    if os.path.exists(p): info(os.path.basename(p), p)
    else: print("missing", p, os.listdir(os.path.dirname(p))[:20])
ho=info("heldout", f"{NAS}/heldout/manifest_heldout.parquet")
info("heldout_tags", f"{NAS}/heldout/manifest_heldout_tags.parquet")
pk=info("picking", f"{NAS}/picking/picking_set.parquet")
