import pandas as pd, numpy as np, json, os, time, yaml, glob, io, zipfile
RD="{paddock_sid}/Alpasim/analysis_3.0/Retrain-data"
P="{dataset_store}"; NAS="{retrain_store}"
SNAP=glob.glob(f"{P}/hf_home/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/*/")[0]
OUT="/tmp/{session}/out"; os.makedirs(OUT, exist_ok=True)
tr=pd.read_parquet(f"{RD}/s3_assembly/manifest.parquet"); va=pd.read_parquet(f"{RD}/s3_assembly/val/manifest_val.parquet")
vf=pd.read_parquet(f"{RD}/s3_assembly/val_full/manifest_val_full.parquet"); ho=pd.read_parquet(f"{NAS}/heldout/manifest_heldout.parquet")
pk=pd.read_parquet(f"{NAS}/picking/picking_set.parquet")
ci=pd.read_parquet(f"{SNAP}/clip_index.parquet"); ci=ci.reset_index()
dc=pd.read_parquet(f"{SNAP}/metadata/data_collection.parquet").reset_index(); print("data_collection cols", list(dc.columns), len(dc))
fp=pd.read_parquet(f"{SNAP}/metadata/feature_presence.parquet"); print("feature_presence cols", list(fp.columns)[:12], len(fp))
raw=pd.read_csv(f"{RD}/s0_census/r3_raw_clips.csv")
print("r3_raw_clips status", raw.status.value_counts().to_dict(), "dur stats", raw.dur.describe().to_dict())
# --- B1 hours
def slice_hours(name, df):
    u=df.clip_uuid.unique(); d=raw.set_index('uuid').dur.reindex(u)
    print(f"{name}: moments {len(df)} clips {len(u)} dur-known {d.notna().sum()} hours(known) {d.sum()/3600:.2f} (assuming 20 s for unknown: {(d.fillna(20).sum())/3600:.2f})")
    return u
tu=slice_hours("train", tr); slice_hours("val(608)", va); vfu=slice_hours("val_full", vf); pku=slice_hours("picking", pk)
# heldout durations from mining scan
cs=pd.read_csv(f"{NAS}/heldout/mining/clips_scan.csv"); print("heldout clips_scan cols", list(cs.columns), len(cs))
hu=ho.clip_uuid.unique(); dh=cs.set_index(cs.columns[0]).reindex(hu)
dcol=[c for c in cs.columns if c.startswith('dur')]; print("heldout dur col", dcol)
if dcol: print(f"heldout: moments {len(ho)} clips {len(hu)} dur-known {dh[dcol[0]].notna().sum()} hours {dh[dcol[0]].sum()/3600:.2f}")
vcs=pd.read_csv(f"{NAS}/heldout/mining/valfull_clips_scan.csv"); dv=vcs.set_index(vcs.columns[0]).reindex(vfu)
if dcol: print(f"val_full via valfull scan: dur-known {dv[dcol[0]].notna().sum()} hours {dv[dcol[0]].sum()/3600:.2f}; picking via same: {dv[dcol[0]].reindex(pku).notna().sum()} known, {dv[dcol[0]].reindex(pku).sum()/3600:.2f} h")
# dataset totals
print("clip_index total", len(ci), ci.split.value_counts().to_dict(), "valid", int(ci.clip_is_valid.sum()), "chunks", ci.chunk.nunique())
print("raw_train scanned clips", len(raw), "sum hours", raw.dur.sum()/3600, "median dur", raw.dur.median())
# what is physically on NAS
print("NAS: s3 clips dirs", len(os.listdir(f"{P}/s3_assembly/s1/clips")), "heldout clip dirs", len(os.listdir(f"{NAS}/heldout/clips")), "pool scenes", len(os.listdir(f"{P}/s0_census/pool")))
# --- B3 leakage by id
our30=pd.read_csv(f"{RD}/s0_census/b2_our30.csv"); ss=pd.read_csv("{paddock_sid}/Alpasim/alpasim/data/scenes/sim_suites.csv")
pub=ss[ss.test_suite_id=='public_2601']; pub_ids=set(pub.scene_id.str.replace('clipgt-','')); print("public_2601 scenes", len(pub_ids), "suites", ss.test_suite_id.unique().tolist())
g10=yaml.safe_load(open("{paddock}-gate/src/paddock/adapters/benchmarks/alpasim/scenes/public_2601_gate10.yaml"))['scenes']; g10=set(s.replace('clipgt-','') for s in g10)
o30=set(our30.uuid); print("our30", len(o30), "gate10", len(g10), "gate10 subset of our30:", g10<=o30, "our30 subset of public_2601:", o30<=pub_ids)
exam=pub_ids|o30|g10
for name,u in [("train",tu),("val",va.clip_uuid.unique()),("val_full",vfu),("picking",pku),("heldout",hu)]:
    hit=set(u)&exam; print(f"id overlap {name} vs (public_2601 ∪ our30 ∪ gate10): {len(hit)} {sorted(hit)[:5]}")
# chunk overlap: chunks of exam clips present in raw clip_index
ex_in_raw=ci[ci.clip_id.isin(exam)]; print("exam clips present in raw clip_index:", len(ex_in_raw), ex_in_raw.split.value_counts().to_dict())
o30_in_raw=ci[ci.clip_id.isin(o30)]; print("our30 in raw clip_index:", len(o30_in_raw), o30_in_raw.split.value_counts().to_dict(), "chunks", sorted(o30_in_raw.chunk.unique().tolist()))
ex_chunks=set(ex_in_raw.chunk)
for name,df in [("train",tr),("val",va),("val_full",vf),("heldout",ho)]:
    ch=set(df.group.str.replace('chunk_','').astype(int)); print(f"chunk overlap {name} ({len(ch)} chunks) vs exam chunks ({len(ex_chunks)}): {sorted(ch&ex_chunks)}")
pkch=set(pk.chunk.str.replace('chunk_','').astype(int)); print("chunk overlap picking vs exam chunks:", sorted(pkch&ex_chunks))
# our30 -> 26.01 artifact uuid -> usdz on NAS
m=pub.set_index(pub.scene_id.str.replace('clipgt-',''))['uuid']; us=set(os.path.basename(p)[:-5] for p in glob.glob("{alpasim_store}/nre-artifacts/all-usdzs/*.usdz"))
have=[(c, m.get(c), (m.get(c) in us)) for c in sorted(o30)]; print("our30 usdz on NAS:", sum(h[2] for h in have), "/30"); json.dump(have, open(f"{OUT}/our30_usdz.json","w"), indent=1)
# data_collection metadata of our30 vs train (weak proxy)
dci=dc.set_index(dc.columns[0]) if dc.columns[0]=='clip_id' else dc.set_index('clip_id')
print(dci.head(2).to_string())
json.dump(dict(train=sorted(tu.tolist()), heldout=sorted(hu.tolist()), picking=sorted(pku.tolist())), open(f"{OUT}/slice_clips.json","w"))
