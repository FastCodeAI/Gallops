"""Build step 1: the graded held-out = reconstructed (NuRec) clips of val_full whose chunk is not among the picking set's 46 chunks.
Reads manifests read-only; writes manifest_graded_heldout.parquet + graded_heldout_clips.csv to {experiment_store}/slices/
and a JSON summary to out/. Checks: chunk overlap with train / picking / current held-out / exam (31 chunks, as b3_pool.py:18 derives
them from clip_index for public_2601 U our 30), clip-id overlap with the 30 scenes, metric cache + slim sample presence (all) and
loadability (sample 20), pool clipgt tracks presence (all)."""
import pandas as pd, numpy as np, pyarrow.parquet as pq, json, os, sys, time, lzma, pickle, glob
RD="{paddock_sid}/Alpasim/analysis_3.0/Retrain-data"; P="{dataset_store}"; NAS="{retrain_store}"
AS="{paddock_sid}/Alpasim"; OUT=os.environ["OUT"]; DST="{experiment_store}/slices"
t0=time.time()
vf=pd.read_parquet(f"{RD}/s3_assembly/val_full/manifest_val_full.parquet"); tr=pd.read_parquet(f"{RD}/s3_assembly/manifest.parquet", columns=["clip_uuid","group","sample_id"])
pk=pd.read_parquet(f"{NAS}/picking/picking_set.parquet"); ho=pd.read_parquet(f"{NAS}/heldout/manifest_heldout.parquet", columns=["clip_uuid","group"])
snap=glob.glob(f"{P}/hf_home/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/*/clip_index.parquet")[0]; ci=pd.read_parquet(snap)
ss=pd.read_csv(f"{AS}/alpasim/data/scenes/sim_suites.csv"); p2601={s.replace("clipgt-","") for s in ss[ss.test_suite_id=="public_2601"].scene_id}; our30=set(pd.read_csv(f"{RD}/s0_census/b2_our30.csv").uuid)
exam=p2601|our30; exam_chunks={f"chunk_{c:04d}" for c in ci.loc[[u for u in exam if u in ci.index],"chunk"]}
pkc=set(pk.chunk); print("val_full rows", len(vf), "clips", vf.clip_uuid.nunique(), "chunks", vf.group.nunique(), "| picking chunks", len(pkc))
rec=vf[vf.source=="nurec"]; print("reconstructed rows", len(rec), "clips", rec.clip_uuid.nunique(), "chunks", rec.group.nunique(), "graded", int(rec.has_label2.sum()))
inpk=rec[rec.group.isin(pkc)]; print("  of which in picking chunks: rows", len(inpk), "clips", inpk.clip_uuid.nunique(), "chunks", inpk.group.nunique())
g=rec[~rec.group.isin(pkc)].copy().sort_values(["clip_uuid","anchor_t"]).reset_index(drop=True)
clips=g.drop_duplicates("clip_uuid")[["clip_uuid","group","type","country"]].reset_index(drop=True)
# hours from the pool clipgt egomotion (NuRec clips carry no raw duration in the mining scans)
def dur(u):
    t=pq.read_table(f"{P}/s0_census/pool/{u}/clipgt__egomotion_estimate.parquet", columns=["key"]).flatten().column("key.timestamp_micros").to_numpy()
    return (t.max()-t.min())/1e6
clips["dur_s"]=[dur(u) for u in clips.clip_uuid]
R=dict(clips=int(len(clips)), chunks=int(clips.group.nunique()), moments=int(len(g)), graded_moments=int(g.has_label2.sum()), hours=float(clips.dur_s.sum()/3600),
       by_type=g.groupby("type").agg(moments=("sample_id","size"), clips=("clip_uuid","nunique")).to_dict("index"),
       moments_per_clip=dict(min=int(g.groupby("clip_uuid").size().min()), median=float(g.groupby("clip_uuid").size().median()), max=int(g.groupby("clip_uuid").size().max())),
       anchor_t=dict(min=float(g.anchor_t.min()), max=float(g.anchor_t.max())))
# overlap checks
gc=set(g.group); checks=dict(chunks_shared_with_train=sorted(gc&set(tr.group)), chunks_shared_with_picking=sorted(gc&pkc), chunks_shared_with_heldout=sorted(gc&set(ho.group)),
    chunks_shared_with_exam31=sorted(gc&exam_chunks), exam_chunks_n=len(exam_chunks), clips_in_exam=sorted(set(g.clip_uuid)&exam), clips_in_our30=sorted(set(g.clip_uuid)&our30),
    clips_shared_with_train=sorted(set(g.clip_uuid)&set(tr.clip_uuid)), sample_ids_shared_with_train=int(g.sample_id.isin(set(tr.sample_id)).sum()),
    clips_shared_with_picking=sorted(set(g.clip_uuid)&set(pk.clip_uuid)))
# files: metric cache + slim + pool tracks
CACHE=f"{P}/s3_assembly/s1/cache"; SLIM=f"{P}/s4_train/slim"
g["slim"]=[os.path.exists(f"{SLIM}/{u}/{s}.npz") for u,s in zip(g.clip_uuid,g.sample_id)]
g["feat"]=[os.path.exists(f"{CACHE}/{u}/{s}/pad_feature.gz") for u,s in zip(g.clip_uuid,g.sample_id)]
g["mc"]=[os.path.exists(f"{CACHE}/{u}/{s}/metric_cache.pkl") for u,s in zip(g.clip_uuid,g.sample_id)]
g["lab2"]=[os.path.exists(f"{CACHE}/{u}/{s}/pad_label2.gz") for u,s in zip(g.clip_uuid,g.sample_id)]
clips["tracks"]=[os.path.exists(f"{P}/s0_census/pool/{u}/clipgt__obstacle.parquet") for u in clips.clip_uuid]
files=dict(slim_present=int(g.slim.sum()), pad_feature_present=int(g.feat.sum()), metric_cache_present=int(g.mc.sum()), metric_cache_present_on_graded=int((g.mc&g.has_label2).sum()),
           graded_without_metric_cache=int((g.has_label2&~g.mc).sum()), ungraded_with_metric_cache=int((~g.has_label2&g.mc).sum()), pad_label2_present=int(g.lab2.sum()), clips_with_pool_tracks=int(clips.tracks.sum()), clips=int(len(clips)))
# load a sample of 20 graded moments
rng=np.random.RandomState(0); samp=g[g.has_label2&g.slim&g.mc].sample(20, random_state=0); loaded=[]
for r in samp.itertuples():
    with np.load(f"{SLIM}/{r.clip_uuid}/{r.sample_id}.npz") as z: keys=sorted(z.files); shp={k:z[k].shape for k in ("lidar2img","img_shape","ego_status","trajectory")}; tok=str(z["token"])
    with lzma.open(f"{CACHE}/{r.clip_uuid}/{r.sample_id}/metric_cache.pkl","rb") as f: mc=pickle.load(f)
    loaded.append(dict(sample_id=r.sample_id, slim_keys=keys, shapes={k:list(v) for k,v in shp.items()}, token_ok=tok==r.sample_id, metric_cache_type=type(mc).__name__))
assert all(x["token_ok"] for x in loaded)
# write
man=g[list(vf.columns)].reset_index(drop=True)
for c in man.columns: man[c]=man[c].astype(vf[c].dtype)
man.to_parquet(f"{DST}/manifest_graded_heldout.parquet", index=False); clips.rename(columns={"group":"chunk"}).to_csv(f"{DST}/graded_heldout_clips.csv", index=False)
import hashlib; sha=lambda p: hashlib.sha256(open(p,"rb").read()).hexdigest()
res=dict(summary=R, checks=checks, files=files, loaded_sample=loaded, outputs=dict(manifest=f"{DST}/manifest_graded_heldout.parquet", manifest_sha256=sha(f"{DST}/manifest_graded_heldout.parquet"), clips=f"{DST}/graded_heldout_clips.csv", clips_sha256=sha(f"{DST}/graded_heldout_clips.csv")), secs=time.time()-t0)
json.dump(res, open(f"{OUT}/s1_graded_heldout.json","w"), indent=1, default=str); print(json.dumps({k:v for k,v in res.items() if k!="loaded_sample"}, indent=1, default=str)); print("loaded sample:", loaded[0], "...", len(loaded))
