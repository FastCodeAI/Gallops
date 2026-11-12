"""Build step 2a: select the fresh raw held-out from the strict validation pool (NAS/heldout/pool_clips.csv, mined by b3_mine.py pool)
minus every chunk of the current held-out. The rule is b3_select.py:13-30 (pull-outs up to 100, fast 75 with held >= 16 m/s, the rest
ordinary to 300, chunk cap 2, seeded draw) with a new seed. Train / val / val_full / exam chunks are already excluded by the strict pool
(b3_pool.py:23) and re-asserted here. Writes heldout_clips.csv (the name b3_download/b3_convert/b3_manifest expect) under
{runs}/generator_scorer_memory/raw_heldout/ and a summary JSON."""
import sys, json, random, pathlib, hashlib, pandas as pd, glob
H=pathlib.Path("{runs}/generator_joint_pai/heldout"); DST=pathlib.Path("{runs}/generator_scorer_memory/raw_heldout"); RD=pathlib.Path("{runs}/paddock_sid/Alpasim/analysis_3.0/Retrain-data")
SEED, N_PULL, N_FAST, N_TOTAL, CAP = 20261107, 100, 75, 300, 2
snap=glob.glob("{runs}/physicalai/hf_home/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/*/clip_index.parquet")[0]; ci=pd.read_parquet(snap)
dc=pd.read_parquet(pathlib.Path(snap).parent/"metadata/data_collection.parquet")
pool=pd.read_csv(H/"pool_clips.csv"); cl=pd.read_csv(H/"mining/clips_scan.csv"); mo=pd.read_csv(H/"mining/moveoffs_actors.csv"); ho=pd.read_csv(H/"heldout_clips.csv")
tr=pd.read_parquet(RD/"s3_assembly/manifest.parquet", columns=["group"]); vf=pd.read_parquet(RD/"s3_assembly/val_full/manifest_val_full.parquet", columns=["group"]); va=pd.read_parquet(RD/"s3_assembly/val/manifest_val.parquet", columns=["group"])
gh=pd.read_csv("{runs}/generator_scorer_memory/slices/graded_heldout_clips.csv")
AS=pathlib.Path("{runs}/paddock_sid/Alpasim"); ss=pd.read_csv(AS/"alpasim/data/scenes/sim_suites.csv"); p2601={s.replace("clipgt-","") for s in ss[ss.test_suite_id=="public_2601"].scene_id}; our30=set(pd.read_csv(RD/"s0_census/b2_our30.csv").uuid); exam=p2601|our30
exam_chunks=set(int(c) for c in ci.loc[[u for u in exam if u in ci.index],"chunk"]); tochunk=lambda s: {int(x.replace("chunk_","")) for x in s}
excl=tochunk(tr.group)|tochunk(vf.group)|tochunk(va.group)|exam_chunks|set(ho.chunk)|tochunk(gh.chunk)
cl=cl[(cl.status=="ok")&cl.uuid.isin(pool.uuid)]; before=dict(clips=len(cl), chunks=int(cl.chunk.nunique()))
cl=cl[~cl.chunk.isin(excl)]; assert not (set(cl.chunk)&excl)
ev=mo[(mo.actor_bum==True)&(mo.dyaw4_deg.abs()<=45)]
pull_all=sorted(set(ev.uuid)&set(cl.uuid)); fast_all=sorted(set(cl.uuid[cl.held>=16.0])-set(pull_all)); ord_all=sorted(set(cl.uuid)-set(pull_all)-set(fast_all))
rng=random.Random(SEED)
pull=sorted(rng.sample(pull_all,N_PULL)) if len(pull_all)>N_PULL else pull_all
cnt={}; chunk=dict(zip(cl.uuid,cl.chunk))
for u in pull: cnt[chunk[u]]=cnt.get(chunk[u],0)+1
def draw(cands,n):
    out=[]
    for u in rng.sample(cands,len(cands)):
        if len(out)==n: break
        if cnt.get(chunk[u],0)>=CAP: continue
        out.append(u); cnt[chunk[u]]=cnt.get(chunk[u],0)+1
    return sorted(out)
fast=draw(fast_all,N_FAST); ordi=draw(ord_all,N_TOTAL-len(pull)-len(fast))
rows=[dict(uuid=u,block="pullout") for u in pull]+[dict(uuid=u,block="fast") for u in fast]+[dict(uuid=u,block="ordinary") for u in ordi]
df=pd.DataFrame(rows); c2=cl.set_index("uuid")
df["chunk"]=df.uuid.map(c2.chunk).astype(int); df["split"]=df.uuid.map(ci.split); df["country"]=df.uuid.map(dc.country)
df["held_mps"]=df.uuid.map(c2.held).round(3); df["vmax_mps"]=df.uuid.map(c2.vmax).round(3); df["moveoffs"]=df.uuid.map(c2.moveoffs)
pe=ev.groupby("uuid").agg(n_pullout_events=("t_move","size"), pullout_t_move=("t_move", lambda x: ";".join(f"{v:g}" for v in x)), pullout_t_depart=("t_depart", lambda x: ";".join(f"{v:g}" for v in x)))
df=df.join(pe,on="uuid"); df["n_pullout_events"]=df.n_pullout_events.fillna(0).astype(int)
df["tags"]=df.apply(lambda r: ";".join(t for t,ok in (("pullout",r.n_pullout_events>0),("fast_clip",r.held_mps>=16.0),("still_clip",r.vmax_mps<3.0)) if ok) or "ordinary", axis=1)
df["seed"]=SEED; df["rule"]=f"pull<= {N_PULL} all, fast {N_FAST} held>=16, rest ordinary to {N_TOTAL}, chunk cap {CAP}; pool = strict pool minus current held-out chunks"
assert df.uuid.is_unique and len(df)==N_TOTAL and (df.split=="val").all() and not (set(df.chunk)&excl) and not (set(df.uuid)&(exam|set(ho.uuid)))
DST.mkdir(parents=True, exist_ok=True); df.sort_values(["block","uuid"]).to_csv(DST/"heldout_clips.csv", index=False)
sha=hashlib.sha256(open(DST/"heldout_clips.csv","rb").read()).hexdigest(); open(DST/"heldout_clips.csv.sha256","w").write(f"{sha}  heldout_clips.csv\n")
r=dict(seed=SEED, strict_pool=before, excluded_chunks=dict(train=len(tochunk(tr.group)), val=len(tochunk(va.group)), val_full=len(tochunk(vf.group)), exam=len(exam_chunks), current_heldout=int(ho.chunk.nunique()), graded_heldout=len(tochunk(gh.chunk)), union=len(excl)),
       remaining_pool=dict(clips=len(cl), chunks=int(cl.chunk.nunique())), available=dict(pullout_clips=len(pull_all), pullout_events=int(ev[ev.uuid.isin(cl.uuid)].shape[0]), fast=len(fast_all), ordinary=len(ord_all)),
       selected=df.block.value_counts().to_dict(), chunks=int(df.chunk.nunique()), max_per_chunk=int(df.groupby("chunk").size().max()), countries=df.country.value_counts().to_dict(), still_clips=int((df.vmax_mps<3).sum()), sha256=sha)
json.dump(r, open(DST/"heldout_select_summary.json","w"), indent=1, default=int); print(json.dumps(r, indent=1, default=int))
