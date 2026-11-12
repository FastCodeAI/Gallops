import pandas as pd, numpy as np, json, os, sys, zipfile, io, time
RD="{runs}/paddock_sid/Alpasim/analysis_3.0/Retrain-data"; P="{runs}/physicalai"; NAS="{runs}/generator_joint_pai"
OUT=os.environ["OUT"]; which=sys.argv[1]; NSAMP=int(sys.argv[2]) if len(sys.argv)>2 else 0
if which=="train":
    man=pd.read_parquet(f"{RD}/s3_assembly/manifest.parquet"); ZR=f"{P}/s0_census/raw/labels/egomotion.offline"; CL=f"{P}/s3_assembly/s1/clips"
else:
    man=pd.read_parquet(f"{NAS}/heldout/manifest_heldout.parquet"); ZR=f"{NAS}/heldout/raw/labels/egomotion.offline"; CL=f"{NAS}/heldout/clips"
man['chunk']=man.group.str.replace('chunk_','').astype(int)
clips=man.drop_duplicates('clip_uuid')[['clip_uuid','chunk']]
if NSAMP: clips=clips.sample(NSAMP, random_state=1)
clips=clips.sort_values('chunk')
EGO_TOL, FR_TOL = 0.15, 0.05   # s: 1.5x the nominal 0.1 s pose step and 1.5x the nominal 1/30 s frame step
t0=time.time(); res={}; zc={}; n=0
for r in clips.itertuples():
    if r.chunk not in zc: zc={r.chunk: zipfile.ZipFile(f"{ZR}/egomotion.offline.chunk_{r.chunk:04d}.zip")}
    e=pd.read_parquet(io.BytesIO(zc[r.chunk].read(f"{r.clip_uuid}.egomotion.offline.parquet"))); ets=np.sort(e.timestamp.values.astype(np.int64))/1e6
    f=pd.read_parquet(f"{CL}/{r.clip_uuid}/{r.clip_uuid}.camera_front_wide_120fov.timestamps.parquet"); fts=np.sort(f.timestamp.values.astype(np.int64))/1e6
    ed=np.diff(ets); fd=np.diff(fts)
    egaps=[(ets[i], ets[i+1]) for i in np.where(ed>EGO_TOL)[0]]; fgaps=[(fts[i], fts[i+1]) for i in np.where(fd>FR_TOL)[0]]
    res[r.clip_uuid]=dict(ego_t0=float(ets[0]), ego_t1=float(ets[-1]), ego_n=int(len(ets)), ego_dt_max=float(ed.max()), fr_t0=float(fts[0]), fr_t1=float(fts[-1]), fr_n=int(len(fts)), fr_dt_max=float(fd.max()), egaps=egaps, fgaps=fgaps)
    n+=1
    if n%50==0: print(f"{n} clips {time.time()-t0:.0f}s", flush=True)
print(f"scanned {n} clips in {time.time()-t0:.0f}s")
R=pd.DataFrame.from_dict(res, orient='index'); print("ego n/dt_max:", R.ego_n.describe()[['min','max']].to_dict(), R.ego_dt_max.describe()[['min','50%','max']].to_dict())
print("frames n/dt_max:", R.fr_n.describe()[['min','max']].to_dict(), R.fr_dt_max.describe()[['min','50%','max']].to_dict(), "frame t0 (s, rel. to pose 0):", R.fr_t0.describe()[['min','50%','max']].to_dict())
print("clips with any pose gap >", EGO_TOL, ":", int((R.egaps.str.len()>0).sum()), " any frame gap >", FR_TOL, ":", int((R.fgaps.str.len()>0).sum()))
# history per moment: anchor_t minus the latest of (pose start, frame start, end of the last gap before the anchor)
sub=man[man.clip_uuid.isin(res)]; hist=[]
for r in sub.itertuples():
    c=res[r.clip_uuid]; start=max(c['ego_t0'], c['fr_t0'])
    for a,b in c['egaps']+c['fgaps']:
        if b<=r.anchor_t: start=max(start,b)
    hist.append(r.anchor_t-start)
h=np.array(hist); bins=[0,0.5,1,2,5,10,20,1e9]; cnt,_=np.histogram(h,bins)
print(f"moments {len(h)}; history median {np.median(h):.2f} s; mean {h.mean():.2f}; share >= 5 s {(h>=5).mean()*100:.1f} %; share >= 2 s {(h>=2).mean()*100:.1f} %; max {h.max():.2f}")
print("bins [0,0.5) [0.5,1) [1,2) [2,5) [5,10) [10,20) 20+:", cnt.tolist(), " shares %:", np.round(cnt/len(h)*100,1).tolist())
# also: the upper bound if frames start is ignored (pose start only), and the whole manifest's anchor_t distribution
a=man.anchor_t.values; cnt2,_=np.histogram(a,bins); print(f"whole slice ({len(a)} moments) anchor_t as history (no-gap assumption): median {np.median(a):.2f}, share >= 5 s {(a>=5).mean()*100:.1f} %, bins {cnt2.tolist()}")
json.dump(dict(which=which, nclips=n, tol=dict(ego=EGO_TOL, frame=FR_TOL), hist_bins=bins[:-1], hist_counts=cnt.tolist(), median=float(np.median(h)), share_ge5=float((h>=5).mean()), whole_counts=cnt2.tolist(), whole_median=float(np.median(a)), whole_share_ge5=float((a>=5).mean()), clips_with_pose_gap=int((R.egaps.str.len()>0).sum()), clips_with_frame_gap=int((R.fgaps.str.len()>0).sum())), open(f"{OUT}/b2_{which}.json","w"), indent=1)
R.drop(columns=['egaps','fgaps']).to_csv(f"{OUT}/b2_{which}_clips.csv")
