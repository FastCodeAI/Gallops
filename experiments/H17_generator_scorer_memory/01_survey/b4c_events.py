"""B4b: the B4 event census (src/b4_events.py, rules UNCHANGED) on all 4,015 training clips and the 300 held-out clips.
Modes: nurec | braw | ordinary | heldout | graded_heldout | raw_heldout2 (b4c: the two approach-3 slices; loaders only) | picking (pilot, 2026-11-10: the 84
picking clips = their val_full rows; reconstructed clips through load_nurec (pool clipgt tracks), raw clips through load_raw with the census
egomotion zips and the census members where a file exists, else tracks = none as in heldout mode; rules unchanged). Only the loaders and the outputs differ from b4_events.py:
 - ordinary: raw clips of type 'ordinary'; obstacle members from {dataset_store}/approach3_tracks/ (fetched by b4b_fetch.py)
 - heldout: the 300 held-out clips (manifest_heldout.parquet); members from the held-out NAS folder where they exist, else tracks = none (ego-only events)
 - writes per-clip event times (out/b4b_<mode>_events.csv) so memory moments can be listed with seconds since the event."""
import pandas as pd, numpy as np, pyarrow.parquet as pq, json, os, sys, io, zipfile, time, math
sys.path.insert(0, "{paddock_sid}/Alpasim/analysis_3.0/Retrain-data/s0_census"); import r3_core as K
RD="{paddock_sid}/Alpasim/analysis_3.0/Retrain-data"; P="{dataset_store}"; HO="{retrain_store}/heldout"; OUT=os.environ["OUT"]; which=sys.argv[1]
RULES=dict(hard_brake="acc = d/dt of 0.5 s-smoothed speed at 10 Hz; event = >= 3 consecutive samples (0.3 s) with acc <= -THR m/s^2 and speed >= 2 m/s at the run start; runs < 1 s apart merged; THR in {3,4,6}",
  cut_in="vehicle track (not person/stroller/protruding_object/animal/unknown), at sample j: ego speed >= 2 m/s, track ahead 0 < lon <= 40 m in the ego frame, lateral distance to the ego's next-5 s path <= 1.25 m, track speed >= 1 m/s, motion direction within 45 deg of ego heading; and within the previous 2 s the same track had lateral distance >= 2.5 m while ahead (lon > 0); one event per track per 5 s",
  out_of_view_return="same track id: a gap >= 1.0 s between consecutive raw observations, with the track within 60 m of the ego at both ends of the gap; one event per gap",
  crossing="track speed >= 1 m/s, motion direction 60-120 deg from ego heading, ahead 0 < lon <= 50 m, lateral distance to the ego's next-5 s path <= 1.75 m; one event per track",
  memory_relevant_moment="a training moment (anchor) within (t_event, t_event + 5 s]")
NONVEH={'person','stroller','protruding_object','animal','unknown','rider','bicycle','motorcycle'}
A3="{experiment_store}"
if which=="heldout":
    tr=pd.read_parquet(f"{HO}/manifest_heldout.parquet"); ZR=f"{HO}/raw/labels/egomotion.offline"; MR=f"{HO}/raw/labels/obstacle.offline.members"
elif which=="graded_heldout":
    tr=pd.read_parquet(f"{A3}/slices/manifest_graded_heldout.parquet"); ZR=None; MR=None          # reconstructed clips: pool clipgt tracks (load_nurec)
elif which=="raw_heldout2":
    tr=pd.read_parquet(f"{A3}/slices/manifest_raw_heldout.parquet"); ZR=f"{HO}/raw/labels/egomotion.offline"; MR=f"{A3}/raw_heldout/raw/labels/obstacle.offline.members"
elif which=="picking":
    vf=pd.read_parquet(f"{RD}/s3_assembly/val_full/manifest_val_full.parquet"); pk=pd.read_parquet("{retrain_store}/picking/picking_set.parquet")
    tr=vf[vf.clip_uuid.isin(set(pk.clip_uuid))].copy(); assert tr.clip_uuid.nunique()==84; ZR=f"{P}/s0_census/raw/labels/egomotion.offline"; MR=f"{P}/s0_census/raw/labels/obstacle.offline.members"
else:
    tr=pd.read_parquet(f"{RD}/s3_assembly/manifest.parquet"); ZR=f"{P}/s0_census/raw/labels/egomotion.offline"; MR=f"{P}/s0_census/raw/labels/obstacle.offline.members"
tr['chunk']=tr.group.str.replace('chunk_','').astype(int)
if which=="nurec": clips=tr[tr.source=='nurec'].drop_duplicates('clip_uuid')
elif which=="braw": clips=tr[(tr.source=='raw')&(tr.type=='B_raw')].drop_duplicates('clip_uuid')
elif which=="ordinary": clips=tr[(tr.source=='raw')&(tr.type=='ordinary')].drop_duplicates('clip_uuid'); MR=f"{P}/approach3_tracks/obstacle.offline.members"
elif which in ("heldout","graded_heldout","raw_heldout2","picking"): clips=tr.drop_duplicates('clip_uuid')
else: raise SystemExit("mode?")
clips=clips.sort_values('chunk'); anchors_by_clip=tr.groupby('clip_uuid').anchor_t.apply(np.array).to_dict()
def load_nurec(u):
    d=f"{P}/s0_census/pool/{u}"; t=pq.read_table(f"{d}/clipgt__egomotion_estimate.parquet").flatten().flatten(); g=lambda n: t.column(n).to_numpy(zero_copy_only=False)
    ts=g("key.timestamp_micros").astype(np.int64); S=K.build_ego(ts/1e6, np.c_[g("egomotion_estimate.location.x"), g("egomotion_estimate.location.y")], K.yaw_q(*[g(f"egomotion_estimate.orientation.{k}") for k in "xyzw"]))
    o=pq.read_table(f"{d}/clipgt__obstacle.parquet").flatten().flatten(); h=lambda n: o.column(n).to_numpy(zero_copy_only=False)
    ob=pd.DataFrame({"tid": o.column("obstacle.trackline_id").to_pylist(), "ts": h("key.timestamp_micros").astype(np.int64)/1e6 - S["t0"], "x": h("obstacle.center.x"), "y": h("obstacle.center.y"), "yaw": K.yaw_q(*[h(f"obstacle.orientation.{k}") for k in "xyzw"]), "L": h("obstacle.size.x"), "W": h("obstacle.size.y"), "cat": o.column("obstacle.category").to_pylist()})
    return S, ob, True
ZC={}; EMPTY=pd.DataFrame({"tid": pd.Series([],dtype=str), "ts": pd.Series([],dtype=float), "x": pd.Series([],dtype=float), "y": pd.Series([],dtype=float), "yaw": pd.Series([],dtype=float), "L": pd.Series([],dtype=float), "W": pd.Series([],dtype=float), "cat": pd.Series([],dtype=str)})
def load_raw(u, c):
    if c not in ZC: ZC.clear(); ZC[c]=zipfile.ZipFile(f"{ZR}/egomotion.offline.chunk_{c:04d}.zip")
    e=pd.read_parquet(io.BytesIO(ZC[c].read(f"{u}.egomotion.offline.parquet"))).sort_values("timestamp"); ts=e.timestamp.to_numpy()/1e6; xy=e[["x","y"]].to_numpy(); yaw=np.unwrap(K.yaw_q(e.qx.to_numpy(), e.qy.to_numpy(), e.qz.to_numpy(), e.qw.to_numpy()))
    S=K.build_ego(ts, xy, yaw); mp=f"{MR}/chunk_{c:04d}/{u}.obstacle.offline.parquet"
    if not os.path.exists(mp):
        if which in ("heldout","raw_heldout2","picking"): return S, EMPTY.copy(), False
        raise FileNotFoundError(mp)
    d=pd.read_parquet(mp); assert (d.reference_frame=="rig").all()
    t=d.reference_frame_timestamp_us.to_numpy().astype(np.float64)/1e6; ex,ey,eyaw=np.interp(t,ts,xy[:,0]),np.interp(t,ts,xy[:,1]),np.interp(t,ts,yaw); cc,ss=np.cos(eyaw),np.sin(eyaw); x,y=d.center_x.to_numpy(),d.center_y.to_numpy()
    ob=pd.DataFrame({"tid": d.track_id.astype(str), "ts": d.timestamp_us.to_numpy().astype(np.float64)/1e6 - ts[0], "x": ex+cc*x-ss*y, "y": ey+ss*x+cc*y, "yaw": K.yaw_q(d.orientation_x.to_numpy(), d.orientation_y.to_numpy(), d.orientation_z.to_numpy(), d.orientation_w.to_numpy())+eyaw, "L": d.size_x.to_numpy(), "W": d.size_y.to_numpy(), "cat": d.label_class.astype(str)})
    return S, ob, True
def runs(mask, minlen, merge):
    idx=np.where(mask)[0]; ev=[]
    if len(idx)==0: return ev
    s=idx[0]; p=idx[0]
    for i in idx[1:]:
        if i-p>merge: 
            if p-s+1>=minlen: ev.append((s,p))
            s=i
        p=i
    if p-s+1>=minlen: ev.append((s,p))
    return ev
def lat_to_path(pts, path):
    """min distance from each point to the polyline path (n,2) -> (m,)"""
    A,B=path[:-1],path[1:]; AB=B-A; out=np.empty(len(pts))
    for i,p in enumerate(pts):
        AP=p-A; u=np.clip((AP*AB).sum(1)/np.maximum((AB*AB).sum(1),1e-9),0,1); out[i]=np.min(np.linalg.norm(A+u[:,None]*AB-p,axis=1))
    return out
def events(S, ob, anchors):
    n=len(S['t']); sp=S['speed']; acc=S['acc']; E=dict(hb3=[],hb4=[],hb6=[],cut_in=[],oov=[],crossing=[]); cats={}
    for thr,key in ((3,'hb3'),(4,'hb4'),(6,'hb6')):
        for s,p in runs(acc<=-thr, 3, 10):
            if sp[s]>=2.0: E[key].append(float(S['t'][s]))
    # raw observation gaps (before interpolation)
    for tid,r in ob.sort_values('ts').groupby('tid', sort=False):
        a=r.ts.to_numpy(); x=r.x.to_numpy(); y=r.y.to_numpy(); 
        for i in np.where(np.diff(a)>=1.0)[0]:
            ok=True
            for q in (i,i+1):
                k=int(round(a[q]/K.DT)); 
                if not (0<=k<n) or np.hypot(x[q]-S['xy'][k,0], y[q]-S['xy'][k,1])>60: ok=False
            if ok: E['oov'].append(float(a[i+1]))
    S=K.build_tracks(S, ob)
    for tid,T in S['tracks'].items():
        cats[T['cat']]=cats.get(T['cat'],0)+1
        k0=int(round(T['t'][0]/K.DT)); m=len(T['t']); k1=k0+m
        if k1>n: m=n-k0; k1=n
        exy=S['xy'][k0:k1]; eyaw=S['yaw'][k0:k1]; esp=sp[k0:k1]; txy=T['xy'][:m]
        d=txy-exy; c,s_=np.cos(eyaw),np.sin(eyaw); lon=c*d[:,0]+s_*d[:,1]
        latp=np.empty(m)
        for j in range(m):
            path=S['xy'][k0+j:min(k0+j+51,n)]
            latp[j]=lat_to_path(txy[j:j+1], path)[0] if len(path)>=2 else abs(-s_[j]*d[j,0]+c[j]*d[j,1])
        hd=K.wrap(T['mdir'][:m]-eyaw); v=T['v'][:m]
        if T['cat'] not in NONVEH:
            ok=(esp>=2)&(lon>0)&(lon<=40)&(latp<=1.25)&(v>=1)&(np.abs(hd)<=math.radians(45)); last=-1000
            for j in np.where(ok)[0]:
                if j-last<50: continue
                w=slice(max(0,j-20),j)
                if np.any((latp[w]>=2.5)&(lon[w]>0)): E['cut_in'].append(float(T['t'][j])); last=j
        okc=(v>=1)&(np.abs(hd)>=math.radians(60))&(np.abs(hd)<=math.radians(120))&(lon>0)&(lon<=50)&(latp<=1.75)
        if okc.any(): E['crossing'].append(float(T['t'][np.argmax(okc)]))
    mom={}
    for key,ts in E.items():
        rel=np.zeros(len(anchors),bool)
        for te in ts: rel|=(anchors>te)&(anchors<=te+5)
        mom[key]=int(rel.sum())
    return E, mom, cats, len(S['tracks'])
t0=time.time(); rows=[]; erows=[]; allE={k:0 for k in ('hb3','hb4','hb6','cut_in','oov','crossing')}; allM=dict(allE); hours=0; nmom=0; catsum={}; ntr=0; fails=0; notrack=0; hours_tracks=0; nmom_tracks=0
for i,r in enumerate(clips.itertuples()):
    try:
        S,ob,has=load_nurec(r.clip_uuid) if (which in ("nurec","graded_heldout") or (which=="picking" and r.source=="nurec")) else load_raw(r.clip_uuid, r.chunk)
        an=anchors_by_clip[r.clip_uuid]; E,mom,cats,nt=events(S,ob,an)
    except Exception as e:
        fails+=1; print("FAIL", r.clip_uuid[:8], repr(e)[:200], flush=True); continue
    hours+=S['dur']/3600; nmom+=len(an); ntr+=nt
    if has: hours_tracks+=S['dur']/3600; nmom_tracks+=len(an)
    else: notrack+=1
    for k in allE: allE[k]+=len(E[k]); allM[k]+=mom[k]
    for k,v in cats.items(): catsum[k]=catsum.get(k,0)+v
    for k,ts in E.items():
        for te in ts: erows.append(dict(clip=r.clip_uuid, type=r.type, event=k, t_event=te))
    rows.append(dict(clip=r.clip_uuid, type=r.type, dur=S['dur'], has_tracks=has, tracks=nt, n_moments=len(an), **{k:len(E[k]) for k in E}, **{'mom_'+k:mom[k] for k in mom}))
    if (i+1)%50==0: print(f"{i+1}/{len(clips)} clips {time.time()-t0:.0f}s hours {hours:.2f} {allE}", flush=True)
print(f"DONE {which}: {len(rows)} clips ok ({notrack} without tracks), {fails} failed, {hours:.2f} clip-hours ({hours_tracks:.2f} with tracks), {nmom} moments ({nmom_tracks} in clips with tracks), {ntr} tracks, {time.time()-t0:.0f}s")
print("track categories:", dict(sorted(catsum.items(), key=lambda kv:-kv[1])))
res=dict(which=which, clips=len(rows), clips_without_tracks=notrack, failed=fails, clip_hours=hours, clip_hours_with_tracks=hours_tracks, moments=nmom, moments_in_clips_with_tracks=nmom_tracks, tracks=ntr, events=allE, rate_per_hour={k: v/hours for k,v in allE.items()}, rate_per_hour_tracked={k: (v/hours_tracks if hours_tracks else None) for k,v in allE.items()}, clips_with_event={k:int((pd.DataFrame(rows)[k]>0).sum()) for k in allE}, memory_relevant_moments=allM, memory_relevant_share={k: allM[k]/nmom for k in allM}, categories=catsum, rules=RULES)
print(json.dumps({k:v for k,v in res.items() if k!='rules'}, indent=1)); json.dump(res, open(f"{OUT}/b4b_{which}.json","w"), indent=1); pd.DataFrame(rows).to_csv(f"{OUT}/b4b_{which}_clips.csv", index=False)
pd.DataFrame(erows, columns=['clip','type','event','t_event']).to_csv(f"{OUT}/b4b_{which}_events.csv", index=False)
