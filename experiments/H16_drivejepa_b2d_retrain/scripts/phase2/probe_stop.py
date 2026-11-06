"""How the 2 Hz path sees Bench2Drive's 0.1 m/s stop. Logged stops vs 0.5 s segment speed; the expert's n_stop by threshold, on the 8 poses and on its 10 Hz log."""
import json, os, sys, numpy as np, torch
import grade2 as G2, grade_all as GA
SC=G2.SC; dev='cuda'; C=GA.CTX
RB='{nas}/Bench2drive/rerank-b2d'
clips=[f[:-6] for f in sorted(os.listdir(C)) if f.endswith('.x.npz')]
stops=[c for c in clips if np.load(f'{C}/{c}.x.npz')['ss_valid'].any()]
print(len(stops),'clips with stop signs')
res={t:[0,0] for t in (0.1,0.25,0.5,1.0,1.5)}; nfr=0; seg_min=[]; passed=0
for cl in stops:
    c=G2.load_clip(C,cl); F=len(c['frame_id'])
    for b in range(0,F,32):
        sl={k:v[b:b+32] for k,v in c.items()}; ctx=G2.collate2([sl],dev); g=G2.prepare2(ctx,dev)
        e=g['expert'][:,None]; xy,yaw,vel=SC.densify(e); off,onc,near,nlon=G2.lanes2(g,xy,yaw)
        e40=ctx['expert40'].float()[:,None]
        P=torch.cat([torch.zeros_like(e40[:,:,:1,:2]),e40[...,:2]],2); v40=(P[:,:,1:]-P[:,:,:-1])/0.1
        nfr+=g['F']
        for t in res:
            G2.SPEED_T=t; res[t][0]+=int(G2.stop_count(g,xy,yaw,vel,near,nlon).sum())
        G2.SPEED_T=0.1
        # the same test, the same poses' lanes, with the 10 Hz log's positions and speed
        xo,yo=e40[...,:2],e40[...,2]; off2,onc2,n2,l2=G2.lanes2(g,xo,yo); res[0.1][1]+=int(G2.stop_count(g,xo,yo,v40,n2,l2).sum())
    z=np.load(f'{RB}/_boxes/{cl}.npz',allow_pickle=True); sp=z['ego_speed'].astype(float); cc=z['ego_center'].astype(float); 
    # every moment the logged ego drops under 0.1 m/s after moving: the lowest 0.5 s average speed around it, on each of the 5 grid phases
    low=np.where((sp[1:]<.1)&(sp[:-1]>=.1))[0]+1
    for i in low:
        best=[]
        for ph in range(5):
            s=[np.linalg.norm(cc[j+5]-cc[j])/.5 for j in range(max(0,i-5+ph-5),min(len(cc)-5,i+ph+1),5)]
            if s: best.append(min(s))
        if best: seg_min.append(max(best))     # worst phase
print('frames',nfr)
print('expert n_stop by threshold on 0.5 s segments:',{t:v[0] for t,v in res.items()},'; on the 10 Hz log at 0.1:',res[0.1][1])
s=np.array(seg_min); print('logged stops',len(s),'worst-phase lowest 0.5 s speed: p50 %.3f p90 %.3f p99 %.3f max %.3f'%tuple(np.percentile(s,[50,90,99,100])))
