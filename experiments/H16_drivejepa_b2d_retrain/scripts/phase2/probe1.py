import numpy as np, torch, json
import grade2 as G2, grade_all as GA
SC=G2.SC; dev='cuda'; Vc=GA.vocab()
sp=json.load(open('{ops}/b2d_drivejepa/phase1/splits/clips.json')); clips=sorted(sp['train'])[::45]
rows=[]
for cl in clips:
    c=G2.load_clip(GA.CTX,GA.ALIAS.get(cl,cl)); F=len(c['frame_id']); idx=np.linspace(0,F-1,min(8,F)).astype(int); sl={k:v[idx] for k,v in c.items()}
    order,valid,t,dref,S,P,te,Se,Pe=GA.grade_batch(sl,Vc)
    for i in range(len(idx)):
        m=valid[i]
        rows.append(dict(clip=cl[:34],f=int(idx[i]),v0=round(float(c['v0'][idx[i]]),1),n=int(m.sum()),expS=round(float(Se[i,0]),2),e=[int(te[k][i,0]) for k in GA.T],eo=round(float(te['outside'][i,0]),2),dref=round(float(dref[i]),1),dexp=round(float(te['d'][i,0]),1),col=round(float(((t['n_ped']+t['n_veh']+t['n_static'])[i][m]>0).float().mean()),2),outs=round(float((t['outside'][i][m]>0).float().mean()),2),clean=round(float(G2.clean({k:v[i:i+1,m] for k,v in t.items()}).float().mean()),2),teach=int((S[i][m]>=.95).sum())))
import collections
e=np.array([r['expS'] for r in rows]); print(len(rows),'frames; expert S>=.95',(e>=.95).mean(),'mean',e.mean()); print('col share mean',np.mean([r['col'] for r in rows]),'clean mean',np.mean([r['clean'] for r in rows]),'teachers>=1',np.mean([r['teach']>=1 for r in rows]),'>=4',np.mean([r['teach']>=4 for r in rows]))
bad=[r for r in rows if r['expS']<.95]
print(len(bad)); 
for r in bad[:26]: print(r)
