"""Per-size summary of grade.py's rows. Numbers only. Bootstrap is by clip, because frames of one clip are not independent."""
import json, numpy as np, sys
D="{ops}/b2d_drivejepa/vocab/b2d_grade/"
VS=["b2d_512","b2d_1024","b2d_2048","b2d_4096","b2d_8192","navsim_8192"]
def load(f): return [json.loads(l) for l in open(D+f)]
def kind(r):
    if r['v0']<0.5: return 'stopped'
    if r['exp_turn']>=20: return 'turn'
    return 'straight'
def table(rows,title):
    print(f"\n== {title}: {len(rows)} frames, {len({r['clip'] for r in rows})} clips; expert grade mean {np.mean([r['exp_grade'] for r in rows]):.3f}, expert>=.95 {np.mean([r['exp_grade']>=.95 for r in rows]):.3f}")
    print(f"{'vocab':12s} {'feas':>6s} {'best':>6s} {'>=1t':>6s} {'>=4t':>6s} {'med t':>6s} {'t/feas':>6s} {'spread':>6s} {'lat':>6s} {'<.95&exp ok':>11s}")
    for v in VS:
        x=[r['v'][v] for r in rows]
        fe=np.array([a['feasible'] for a in x]); b=np.array([a['best'] for a in x]); t=np.array([a['teachers'] for a in x])
        sp=np.array([a['spread'] for a in x if a['teachers']>1]); la=np.array([a.get('lat',0) for a in x if a['teachers']>1])
        ok=np.array([r['exp_grade']>=.95 for r in rows])
        print(f"{v:12s} {np.median(fe):6.0f} {b.mean():6.3f} {(t>=1).mean():6.3f} {(t>=4).mean():6.3f} {np.median(t):6.0f} {np.mean(t/np.maximum(fe,1)):6.3f} {sp.mean() if len(sp) else 0:6.2f} {la.mean() if len(la) else 0:6.2f} {((t==0)&ok).sum()/max(ok.sum(),1):11.3f}")
def boot(rows,a,b,key,n=4000):
    cl=sorted({r['clip'] for r in rows}); by={c:[r for r in rows if r['clip']==c] for c in cl}
    f={'best':lambda x:x['best'],'t1':lambda x:float(x['teachers']>=1),'t4':lambda x:float(x['teachers']>=4)}[key]
    d={c:(sum(f(r['v'][b])-f(r['v'][a]) for r in by[c]),len(by[c])) for c in cl}
    rng=np.random.default_rng(0); S=np.array([d[c][0] for c in cl]); N=np.array([d[c][1] for c in cl]); idx=rng.integers(0,len(cl),(n,len(cl)))
    m=S[idx].sum(1)/N[idx].sum(1); return S.sum()/N.sum(),np.percentile(m,2.5),np.percentile(m,97.5)
H=load('heldout.jsonl'); T=load('train.jsonl')
table(H,'HELD-OUT all'); table(T,'TRAIN all')
for k in ('straight','turn','stopped'):
    table([r for r in H if kind(r)==k],f'HELD-OUT {k}')
print('\nheld-out, step between sizes, clip bootstrap 95%')
for a,b in zip(VS[:4],VS[1:5]):
    print(a,'->',b,*[f"{k} {boot(H,a,b,k)[0]:+.3f} [{boot(H,a,b,k)[1]:+.3f},{boot(H,a,b,k)[2]:+.3f}]" for k in ('best','t1','t4')])
print('b2d_8192 vs navsim_8192',*[f"{k} {boot(H,'navsim_8192','b2d_8192',k)[0]:+.3f} [{boot(H,'navsim_8192','b2d_8192',k)[1]:+.3f},{boot(H,'navsim_8192','b2d_8192',k)[2]:+.3f}]" for k in ('best','t1','t4')])
print('\nheld-out frames with no teacher by size, by type')
for v in VS:
    print(v,{k:int(sum(r['v'][v]['teachers']==0 for r in H if kind(r)==k)) for k in ('straight','turn','stopped')})
print({k:sum(kind(r)==k for r in H) for k in ('straight','turn','stopped')})
print('\nheld-out clips with any no-teacher frame at b2d_1024 and which sizes rescue')
miss=[r for r in H if r['v']['b2d_512']['teachers']==0 and r['exp_grade']>=.95]
print(len(miss),'frames no teacher at 512 with good expert; v0 med',np.median([r['v0'] for r in miss]) if miss else None,'turn med',np.median([r['exp_turn'] for r in miss]) if miss else None)
