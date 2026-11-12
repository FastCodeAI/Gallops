"""Per-route DS / RC for this run against Run 3A (rep a) and the released model (native-50). Only routes finished here."""
import glob, json, os, sys
import numpy as np
def rows(root):
    out = {}
    for f in glob.glob(f'{root}/*/*/results.json'):
        r = os.path.basename(os.path.dirname(f)).split('-')[0]
        try: x = json.load(open(f))['_checkpoint']['records'][0]
        except Exception: continue
        if 'scores' in x: out[r] = (x['scores']['score_composed'], x['scores']['score_route'], x['status'][:28], x['scores']['score_penalty'])
    return out
p2 = rows('{runs}/b2d_drivejepa/phase2/closed_loop/repa/bench2drive/drivejepa_p2')
r3 = rows('{runs_p3}/run3a_cmdfix/repa/bench2drive/wajepa_dj_run3a')
d = json.load(open('{ops}/native-50/dj_native50.json'))
rel = {str(k): v for k, v in (d.items() if isinstance(d, dict) else [])}
print(type(d), list(d)[:3] if isinstance(d, dict) else d[:1])
def ds(v):
    if isinstance(v, dict): return v.get('ds', v.get('driving_score', v.get('score_composed')))
    return v[0] if isinstance(v, (list, tuple)) else v
print('route   p2 DS  RC   | 3A DS  RC  | released DS')
a, b, c = [], [], []
for r, v in p2.items():
    q = r3.get(r); s = rel.get(r); print(r.ljust(7), round(v[0], 1), round(v[1]), v[2].ljust(28), '|', (round(q[0], 1), round(q[1])) if q else None, '|', ds(s) if s is not None else None)
    if q: a.append(v[0]); b.append(q[0])
    if s is not None and ds(s) is not None: c.append(float(ds(s)))
print(len(p2), 'routes; means p2', round(np.mean([v[0] for v in p2.values()]), 2), 'RC', round(np.mean([v[1] for v in p2.values()]), 2), '; 3A on same', round(np.mean(b), 2) if b else None, '; released on same', round(np.mean(c), 2) if c else None)
