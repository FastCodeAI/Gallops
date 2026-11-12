"""CPU only. (1) every ported module imports under the training interpreter against the real tree, batch fix included.
(2) grade_all's path, on a few real frames, reproduces what Step A v2 recorded."""
import importlib, json, sys, time
t0 = time.time()
import numpy as np, torch
torch.set_num_threads(2)
from gallops.drivejepa_b2d_retrain import wiring as W
for m in ('scorer', 'grade', 'frames', 'ade', 'samples', 'build_cache', 'build_vocab', 'train_lora', 'train_stage2'):
    importlib.import_module(f'gallops.drivejepa_b2d_retrain.{m}'); print('imported', m, round(time.time() - t0), 's', flush=True)
import gallops.drivejepa_b2d_retrain.train_stage2 as T
print('agent class', T.B2DAgent.__mro__[1].__name__, 'compute_score' in T.B2DAgent.__dict__, flush=True)
from gallops.drivejepa_b2d_retrain import grade_all as GA, grade as G2
print('imported grade_all', len(GA.ROUTE), 'routes', flush=True)
Vc = GA.vocab('cpu')
worst = {}
for clip in sys.argv[1:]:
    name = GA.ALIAS.get(clip, clip)
    c = G2.load_clip(GA.CTX, name); z = np.load(f'{GA.D}/grade_v2/{clip}.npz')
    F = len(c['frame_id']); pick = np.linspace(0, F - 1, 3).astype(int)
    off = np.r_[0, np.cumsum(z['n_paths'])]
    for i in pick:
        sl = {k: v[i:i + 1] for k, v in c.items()}
        order, valid, t, dref, S, P, te, Se, Pe = GA.grade_batch(sl, Vc, GA.ROUTE[clip], dev='cpu')
        v = valid[0].numpy(); idx = order[0].numpy()[v]; s = S[0].numpy()[v]
        lo, hi = off[i], off[i + 1]
        assert (z['idx'][lo:hi] == idx).all(), (clip, i, 'different feasible set')
        d = float(np.abs(z['S'][lo:hi] - s).max()) if len(s) else 0.0
        e = abs(float(Se[0, 0]) - float(z['exp_S'][i]))
        n = sum(int(np.abs(z[k][lo:hi].astype(int) - t[k][0].numpy()[v]).sum()) for k in GA.T)
        print(clip[:44], 'frame', i, 'paths', len(s), 'max|dS|', f'{d:.2e}', 'expert |dS|', f'{e:.2e}', 'count diffs', n, flush=True)
        worst[clip] = max(worst.get(clip, 0.0), d, e)
print('LARGEST', max(worst.values()), 'seconds', round(time.time() - t0))
