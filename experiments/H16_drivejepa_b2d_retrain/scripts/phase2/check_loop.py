"""Step B checks 1-3. 1: encoder sha + tensors. 2: vocabulary paths through the training path's grade against Step A's saved rows. 3: inputs."""
import json, sys, time
import numpy as np, torch
import train_b2d as TB
D2, G2 = TB.D2, TB.G2; dev = 'cuda'
T = D2.Tables('train'); ds = D2.Samples(T); R = {}
ag, cfg = TB.build(None, dev); sd = torch.load(TB.ENC, map_location='cpu'); e = ag._pad_model._backbone.img_backbone.state_dict()
R['check1'] = dict(sha256=TB.sha256(TB.ENC), expected=TB.ENC_SHA, match=TB.sha256(TB.ENC) == TB.ENC_SHA, tensors_equal=all(torch.equal(e[k].cpu(), v) for k, v in sd.items()), tensors=len(sd))
del ag; torch.cuda.empty_cache()
# 2
rng = np.random.default_rng(0); ks = rng.choice(len(ds), 192, replace=False); V = torch.from_numpy(np.load(TB.VOCAB)[:, 4::5, :]).to(dev); worst = dict(S=0.0, d=0.0, outside=0.0, n=0, paths=0, frames=0); cache = {}
for b0 in range(0, len(ks), 32):
    b = D2.collate([ds[int(k)] for k in ks[b0:b0 + 32]]); g = G2.prepare2(b['ctx'], dev); g['route_len'] = b['route'].to(dev); rows = []; idxs = []
    for k in ks[b0:b0 + 32]:
        clip, j, *_ = T.items[int(k)]
        z = cache.setdefault(clip, dict(np.load(f'{D2.D}/grade_v2/{clip}.npz'))); o = np.r_[0, np.cumsum(z['n_paths'])]; sl = slice(o[j], o[j + 1]); take = rng.permutation(np.arange(sl.start, sl.stop))[:32]
        rows.append((z, take)); idxs.append(z['idx'][take].astype(np.int64))
    C = max(len(i) for i in idxs); ix = torch.zeros(len(idxs), C, dtype=torch.long, device=dev); val = torch.zeros(len(idxs), C, dtype=torch.bool, device=dev)
    for i, x in enumerate(idxs): ix[i, :len(x)] = torch.from_numpy(x).to(dev); val[i, :len(x)] = True
    t = G2.terms(g, G2.to_centre_t(V[ix]), steps=True); S, P = G2.grade(t, b['dref'].to(dev))
    for i, (z, take) in enumerate(rows):
        m = val[i]; worst['frames'] += 1; worst['paths'] += int(m.sum())
        worst['S'] = max(worst['S'], float(np.abs(S[i][m].cpu().numpy() - z['S'][take]).max())) if m.any() else worst['S']
        if m.any():
            worst['d'] = max(worst['d'], float(np.abs(t['d'][i][m].cpu().numpy() - z['d'][take]).max())); worst['outside'] = max(worst['outside'], float(np.abs(t['outside'][i][m].cpu().numpy() - z['outside'][take]).max()))
            worst['n'] = max(worst['n'], max(int(np.abs(t[q][i][m].cpu().numpy() - z[q][take]).max()) for q in ('n_ped', 'n_veh', 'n_static', 'n_red', 'n_stop')))
R['check2'] = worst
# 3
from data import FrameCache, _to_tensor
fc = FrameCache(); same = True
for k in rng.choice(len(ds), 64, replace=False):
    clip, j, i, *_ = T.items[int(k)]; r = ds[int(k)]; same &= torch.equal(r['new'], _to_tensor(fc.frame(clip, i))) and torch.equal(r['old'], _to_tensor(fc.frame(clip, max(i - 1, 0))))
ei = np.array([it[4] for it in T.items]); oh = T.one_hot[ei]; sp, ac = T.speed[ei], T.accel[ei]
R['check3'] = dict(crop_bit_identical_to_phase1_cache=bool(same), frames=len(ei), first_frames_repeated=int(sum(1 for it in T.items if it[2] == 0)), command_mix=dict(zip(('left', 'straight', 'right', 'unknown'), [float(x) for x in oh.mean(0)])), speed=dict(min=float(sp.min()), p50=float(np.median(sp)), p99=float(np.percentile(sp, 99)), max=float(sp.max())), accel=dict(min=float(ac.min()), p1=float(np.percentile(ac, 1)), p50=float(np.median(ac)), p99=float(np.percentile(ac, 99)), max=float(ac.max())))
print(json.dumps(R, indent=1)); json.dump(R, open('{ops}/b2d_drivejepa/phase2/checks_1_3_v2.json', 'w'), indent=1)
