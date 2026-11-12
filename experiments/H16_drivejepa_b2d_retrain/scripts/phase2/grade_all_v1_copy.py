"""Step A run: every feasible path of the 2,048 on every frame. One npz per clip in paddock-data/b2d_drivejepa_phase2/grade: frame table + every graded path's index, each term, d_ref.
    $PY grade_all.py [--shard i n] [--clips SUBSTR ...] [--batch 8] [--out DIR]"""
import argparse, csv, json, os, sys, time
import numpy as np, torch
import grade2 as G2
D = '{data.b2d_retrain}'; CTX = f'{D}/context'
V = '{ops}/b2d_drivejepa/vocab/b2d_2048.npy'
ALIAS = {'VehicleTurningRoutePedestrian_Town15_Route523_Weathe': 'VehicleTurningRoutePedestrian_Town15_Route523_Weather2'}
T = ('n_ped', 'n_veh', 'n_static', 'n_red', 'n_stop')


def vocab(dev='cuda'):
    p = torch.from_numpy(np.load(V)[:, 4::5, :].astype(np.float64)).to(dev)
    return G2.to_centre_t(p).float()


def feasible(Vc, v0):
    vi = Vc[:, 0, :2].norm(dim=-1) / 0.5; dy = torch.rad2deg(Vc[:, 0, 2].abs())
    return ((vi[None] - v0[:, None]).abs() <= 2.0) & (dy[None] < 15.0)


def grade_batch(c, Vc, dev='cuda'):
    """c: context dict of a few frames -> per-frame results."""
    ctx = G2.collate2([c], dev); g = G2.prepare2(ctx, dev); F = g['F']
    m = feasible(Vc, g['v0']); C = max(1, int(m.sum(1).max())); order = torch.argsort(m.int(), dim=1, descending=True, stable=True)[:, :C]; valid = torch.gather(m, 1, order)
    t = G2.terms(g, Vc[order]); dref = G2.d_ref_of(t, valid); S, P = G2.grade(t, dref)
    te = G2.terms(g, g['expert'][:, None]); Se, Pe = G2.grade(te, dref)
    return order, valid, t, dref, S, P, te, Se, Pe


def run_clip(clip, Vc, batch, out):
    name = ALIAS.get(clip, clip); c = G2.load_clip(CTX, name); F = len(c['frame_id'])
    R = {k: [] for k in ('idx', 'S', 'P', 'd', 'outside', 'off_share') + T}; n_per = np.zeros(F, np.int32); fr = {k: np.zeros(F, np.float32) for k in ('d_ref', 'exp_S', 'exp_P', 'exp_d', 'exp_outside', 'exp_off_share')}; fi = {k: np.zeros(F, np.int16) for k in ('exp_' + x for x in T)}
    for b in range(0, F, batch):
        sl = {k: v[b:b + batch] for k, v in c.items()}; order, valid, t, dref, S, P, te, Se, Pe = grade_batch(sl, Vc)
        v = valid.cpu().numpy(); n_per[b:b + len(v)] = v.sum(1)
        for k, x in (('idx', order), ('S', S), ('P', P), ('d', t['d']), ('outside', t['outside']), ('off_share', t['off_share'])) + tuple((k, t[k]) for k in T):
            R[k].append(x.cpu().numpy()[v])
        for k, x in (('d_ref', dref), ('exp_S', Se[:, 0]), ('exp_P', Pe[:, 0]), ('exp_d', te['d'][:, 0]), ('exp_outside', te['outside'][:, 0]), ('exp_off_share', te['off_share'][:, 0])): fr[k][b:b + len(v)] = x.cpu().numpy()
        for k in T: fi['exp_' + k][b:b + len(v)] = te[k][:, 0].cpu().numpy()
    o = dict(frame_id=c['frame_id'], v0=c['v0'], n_paths=n_per, idx=np.concatenate(R['idx']).astype(np.int16), S=np.concatenate(R['S']).astype(np.float32), P=np.concatenate(R['P']).astype(np.float16), d=np.concatenate(R['d']).astype(np.float32),
             outside=np.concatenate(R['outside']).astype(np.float32), off_share=np.concatenate(R['off_share']).astype(np.float16), **{k: np.concatenate(R[k]).astype(np.int8) for k in T}, **fr, **fi)
    p = f'{out}/{clip}.npz'; tmp = p + '.tmp.npz'; np.savez_compressed(tmp, **o); os.replace(tmp, p)
    return F, int(n_per.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--shard', type=int, nargs=2, default=(0, 1)); ap.add_argument('--clips', nargs='*'); ap.add_argument('--batch', type=int, default=8); ap.add_argument('--out', default=f'{D}/grade'); a = ap.parse_args()
    sp = json.load(open('{ops}/b2d_drivejepa/phase1/splits/clips.json')); clips = sorted(sp['train'] + sp['heldout'])[a.shard[0]::a.shard[1]]
    if a.clips: clips = [c for c in clips if any(s in c for s in a.clips)]
    clips = [c for c in clips if not os.path.exists(f'{a.out}/{c}.npz')]
    os.makedirs(a.out, exist_ok=True); Vc = vocab(); t0 = time.time(); nf = npth = 0
    print(len(clips), 'clips', flush=True)
    for i, c in enumerate(clips):
        f, n = run_clip(c, Vc, a.batch, a.out); nf += f; npth += n
        if i % 10 == 0 or i == len(clips) - 1: print(f'{i + 1}/{len(clips)} frames {nf} paths {npth} {time.time() - t0:.0f} s {npth / max(time.time() - t0, 1e-3):.0f} paths/s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)
    print('GRADE_DONE', nf, npth, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
