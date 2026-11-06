"""Checks on the Step A grade before the long run."""
import json, sys, time
import numpy as np, torch
import grade2 as G2, grade_all as GA
SC = G2.SC; dev = 'cuda'
Vc = GA.vocab(); CTX = GA.CTX
# 1 frame alignment: the context's expert against the vocabulary's own path cache (rear axle -> centre)
z = np.load('{ops}/b2d_drivejepa/vocab/paths_cache.npz', allow_pickle=True); names = z['clip_names']
worst = 0
for cl in ('Accident_Town12_Route768_Weather14', 'HighwayCutIn_Town06_Route300_Weather14', 'VanillaNonSignalizedTurnEncounterStopsign_Town12_Route877_Weather19'):
    c = G2.load_clip(CTX, cl); ci = list(names).index(cl); ii = np.where(z['clip_id'] == ci)[0]; pos = {int(str(f).rsplit('|', 1)[1]) // 100: i for i, f in enumerate(c['frame_id'])}
    for i in ii:
        r = G2.to_centre_t(torch.from_numpy(z['rear'][i, 4::5].astype(np.float64))).numpy(); e = c['expert'][pos[int(z['frame'][i])]]
        worst = max(worst, float(np.abs(r[:, :2] - e[:, :2]).max()))
print('1 expert vs vocabulary path cache, largest gap (m):', round(worst, 4))
# 2 new terms against the scorer's own flags, same frames
cl = 'Accident_Town12_Route768_Weather14'; c = G2.load_clip(CTX, cl); sl = {k: v[10:18] for k, v in c.items()}
t0 = time.time(); order, valid, t, dref, S, P, te, Se, Pe = GA.grade_batch(sl, Vc); torch.cuda.synchronize(); dt = time.time() - t0
ctx = G2.collate2([sl], dev); g = G2.prepare2(ctx, dev); o = SC.score(ctx, Vc[order], dev, prepared=g, steps=True)
m = valid; col = (1 - o['collision_reactive'])[m].bool(); n = (t['n_ped'] + t['n_veh'] + t['n_static'])[m] > 0
print('2 paths', int(m.sum()), 'in', round(dt, 2), 's; scorer collision', int(col.sum()), 'ours', int(n.sum()), 'agree', float((col == n).float().mean()), '; scorer red', int((1 - o['red'])[m].sum()), 'ours', int(t['n_red'][m].sum()))
print('  scorer any off', float((o['off'].any(-1))[m].float().mean()), 'ours outside>0', float((t['outside'][m] > 0).float().mean()))
# 3 stop test on a real stop-sign clip, by hand-made paths
cl = 'VanillaNonSignalizedTurnEncounterStopsign_Town12_Route877_Weather19'; c = G2.load_clip(CTX, cl); x = np.load(f'{CTX}/{cl}.x.npz')
print('3 stop signs', x['ss_id'][:6], 'log fails', x['log_stop_fails'], 'target frames', int((x['ss_target'] >= 0).sum()), 'done frames', int(x['ss_done'].sum()), 'of', len(x['ss_target']))
# frame just before the sign: the first frame where a sign's centre is 6-14 m ahead
d = np.linalg.norm(x['ss_c'], axis=-1); ahead = (x['ss_c'][..., 0] > 3) & x['ss_valid'] & (d < 14) & (d > 5)
fr = np.where(ahead.any(1) & (x['ss_target'] < 0))[0]; print('  candidate frames', fr[:8], 'v0', c['v0'][fr[:8]].round(2))
if len(fr):
    i = int(fr[0]); sl = {k: v[i:i + 1] for k, v in c.items()}; ctx = G2.collate2([sl], dev); g = G2.prepare2(ctx, dev)
    sp = max(float(c['v0'][i]), 4.0); tt = np.arange(1, 9) * .5
    thru = np.stack([sp * tt, 0 * tt, 0 * tt], 1); stop = thru.copy(); stop[:, 0] = np.minimum(thru[:, 0], float(x['ss_c'][i][ahead[i]][0, 0]) - 2.0)
    gostop = stop.copy(); gostop[5:, 0] = stop[4, 0] + np.array([0., 3., 9.])         # stop, then go
    tr = G2.terms(g, torch.from_numpy(np.stack([thru, stop, gostop, c['expert'][i]])[None].astype(np.float32)).to(dev))
    print('  sign ahead (ego x, y)', x['ss_c'][i][ahead[i]].round(1).tolist(), '\n  n_stop: straight through', int(tr['n_stop'][0, 0]), ', stop short', int(tr['n_stop'][0, 1]), ', stop then go', int(tr['n_stop'][0, 2]), ', expert', int(tr['n_stop'][0, 3]))
