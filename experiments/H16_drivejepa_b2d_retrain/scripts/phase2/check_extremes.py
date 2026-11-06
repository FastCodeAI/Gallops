"""Spec v2 check 3: percentiles of the speed and acceleration fed to the model, and where the extremes come from."""
import json, numpy as np
D = '{data.b2d_retrain}'; RB = '{nas}/Bench2drive/rerank-b2d'; A = {'VehicleTurningRoutePedestrian_Town15_Route523_Weathe': 'VehicleTurningRoutePedestrian_Town15_Route523_Weather2'}
e = np.load(f'{D}/inputs/ego_inputs.npz'); sp = json.load(open('{ops}/b2d_drivejepa/phase1/splits/clips.json')); tr = set(sp['train']); m = np.array([str(c) in tr for c in e['clip']]); v, a, cl, fr = e['speed'][m], e['accel'][m], e['clip'][m], e['frame'][m]
print('speed p1 %.2f p99 %.2f ; accel p1 %.2f p99 %.2f ; frames %d' % (*np.percentile(v, [1, 99]), *np.percentile(a, [1, 99]), m.sum()))
print('share speed < 0: %.4f ; |accel| > 10: %.4f ; accel < -10 %.4f ; accel > 10 %.4f' % ((v < 0).mean(), (np.abs(a) > 10).mean(), (a < -10).mean(), (a > 10).mean()))
def show(i, why):
    c, n = str(cl[i]), int(fr[i]); z = np.load(f'{RB}/_boxes/{A.get(c, c)}.npz', allow_pickle=True); ef = z['ego_frame'].astype(int); k = int(np.where(ef == n)[0][0]); s = z['ego_speed'].astype(float); p = z['ego_center'].astype(float); lo = max(0, k - 3); hi = min(len(s), k + 3)
    print(why, c[:46], 'frame', n, 'speed', round(float(v[i]), 2), 'accel', round(float(a[i]), 2)); print('   anno speed', np.round(s[lo:hi], 2).tolist(), ' dist/0.1s', np.round(np.linalg.norm(np.diff(p[lo:hi], axis=0), axis=1) / .1, 2).tolist())
for i in np.argsort(v)[:3]: show(i, 'min speed')
for i in np.argsort(a)[:4]: show(i, 'min accel')
for i in np.argsort(-a)[:4]: show(i, 'max accel')
import collections
neg = collections.Counter(str(c).split('_')[0] for c in cl[v < -0.2]); big = collections.Counter(str(c).split('_')[0] for c in cl[np.abs(a) > 10]); print('speed < -0.2 by scenario', neg.most_common(5)); print('|accel| > 10 by scenario', big.most_common(8))
# how fast the ego stops across the whole dataset, at the 10 Hz anno step: is -15 m/s2 normal braking in this data?
d = np.diff(v[:0]); 
