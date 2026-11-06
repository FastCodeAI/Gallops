"""Spec v2 report: three frames where the expert crosses into the oncoming lane to pass an obstacle. Expert's grade, best path that stops, teachers, under v1 and v2.
A path stops when its last 0.5 s segment is under 1.0 m/s (the stop rule). Frames: one per TwoWays scenario type, training clips, the frame with the most oncoming distance on the expert's path, ego moving at 3 m/s or more."""
import json, numpy as np, os
D = '{data.b2d_retrain}'; sp = json.load(open('{ops}/b2d_drivejepa/phase1/splits/clips.json')); T = ('n_ped', 'n_veh', 'n_static', 'n_red', 'n_stop')
V = np.load('{ops}/b2d_drivejepa/vocab/b2d_2048.npy')[:, 4::5, :2]; stopv = np.linalg.norm(V[:, 7] - V[:, 6], axis=1) / .5 < 1.0
print('vocabulary paths that stop:', int(stopv.sum()), 'of', len(V))
best = {}
for c in sorted(sp['train']):
    t = c.split('_')[0]
    if 'TwoWays' not in t: continue
    z = np.load(f'{D}/grade/{c}.npz')
    onc = z['exp_outside'] - z['exp_off_share'].astype(np.float32); ok = (z['v0'] >= 3) & (z['exp_off_share'] < .02) & (onc > .1)
    for i in np.where(ok)[0]:
        if t not in best or onc[i] > best[t][0]: best[t] = (float(onc[i]), c, int(i))
print({k: (round(v[0], 2), v[1][:40], v[2]) for k, v in best.items()})
out = []
for t, (o, c, i) in sorted(best.items(), key=lambda kv: -kv[1][0])[:3]:
    r = dict(scenario=t, clip=c, frame=str(np.load(f'{D}/grade/{c}.npz')['frame_id'][i]).rsplit('|', 1)[1])
    for tag, G in (('v1', 'grade'), ('v2', 'grade_v2')):
        z = np.load(f'{D}/{G}/{c}.npz'); of = np.r_[0, np.cumsum(z['n_paths'])]; sl = slice(of[i], of[i + 1]); idx = z['idx'][sl].astype(int); S = z['S'][sl]; st = stopv[idx]; te = S >= .95; yl = np.abs(V[idx, 7, 1])
        r[tag] = dict(v0=round(float(z['v0'][i]), 1), feasible=int(len(idx)), expert_S=round(float(z['exp_S'][i]), 3), expert_terms={k: int(z['exp_' + k][i]) for k in T if z['exp_' + k][i]}, expert_outside=round(float(z['exp_outside'][i]), 3), expert_P=round(float(z['exp_P'][i]), 3),
                      best_stop_S=round(float(S[st].max()), 3) if st.any() else None, stop_paths=int(st.sum()), best_path_S=round(float(S.max()), 3) if len(S) else None, teachers=int(te.sum()), teachers_that_stop=int((te & st).sum()), teachers_go=int((te & ~st).sum()), teachers_mean_end_y=round(float(yl[te].mean()), 1) if te.any() else None)
    out.append(r); print(json.dumps(r))
json.dump(out, open('{ops}/b2d_drivejepa/phase2/three_frames.json', 'w'), indent=1)
