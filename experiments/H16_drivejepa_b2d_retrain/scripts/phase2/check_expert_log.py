"""Spec v2: the expert's collisions against its log, and where the ego box sits after the rear-axle shift.
  path   the grade's view: the expert's 8 poses, densified, scorer boxes and margins (0.3 m on static objects)
  log    the 10 Hz log: same agents and margins, true positions and headings
  true   the 10 Hz log with the real boxes, no margin
Agents here are physical hits, same speed rule (ego at 0.1 m/s or more). Followers are counted apart: they are the reactive model, not contact."""
import gzip, json, os, sys
import numpy as np, torch
import grade2 as G2, grade_all as GA
SC = G2.SC; dev = 'cuda'; sp = json.load(open('{ops}/b2d_drivejepa/phase1/splits/clips.json')); clips = sp['train'] + sp['heldout']
NAMES = ('pedestrian', 'vehicle', 'static')
cnt = {k: {n: 0 for n in NAMES} for k in ('path', 'log', 'true', 'path_and_log', 'path_and_true', 'path_not_true')}; fr = 0; follower_only = 0; any_path = any_true = both = 0; hit_gap = []
for ci, clip in enumerate(clips):
    c = G2.load_clip(GA.CTX, GA.ALIAS.get(clip, clip)); F = len(c['frame_id'])
    for b in range(0, F, 32):
        ctx = G2.collate2([{k: v[b:b + 32] for k, v in c.items()}], dev); g = G2.prepare2(ctx, dev); g['route_len'] = torch.full((g['F'],), float(GA.ROUTE[clip]), device=dev); fr += g['F']
        av = g['valid'][:, :, 1:].transpose(1, 2)                                                   # (F, K, A)
        xy, yaw, vel = SC.densify(g['expert'][:, None]); xy, yaw, spd = xy[:, 0], yaw[:, 0], vel[:, 0].norm(dim=-1)
        ab = SC._agent_box(g, 1); box = lambda p, h, ahl, ahw: SC.overlap(p[..., 0, None], p[..., 1, None], torch.cos(h)[..., None], torch.sin(h)[..., None], SC.HL, SC.HW, ab[0][:, 0], ab[1][:, 0], ab[2][:, 0], ab[3][:, 0], ahl, ahw)
        hp = box(xy, yaw, ab[4][:, 0], ab[5][:, 0]) & av & (spd >= G2.COL_EPS)[..., None]
        e40 = ctx['expert40'].to(dev).float(); lx, ly = e40[..., :2], e40[..., 2]
        P = torch.cat([torch.zeros_like(lx[:, :1]), lx], 1); ls = (P[:, 1:] - P[:, :-1]).norm(dim=-1) / 0.1
        hl = box(lx, ly, ab[4][:, 0], ab[5][:, 0]) & av & (ls >= G2.COL_EPS)[..., None]
        ht = box(lx, ly, (g['alen'] / 2)[:, None, :], (g['awid'] / 2)[:, None, :]) & av & (ls >= G2.COL_EPS)[..., None]
        ty = g['ctype']
        te = G2.terms(g, g['expert'][:, None]); anyp = hp.any(1); anyl = hl.any(1); anyt = ht.any(1)      # (F, A)
        for q, n in enumerate(NAMES):
            m = ty == q; fp, fl, ft = (anyp & m).any(1), (anyl & m).any(1), (anyt & m).any(1)
            cnt['path'][n] += int(fp.sum()); cnt['log'][n] += int(fl.sum()); cnt['true'][n] += int(ft.sum()); cnt['path_and_log'][n] += int((fp & fl).sum()); cnt['path_and_true'][n] += int((fp & ft).sum()); cnt['path_not_true'][n] += int((fp & ~ft).sum())
        follower_only += int(((te['n_veh'][:, 0] > 0) & ~(anyp & (ty == 1)).any(1)).sum())
    if ci % 200 == 0: print(ci, fr, flush=True)
print(json.dumps(dict(frames=fr, frames_with_hit=cnt, expert_vehicle_infraction_with_no_path_contact=follower_only), indent=1))
# ego box
B = '{data.b2d_clips}'; RB = '{nas}/Bench2drive/rerank-b2d'; rng = np.random.default_rng(0); rows = []
for clip in rng.choice(sp['train'], 40, replace=False):
    clip = str(clip); name = GA.ALIAS.get(clip, clip); d = next(f'{B}/{s}/{clip}/anno' for s in ('b2d_base', 'b2d_pilot') if os.path.isdir(f'{B}/{s}/{clip}/anno')); z = np.load(f'{RB}/_boxes/{name}.npz', allow_pickle=True)
    for i in rng.choice(len(z['ego_frame']), 6, replace=False):
        n = int(z['ego_frame'][i]); a = json.load(gzip.open(f'{d}/{n:05d}.json.gz')); e = next(x for x in a['bounding_boxes'] if x['class'] == 'ego_vehicle')
        cen = np.array(e['center'][:2]) * [1, -1]; loc = np.array(e['location'][:2]) * [1, -1]; h = float(z['ego_heading'][i]); f = np.array([np.cos(h), np.sin(h)]); l = np.array([-f[1], f[0]]); c = z['ego_center'][i].astype(float)
        rows.append(((cen - c) @ f, (cen - c) @ l, (loc - c) @ f, (loc - c) @ l, *e['extent'][:2]))
r = np.array(rows); print('ego_center used by the scorer vs the anno box centre, along / across (m): mean %.4f %.4f, largest %.4f %.4f' % (r[:, 0].mean(), r[:, 1].mean(), np.abs(r[:, 0]).max(), np.abs(r[:, 1]).max()))
print('anno actor location relative to the box centre, along / across: mean %.4f %.4f' % (r[:, 2].mean(), r[:, 3].mean())); print('anno half extent x, y: %.4f %.4f; scorer half box %.3f %.3f; rear-axle offset %.4f' % (r[:, 4].mean(), r[:, 5].mean(), SC.HL, SC.HW, G2.OFF_REAR))
