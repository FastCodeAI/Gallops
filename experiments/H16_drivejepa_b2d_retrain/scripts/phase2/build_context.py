"""Scorer context for every Phase 2 frame: mtd/build_context_b2d.build_clip unchanged, on all 944 clips and all 39,148 frames, plus a sidecar per clip with what the Step A grade needs and MTD's context lacks:
  Parking lanes (Bench2Drive's OutsideRouteLanesTest counts them) and stop signs with the RunningStopTest state at t, replayed from the clip's start on the logged ego at 10 Hz.
Waypoints: nearest Driving lane point projected on the lane, then 8 more at 0.5 m along its direction (a straight stand-in for carla's wp.next).
    <alpamayo15 python> build_context.py [--workers 6] [--clips SUBSTR ...]"""
import argparse, csv, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '{ops}/mtd')
import build_context_b2d as BC
D = '{data.b2d_retrain}'
OUT = f'{D}/context'; ALIAS = {'VehicleTurningRoutePedestrian_Town15_Route523_Weathe': 'VehicleTurningRoutePedestrian_Town15_Route523_Weather2'}
PROX, SPEED_T, STEP, NW, MULT, BACK = 4.0, 0.1, 0.5, 9, 1.2, -0.17       # RunningStopTest constants; NW = int(4 / 0.5) + 1 waypoints
_P = {}


def parking(town):
    if _P.get('town') != town:
        f = f'{D}/inputs/park_{town}.npz'
        if not os.path.exists(f):
            m = np.load(f'{BC.RB}/_map/{town}.npz'); d = m['lane_type'] == 1
            xy = m['xy'][d].astype(np.float64); grp = m['road_id'][d].astype(np.int64) * 1000 + m['lane_id'][d].astype(np.int64); keep = np.zeros(len(xy), bool)
            if len(xy):
                pairs = cKDTree(xy).query_pairs(BC.LANE_THIN, output_type='ndarray'); pairs = pairs[grp[pairs[:, 0]] == grp[pairs[:, 1]]]
                nb = [[] for _ in range(len(xy))]
                for i, j in pairs: nb[i].append(j); nb[j].append(i)
                for i in range(len(xy)):
                    if not any(keep[j] for j in nb[i]): keep[i] = True
            tmp = f + f'.{os.getpid()}.tmp.npz'; np.savez(tmp, xy=xy[keep].astype(np.float32), hw=m['half_width'][d][keep]); os.replace(tmp, f)
        z = np.load(f); _P.update(town=town, xy=z['xy'].astype(np.float64), hw=z['hw'], T=cKDTree(z['xy'].astype(np.float64)) if len(z['xy']) else None)
    return _P


def stop_replay(z, Ld, LT, ids, cen, ext):
    """RunningStopTest.update on the logged ego. -> {frame: (target index or -1, stop completed, last failed index or -1)} at each 10 Hz ego frame, and the failures."""
    ef = z['ego_frame'].astype(int); c = z['ego_center'].astype(np.float64); h = z['ego_heading'].astype(np.float64); v = z['ego_speed'].astype(np.float64)
    tgt, done, lastf, out, fails = -1, False, -1, {}, []
    for i, n in enumerate(ef):
        _, j = LT.query(c[i]); t = np.array([math.cos(Ld['yaw'][j]), math.sin(Ld['yaw'][j])]); lon = float(np.clip((c[i] - Ld['xy'][j]) @ t, -0.5, 0.5)); wp0 = Ld['xy'][j] + lon * t
        wps = wp0[None] + (STEP * np.arange(NW))[:, None] * t[None]
        rel = wps[:, None] - cen[None]
        aff = (np.linalg.norm(cen - wp0, axis=1) <= PROX) & ((np.abs(rel[..., 0]) < MULT * ext[None, :, 0]) & (np.abs(rel[..., 1]) < MULT * ext[None, :, 1])).any(0)
        f = np.array([math.cos(h[i]), math.sin(h[i])])
        vel = (c[i + 1] - c[i]) / 0.1 if i + 1 < len(ef) else f * v[i]
        if tgt < 0:
            if vel @ f >= BACK and f @ t >= BACK and aff.any(): tgt = int(np.argmax(aff))
        else:
            if v[i] < SPEED_T: done = True
            if not aff[tgt]:
                if not done and lastf != tgt: lastf = tgt; fails.append((int(n), int(ids[tgt])))
                tgt, done = -1, False
        out[int(n)] = (tgt, done, lastf)
    return out, fails


def extras(clip, frames):
    town = clip.split('_')[1]; Ld, LT, _ = BC.lanes(town); pk = parking(town)
    z = dict(np.load(f'{BC.RB}/_boxes/{clip}.npz', allow_pickle=True)); ef = {int(f): i for i, f in enumerate(z['ego_frame'])}
    st = {}
    for j in np.where(z['sig_kind'] == 6)[0]: st.setdefault(int(z['sig_id'][j]), (z['sig_trig_center'][j].astype(np.float64), z['sig_trig_extent'][j].astype(np.float64)))
    ids = np.array(list(st), np.int64); cen = np.array([st[i][0] for i in ids]).reshape(-1, 2); ext = np.array([st[i][1] for i in ids]).reshape(-1, 2)
    state, fails = stop_replay(z, Ld, LT, ids, cen, ext) if len(ids) else ({}, [])
    frames = [n for n in frames if n in ef]; F = len(frames); Q = max(1, len(ids)); rows = []
    for n in frames:
        i = ef[n]; x0, y0 = z['ego_center'][i].astype(np.float64); h0 = float(z['ego_heading'][i]); c0, s0 = math.cos(h0), math.sin(h0)
        toE = lambda p: np.stack([(p[..., 0] - x0) * c0 + (p[..., 1] - y0) * s0, -(p[..., 0] - x0) * s0 + (p[..., 1] - y0) * c0], -1)   # noqa: E731
        R = float(np.clip(4 * abs(float(z['ego_speed'][i])) + 30, 60, 100))
        li = np.array(pk['T'].query_ball_point((x0, y0), R), dtype=np.int64) if pk['T'] is not None else np.zeros(0, np.int64)
        rows.append((toE(pk['xy'][li]).astype(np.float32) if len(li) else np.zeros((0, 2), np.float32), pk['hw'][li].astype(np.float16) if len(li) else np.zeros(0, np.float16),
                     toE(cen).astype(np.float32) if len(ids) else np.zeros((0, 2), np.float32), np.array([c0, -s0], np.float32), np.array([s0, c0], np.float32), state.get(n, (-1, False, -1))))
    Lp = max(1, max(len(r[0]) for r in rows)) if rows else 1
    o = dict(frame_id=np.array([f'b2d|{clip}|{n * 100}' for n in frames]), park_pts=np.zeros((F, Lp, 2), np.float32), park_hw=np.zeros((F, Lp), np.float16), park_valid=np.zeros((F, Lp), bool),
             ss_c=np.zeros((F, Q, 2), np.float32), ss_ext=np.tile(np.pad(ext, ((0, Q - len(ids)), (0, 0))).astype(np.float32), (F, 1, 1)), ss_valid=np.zeros((F, Q), bool), ss_ax=np.zeros((F, 2), np.float32),
             ss_ay=np.zeros((F, 2), np.float32), ss_target=np.full(F, -1, np.int16), ss_done=np.zeros(F, bool), ss_lastfail=np.full(F, -1, np.int16), ss_id=np.pad(ids, (0, Q - len(ids))), log_stop_fails=np.array(fails, np.int64).reshape(-1, 2))
    for i, r in enumerate(rows):
        o['park_pts'][i, :len(r[0])] = r[0]; o['park_hw'][i, :len(r[1])] = r[1]; o['park_valid'][i, :len(r[0])] = True
        o['ss_c'][i, :len(r[2])] = r[2]; o['ss_valid'][i, :len(r[2])] = True; o['ss_ax'][i] = r[3]; o['ss_ay'][i] = r[4]; o['ss_target'][i], o['ss_done'][i], o['ss_lastfail'][i] = r[5]
    p = f'{OUT}/{clip}.x.npz'; tmp = p + '.tmp.npz'; np.savez_compressed(tmp, **o); os.replace(tmp, p)
    return F, len(ids), len(fails)


def job(a):
    clip, frames = a
    c, n, b = BC.build_clip((clip, frames, OUT))
    F, q, fails = extras(clip, frames)
    return clip, n, F, q, fails, b


def init():
    BC._init(); BC.BLOCKED = json.load(open(f'{D}/inputs/blocked_all.json'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--clips', nargs='*'); a = ap.parse_args(); t0 = time.time()
    fr = {}
    for r in csv.DictReader(open('{ops}/b2d_drivejepa/vocab/frames_used.csv')): fr.setdefault(ALIAS.get(r['clip'], r['clip']), []).append(int(r['frame']))
    jobs = sorted(((c, sorted(f)) for c, f in fr.items() if (not a.clips or any(s in c for s in a.clips)) and not os.path.exists(f'{OUT}/{c}.x.npz')), key=lambda kv: (kv[0].split('_')[1], kv[0]))
    print(len(jobs), 'clips', sum(len(f) for _, f in jobs), 'frames', flush=True)
    tot = sf = fl = bad = 0
    with Pool(a.workers, initializer=init) as p:
        for i, (c, n, F, q, fails, b) in enumerate(p.imap(job, jobs, chunksize=1)):
            tot += n; sf += q > 0; fl += fails; bad += n != F
            if i % 50 == 0: print(f'{i + 1}/{len(jobs)} clips {tot} frames, clips with stop signs {sf}, logged stop fails {fl}, mismatch {bad}, {time.time() - t0:.0f} s', flush=True)
    print(f'CONTEXT_DONE clips {len(jobs)} frames {tot} stop-sign clips {sf} logged stop fails {fl} mismatch {bad} {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
