"""Step B ego inputs for every Phase 2 frame, as the runner sends them (spec: Input row).
  speed (v, 0); acceleration (dv/dt, 0) between the 10 Hz annos, not the labels' accelerometer reading.
  command: the runner's own function (bench2drive/agent.py _command): the RoadOption of the plan point nearest the ego, through its CMD table. The plan is rebuilt from the clip's anno
           (every distinct x_command_near, y_command_near, command_near, about 1-2 m apart). FOLLOW goes in as straight, as wajepa_dj/policy.py:102-103 does.
    python3 build_inputs.py [--workers 4] -> paddock-data/b2d_drivejepa_phase2/inputs/ego_inputs.npz"""
import argparse, csv, gzip, json, os, time
from multiprocessing import Pool
import numpy as np
BASE = '{data.b2d_clips}'; OUT = '{data.b2d_retrain}/inputs/ego_inputs.npz'
# RoadOption -> agent.py CMD -> COMMAND_ONE_HOT, with FOLLOW -> straight
ONE_HOT = {1: (1, 0, 0, 0), 2: (0, 0, 1, 0), 3: (0, 1, 0, 0), 4: (0, 1, 0, 0), 5: (0, 1, 0, 0), 6: (0, 1, 0, 0)}


def job(a):
    clip, frames = a
    d = next(f'{BASE}/{s}/{clip}/anno' for s in ('b2d_base', 'b2d_pilot') if os.path.isdir(f'{BASE}/{s}/{clip}/anno'))
    names = sorted(os.listdir(d)); A = {}
    for f in names:
        A[int(f.split('.')[0])] = json.load(gzip.open(f'{d}/{f}'))
    plan = []
    for n in sorted(A):
        p = (round(A[n]['x_command_near'], 3), round(A[n]['y_command_near'], 3), int(A[n]['command_near']))
        if p not in plan[-3:]: plan.append(p)
    P = np.array(plan, np.float64); rows = []
    for n in frames:
        a1 = A[n]; v = float(a1['speed']); acc = (v - float(A[n - 1]['speed'])) / 0.1 if n - 1 in A else 0.0
        i = int(np.linalg.norm(P[:, :2] - np.array([a1['x'], a1['y']]), axis=1).argmin()); o = int(P[i, 2])
        rows.append((n, v, acc, o, int(a1['command_near'])))
    return clip, rows, len(P)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4); a = ap.parse_args(); t0 = time.time()
    fr = {}
    for r in csv.DictReader(open('{ops}/b2d_drivejepa/vocab/frames_used.csv')): fr.setdefault(r['clip'], []).append(int(r['frame']))
    clip, frame, v, acc, opt, near = [], [], [], [], [], []
    with Pool(a.workers) as p:
        for i, (c, rows, n) in enumerate(p.imap(job, sorted(fr.items()), chunksize=4)):
            for r in rows:
                clip.append(c); frame.append(r[0]); v.append(r[1]); acc.append(r[2]); opt.append(r[3]); near.append(r[4])
            if i % 100 == 0: print(i + 1, 'clips', round(time.time() - t0), 's', flush=True)
    opt = np.array(opt, np.int8); oh = np.array([ONE_HOT.get(int(o), (0, 1, 0, 0)) for o in opt], np.float32)
    assert not os.path.exists(OUT)
    np.savez(OUT, clip=np.array(clip), frame=np.array(frame, np.int32), speed=np.array(v, np.float32), accel=np.array(acc, np.float32), option=opt, option_anno_near=np.array(near, np.int8), one_hot=oh)
    print('INPUTS_DONE', len(clip), dict(zip(*[x.tolist() for x in np.unique(opt, return_counts=True)])), round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
