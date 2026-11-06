"""v6 own-lane-blocked flag for every Phase 2 frame, by the existing job (b2d-labels/v6/blocked_v6.clip_job), unchanged. The earlier table only covers MTD's pool frames.
    <b2d python> blocked_all.py [--workers 3] -> paddock-data/b2d_drivejepa_phase2/inputs/blocked_all.json {clip/nnnnn: {v5, v6, ...}}"""
import argparse, csv, json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, '{ops}/b2d-labels/v6'); sys.path.insert(0, '{ops}/b2d-labels/v5')
import blocked_v6 as B6
OUT = '{data.b2d_retrain}/inputs/blocked_all.json'
ALIAS = {'VehicleTurningRoutePedestrian_Town15_Route523_Weathe': 'VehicleTurningRoutePedestrian_Town15_Route523_Weather2'}   # the local folder name is cut short


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=3); a = ap.parse_args(); t0 = time.time()
    fr = {}
    for r in csv.DictReader(open('{ops}/b2d_drivejepa/vocab/frames_used.csv')):
        fr.setdefault(ALIAS.get(r['clip'], r['clip']), []).append(int(r['frame']))
    jobs = sorted(((c, sorted(f)) for c, f in fr.items()), key=lambda kv: (kv[0].split('_')[1], kv[0]))
    print(len(jobs), 'clips', sum(len(f) for _, f in jobs), 'frames', flush=True)
    res = {}
    with Pool(a.workers) as p:
        for i, o in enumerate(p.imap(B6.clip_job, jobs, chunksize=1)):
            res.update(o)
            if i % 50 == 0: print(f'{i + 1}/{len(jobs)} {time.time() - t0:.0f} s', flush=True)
    assert not os.path.exists(OUT)
    json.dump(res, open(OUT, 'w'))
    print('BLOCKED_DONE', len(res), 'v6 true', sum(1 for r in res.values() if r.get('v6')), 'none', sum(1 for r in res.values() if r.get('v6') is None), f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
