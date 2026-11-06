"""State of each route for run_lane_multi.sh: 'done' (a scored result the model earned: completed, deviated, blocked, tick cap,
timed out), 'infra' (the run broke: simulation / agent crash, no scores) or 'todo' (no run dir). With --move, every 'infra'
route dir is moved to <root>/_failed_infra/<run id>/ so the next pass re-runs it and results.py never reads it.
    python3 route_state.py <runs dir> <route,route,...> [--move]  -> one 'route state' line each"""
import glob, json, os, shutil, sys
runs, routes, move = sys.argv[1].rstrip('/'), sys.argv[2].split(','), '--move' in sys.argv
INFRA = ('Simulation crashed', 'Agent crashed', "couldn't be set up", 'Agent took longer')
for r in routes:
    ds = sorted(glob.glob(f'{runs}/*/{r}-*/'), key=os.path.getmtime); st = 'todo'
    for d in ds:
        try: rec = json.load(open(d + 'results.json'))['_checkpoint']['records'][0]
        except Exception: rec = None
        ok = rec is not None and 'scores' in rec and not any(k in rec.get('status', '') for k in INFRA)
        if ok: st = 'done'; continue
        st = 'infra' if st != 'done' else st
        if move:
            dst = os.path.join(os.path.dirname(os.path.dirname(runs)), '_failed_infra', os.path.basename(os.path.dirname(d.rstrip('/'))))
            dst = os.path.normpath(dst); os.makedirs(dst, exist_ok=True); shutil.move(d.rstrip('/'), dst)
    print(r, st)
