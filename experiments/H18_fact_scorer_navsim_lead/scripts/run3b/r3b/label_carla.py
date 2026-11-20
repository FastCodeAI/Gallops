"""run 3B live labelling on LEAD samples: carla_full_v1's own steps (full.py, imported unchanged), on given paths instead of
the pool full.label picks:
    scene(rt, f, shift)                 the CARLA stand-in of the metric cache, the Phase 0a-like sample, frames
    cns.before_score_proposals          the car model (LQR N5 q100 + no-reverse bicycle, MKZ): row 0 = LEAD's logged future,
                                        rows 1.. = the given paths
    facts_for + apply_rules             facts.scene_facts with those states, then the CARLA rules (debris is no hit; off road
                                        from the first fully-inside state)
Only the facts the run 3B scorer reads are returned, rows 1.. (= the given paths).
The route comes from the slim local copy (lead_slim, metas + bboxes only) when it exists, else from the LEAD archive.
Must run in a process that never imported the NAVSIM fact code (carla_facts patches run 3A's car constants).
    check: cd $NAVSIM_DEVKIT_ROOT && PYTHONPATH=. $PY label_carla.py check   (stored pool rows relabelled through this path)"""
import gzip
import os
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, "{runs}/run3a/facts/lead/carla_full_v1")
import full as FU  # noqa: E402  (unchanged)

SLIM = Path("{data.paddock_data}/run3b/lead_slim")
KEYS = ("hit", "offroad", "left_route", "red", "stop", "wrong_way", "metres", "lane_off", "ttc", "cutin_brake")
_ROUTES = OrderedDict()
_DEBRIS = {}


def route(zrel, cache=4):
    if zrel in _ROUTES:
        _ROUTES.move_to_end(zrel)
        return _ROUTES[zrel]
    s = FU.setup()
    p = SLIM / zrel
    rt = s["CF"].Route(str(p if p.exists() else FU.DATA / zrel))
    _ROUTES[zrel] = rt
    _DEBRIS[zrel] = FU.debris_ids(rt)
    while len(_ROUTES) > cache:
        k, _ = _ROUTES.popitem(last=False)
        _DEBRIS.pop(k, None)
    return rt


def context(zrel, f, shifted):
    rt = route(zrel)
    shift = (float(rt.meta[f]["perturbation_translation"]), float(rt.meta[f]["perturbation_rotation"])) if shifted else None
    mc, smp, frames, T, ef = FU.scene(rt, f, shift)
    return dict(mc=mc, smp=smp, frames=frames, deb=_DEBRIS[zrel])


def label(ctx, poses):
    """poses (P, 8, 3) float64, rear axle, ego frame at t -> {fact: (P, 8)}."""
    s = FU.setup()
    CF, F = s["CF"], s["F"]
    mc = ctx["mc"]
    poses = np.asarray(poses, np.float64)
    states = CF.cns.before_score_proposals(mc, poses, mc.ego_state)
    fa = FU.facts_for(mc, poses, ctx["smp"], ctx["frames"], states)
    FU.apply_rules(fa, F.cns.scorer, ctx["deb"])
    out = {k: np.asarray(fa[k])[1:] for k in KEYS}
    out["cutin_brake"] = np.nan_to_num(out["cutin_brake"].astype(np.float32), nan=0.0)
    return out


def check(n_routes=6, rows=32):
    s = FU.setup()
    anch = np.concatenate([np.load("data/8192.npy")[:, 4::5], np.load(FU.ANCH_ADD_FILE)[:, 4::5]]).astype(np.float64)
    root = Path("{data.paddock_data}/run3a-facts/carla_full_v1/plan/routes")
    rng = np.random.default_rng(3)
    files = sorted(os.listdir(root))
    bad, done, secs = 0, 0, []
    for fn in rng.choice(files, n_routes, replace=False):
        d = pickle.load(gzip.open(root / fn))
        for r in [d["records"][i] for i in rng.choice(len(d["records"]), min(2, len(d["records"])), replace=False)]:
            t0 = time.time()
            ctx = context(d["route"], int(r["frame"]), bool(r["shifted"]))
            t1 = time.time()
            got = label(ctx, anch[r["anchor"][2:2 + rows]])
            t2 = time.time()
            ok = []
            for k in KEYS:
                st = np.asarray(r["facts"][k])[2:2 + rows]
                if k == "cutin_brake":
                    st = np.nan_to_num(st.astype(np.float32), nan=0.0)
                ok.append(np.array_equal(got[k].astype(st.dtype), st, equal_nan=True))
            if not all(ok):
                print("DIFF", r["token"], r["shifted"], [k for k, o in zip(KEYS, ok) if not o], flush=True)
            bad += not all(ok)
            done += 1
            secs.append((t1 - t0, t2 - t1))
    sc = np.array(secs)
    print("CARLA_LABEL_CHECK", done, "frames", bad, "differ; context s p50", round(float(np.median(sc[:, 0])), 2),
          "label s p50 (32 paths)", round(float(np.median(sc[:, 1])), 2), flush=True)
    assert bad == 0


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
