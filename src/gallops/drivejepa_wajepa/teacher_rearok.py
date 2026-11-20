"""Stage teacher-rearok: the Run 3A teacher list (paddock-ops/run3a/CLAUDE.md, "Teacher list"; option (a),
user 24 Oct "start building the teacher list with the rear ok asap"): for every sample in
anchors_scores_index_v2, simulate its listed teacher anchors as the training scorer does
(before_score_proposals, ego areas), run rear_ok (with the lane-chain rule) and keep the anchors that pass,
in the stored order. v2 stores only anchor indices (no scores), so stored score x rear_ok = this filtered
list. Writes <OUT>/<log>/unknown/<token>/metric_cache.pkl.npy (int64, same layout as v2) via a temp file +
rename; never overwrites (existing outputs are skipped, so the job resumes). Per-sample stats to
<stats>/w<i>.jsonl. The 24 Oct list (lane gate on): 103,288 samples, 0 errors, 17,257,415 -> 17,201,172
anchors (0.33 % dropped).

Ported from paddock-ops run3a/teacher/build_teacher_rearok.py; changed: this docstring, and the v2 list,
metric cache, output and stats directories are arguments (as run they were constants naming that machine's
paths).

    python -m gallops.drivejepa_wajepa.teacher_rearok <workers> --v2 DIR --metric-cache DIR --out DIR --stats
        DIR
        [--limit N]
(Drive-JEPA's interpreter, cwd = the Run 3A tree's navsim_v2, PYTHONPATH = the same.)"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

# Set by main() before the fork-context pool starts, so the workers inherit them.
V2 = MC = None


def work_list(limit=None):
    items = []
    for lg in sorted(os.listdir(V2)):
        d = f"{V2}/{lg}/unknown"
        if not os.path.isdir(d):
            continue
        for tok in sorted(os.listdir(d)):
            items.append((lg, tok))
    if limit:
        items = items[:: max(len(items) // limit, 1)][:limit]
    return items


def run(args):
    wid, items, out, stats = args
    from navsim.agents.drive_jepa_run3a.score_module import compute_navsim_score as cns
    from navsim.agents.drive_jepa_run3a.score_module.run3a_labels import rear_ok

    anchors = np.load("./data/8192.npy")[:, 4::5]
    sc = cns.scorer
    done = 0
    with open(f"{stats}/w{wid}.jsonl", "a") as log:
        for lg, tok in items:
            dst = f"{out}/{lg}/unknown/{tok}/metric_cache.pkl.npy"
            if os.path.exists(dst):
                continue
            t0 = time.time()
            rec = {"log": lg, "token": tok}
            try:
                idx = np.load(f"{V2}/{lg}/unknown/{tok}/metric_cache.pkl.npy")
                rec["n_in"] = int(len(idx))
                keep = idx
                if len(idx):
                    mc = cns.load_metric_cache(f"{MC}/{lg}/unknown/{tok}/metric_cache.pkl")
                    states = cns.before_score_proposals(mc, anchors[idx], mc.ego_state)
                    sc._reset(
                        states,
                        mc.observation,
                        mc.centerline,
                        mc.route_lane_ids,
                        mc.drivable_area_map,
                        mc.past_human_trajectory,
                    )
                    sc._calculate_ego_area()
                    ok, det = rear_ok(
                        states,
                        sc._ego_areas,
                        mc.observation,
                        mc.drivable_area_map,
                        mc.map_parameters.map_name,
                        return_detail=True,
                    )
                    ok = ok[1:]  # row 0 is PDM-Closed
                    keep = idx[ok]
                    rec.update(
                        lane_change=int(det["lane_change"][1:].sum()),
                        pairs=int(det["pairs"]),
                        lane0=det.get("lane0_vehicles"),
                        failed=[int(i) for i in idx[~ok]],
                    )
                rec["n_out"] = int(len(keep))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                tmp = f"{dst}.tmp{os.getpid()}"
                with open(tmp, "wb") as f:
                    np.save(f, keep.astype(np.int64))
                if os.path.exists(dst):  # never overwrite
                    os.remove(tmp)
                else:
                    os.rename(tmp, dst)
            except Exception as e:  # noqa: BLE001 - recorded, sample left without output for a rerun
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["s"] = round(time.time() - t0, 3)
            log.write(json.dumps(rec) + "\n")
            done += 1
            if done % 200 == 0:
                log.flush()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workers", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--v2", required=True, help="anchors_scores_index_v2 (Drive-JEPA's teacher list)")
    ap.add_argument("--metric-cache", required=True, help="train_metric_cache_v2_pruned")
    ap.add_argument("--out", required=True, help="the new list (Run 3A: anchors_scores_index_v2_rearok_lane)")
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()
    global V2, MC
    V2, MC = a.v2, a.metric_cache
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.stats, exist_ok=True)
    items = work_list(a.limit)
    print(f"{len(items)} samples, {a.workers} workers -> {a.out}", flush=True)
    t = time.time()
    chunks = [(i, items[i :: a.workers], a.out, a.stats) for i in range(a.workers)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        n = sum(pool.map(run, chunks))
    print(f"TEACHER_DONE {n} processed in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
