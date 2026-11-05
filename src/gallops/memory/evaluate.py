"""H15's go/no-go numbers on the val logs, for arms (a) carried, (b) wiped and (c) released.

All offline, momentum off (Sid D10). Every arm picks the argmax of its own score at every val moment;
(c) is the released final logit. For (a) and (b), each of the 3 seeds picks on its own and a moment's
value is the mean over the seeds; the bootstrap is of that mean (Sid D15), and each seed's value is
shown too. Intervals: log-clustered bootstrap over the 214 val logs, 2,000 draws, seed 0, 95%; every
metric and every arm uses the same draws, so differences are paired.

Metrics (PREDICTIONS.md):
- P1  mean strict grade of the pick, graded val moments (n = 15,320); also the forgiving grade (lane
      keeping and traffic light the benchmark's way, the final recomputed), reported, not tested;
- pick is a best path (ties count);
- P2a flip rate: back-to-back val pairs (n = 18,798) whose picks are more than 1 m apart (mean distance
      over the 3.5 s they share, ego-motion aligned: `align.distances`);
- P2b two-frame extended comfort: pairs where both moments are graded (n = 12,768), the vendor's
      `ego_is_two_frame_extended_comfort` on the two picks' simulated states with the overlap rule of
      `run_pdm_score_sequential.calculate_two_frame_extended_comfort` (offset int(interval / 0.1));
- R1  on graded moments where a best path brakes and the released pick neither brakes nor is a best
      path: the share of picks that brake (no prediction attached).

    python -m gallops.memory.evaluate        # reads {data.navtrain.cache}/h15/final.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gallops.memory import align, probe

ACCEL_X, BRAKE, FIRST_2S = 5, -1.5, 21  # StateIndex.ACCELERATION_X; first 2.0 s at 0.1 s
FLIP_M = 1.0
REPS, SEED = 2000, 0
#: the final-grade weights of train_pdm_scorer.PDMScorerConfig (progress, ttc, lane keeping, history comfort)
WEIGHTS = {
    "ego_progress": 5.0,
    "time_to_collision_within_bound": 5.0,
    "lane_keeping": 2.0,
    "history_comfort": 2.0,
}
GATES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "traffic_light_compliance",
    "driving_direction_compliance",
)


def final_from(sub: np.ndarray, names) -> np.ndarray:
    """The vendor's final grade from its eight sub-scores: product of the gates x weighted mean."""
    idx = {n: k for k, n in enumerate(names)}
    gates = np.prod([sub[..., idx[g]] for g in GATES], axis=0)
    weighted = sum(w * sub[..., idx[n]] for n, w in WEIGHTS.items()) / sum(WEIGHTS.values())
    return gates * weighted


def boot(values: np.ndarray, groups: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Point mean and its per-draw means (log-clustered)."""
    n_groups = weights.shape[1]
    s = probe.per_log_sums(groups, values, n_groups)
    c = probe.per_log_sums(groups, np.ones_like(values), n_groups)
    return float(values.mean()), (weights @ s) / (weights @ c)


def summarise(point: float, draws: np.ndarray) -> dict:
    lo, hi = probe.ci(draws)
    return {"value": point, "ci95": [lo, hi], "ci_excludes_0": bool(lo > 0 or hi < 0)}


def run(cfg, h15: Path) -> dict:
    import torch

    from gallops.memory import grade, train
    from gallops.memory.model import MemoryPicker

    t0 = time.time()
    final = json.loads((h15 / "final.json").read_text())
    length = int(final["length"])
    device = train.pick_device()
    data = train.Data(cfg, device)
    table, prev = data.table, data.prev
    cache = Path(cfg.resolve("{data.navtrain.cache}"))
    val_rows = np.flatnonzero(table["split"] == "val")
    graded_val = data.rows["val"]

    # grades of every path at the graded val moments
    gdir = cache / "grades"
    row, done = np.load(gdir / "row.npy"), np.load(gdir / "done.npy")
    gpos = np.full(len(table["log"]), -1, np.int64)
    gpos[row[done]] = np.flatnonzero(done)
    gi = gpos[graded_val]
    assert (gi >= 0).all()
    scores = np.asarray(np.load(gdir / "scores.npy", mmap_mode="r")[gi], np.float64)
    forgiving_two = np.asarray(np.load(gdir / "forgiving.npy", mmap_mode="r")[gi], np.float64)
    names = list(grade.SCORE_NAMES)
    recomputed = final_from(scores[..., :8], names[:8])
    formula_error = float(np.abs(recomputed - scores[..., -1]).max())
    sub_f = scores[..., :8].copy()
    sub_f[..., names.index("lane_keeping")] = forgiving_two[..., 0]
    sub_f[..., names.index("traffic_light_compliance")] = forgiving_two[..., 1]
    strict = scores[..., -1]
    forgiving = final_from(sub_f, names[:8])
    accel = np.asarray(np.load(gdir / "states.npy", mmap_mode="r")[gi][:, :, :FIRST_2S, ACCEL_X], np.float64)
    brakes = accel.min(-1) < BRAKE
    best = strict == strict.max(1, keepdims=True)

    # picks of every arm and seed at every val moment
    released = data.logits_np[val_rows][:, :, -1]
    picks = {"c": [released.argmax(1)]}
    per_run = []
    for r in final["runs"]:
        mode = r["mode"]
        state = torch.load(Path(r["checkpoint"]) / "memory.pt", map_location=device)
        model = MemoryPicker().to(device)
        model.load_state_dict(state)
        s = train.scores_of(model, data, val_rows, mode, length)
        arm = "a" if mode == "carry" else "b"
        picks.setdefault(arm, []).append(s.argmax(1))
        per_run.append({"run": r["run"], "arm": arm, "seed": r["seed"]})
    where = {int(v): k for k, v in enumerate(val_rows)}
    at_graded = np.array([where[int(v)] for v in graded_val])

    # back-to-back val pairs; pick-to-pick distance and comfort per arm/seed
    pairs = np.array([r for r in val_rows if prev[r] >= 0 and int(prev[r]) in where])
    pk_now = np.array([where[int(r)] for r in pairs])
    pk_prev = np.array([where[int(prev[r])] for r in pairs])
    poses = np.load(cache / "align" / "poses.npy")
    proposals = np.load(cache / "release" / "proposals.npy", mmap_mode="r")
    aligned_prev = {}

    def pick_distance(k, old_pick, new_pick):
        r, p = int(pairs[k]), int(prev[pairs[k]])
        if k not in aligned_prev:
            aligned_prev[k] = align.align_previous(proposals[p], poses[p], poses[r])
        new = np.asarray(proposals[r], float)[new_pick, :7, :2]
        return float(np.linalg.norm(aligned_prev[k][old_pick] - new, axis=-1).mean())

    both = np.array([gpos[int(prev[r])] >= 0 and gpos[int(r)] >= 0 for r in pairs])
    states_mm = np.load(gdir / "states.npy", mmap_mode="r")
    origin = np.load(gdir / "origin.npy")
    grade.init_worker(cfg)
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )

    comfort_cache: dict = {}

    def comfort(k, old_pick, new_pick):
        key = (k, int(old_pick), int(new_pick))
        if key not in comfort_cache:
            r, p = int(pairs[k]), int(prev[pairs[k]])
            a = np.asarray(states_mm[gpos[p], old_pick], np.float64)
            b = np.asarray(states_mm[gpos[r], new_pick], np.float64)
            a[:, :2] += origin[gpos[p]]
            b[:, :2] += origin[gpos[r]]
            interval = (int(table["timestamp"][r]) - int(table["timestamp"][p])) / 1e6
            off = int(interval / 0.1)  # the vendor's rule, truncation included
            n = a.shape[0] - off
            comfort_cache[key] = float(
                ego_is_two_frame_extended_comfort(a[off:][None], b[:-off][None], np.arange(n) * 0.1)[0]
            )
        return comfort_cache[key]

    log_names = np.unique(table["log"][val_rows])
    log_index = {n: k for k, n in enumerate(log_names)}
    weights = probe.log_weights(len(log_names), REPS, SEED)
    g_graded = np.array([log_index[n] for n in table["log"][graded_val]])
    g_pairs = np.array([log_index[n] for n in table["log"][pairs]])
    g_both = g_pairs[both]

    released_pick = picks["c"][0][at_graded]
    r1 = (
        (best & brakes).any(1)
        & ~brakes[np.arange(len(gi)), released_pick]
        & ~best[np.arange(len(gi)), released_pick]
    )
    g_r1 = g_graded[r1]

    metric_values: dict = {}
    per_seed: dict = {}
    for arm, plist in picks.items():
        vals = {
            "P1_strict": [],
            "P1_forgiving": [],
            "pick_is_best": [],
            "P2a_flip": [],
            "P2b_comfort": [],
            "R1_brakes": [],
        }
        for pick in plist:
            p_g = pick[at_graded]
            ix = np.arange(len(gi))
            vals["P1_strict"].append(strict[ix, p_g])
            vals["P1_forgiving"].append(forgiving[ix, p_g])
            vals["pick_is_best"].append(best[ix, p_g].astype(float))
            vals["R1_brakes"].append(brakes[ix, p_g][r1].astype(float))
            old, new = pick[pk_prev], pick[pk_now]
            vals["P2a_flip"].append(
                np.array([pick_distance(k, old[k], new[k]) > FLIP_M for k in range(len(pairs))], float)
            )
            vals["P2b_comfort"].append(np.array([comfort(k, old[k], new[k]) for k in np.flatnonzero(both)]))
        metric_values[arm] = {m: np.mean(v, 0) for m, v in vals.items()}
        per_seed[arm] = {m: [float(np.mean(x)) for x in v] for m, v in vals.items()}

    groups = {
        "P1_strict": g_graded,
        "P1_forgiving": g_graded,
        "pick_is_best": g_graded,
        "P2a_flip": g_pairs,
        "P2b_comfort": g_both,
        "R1_brakes": g_r1,
    }
    results: dict = {"arms": {}, "differences": {}}
    draws: dict = {}
    for arm, mv in metric_values.items():
        results["arms"][arm] = {}
        for m, v in mv.items():
            point, d = boot(v, groups[m], weights)
            draws[(arm, m)] = d
            results["arms"][arm][m] = {**summarise(point, d), "n": int(len(v)), "per_seed": per_seed[arm][m]}
    for a, b in (("a", "b"), ("a", "c"), ("b", "c")):
        results["differences"][f"{a}-{b}"] = {}
        for m in metric_values[a]:
            pa, pb = results["arms"][a][m]["value"], results["arms"][b][m]["value"]
            results["differences"][f"{a}-{b}"][m] = {
                **summarise(pa - pb, draws[(a, m)] - draws[(b, m)]),
                "n": results["arms"][a][m]["n"],
            }
    results["verdict"] = verdict(results)
    results["counts"] = {
        "val_logs": int(len(log_names)),
        "graded_val_moments": int(len(gi)),
        "pairs": int(len(pairs)),
        "both_graded_pairs": int(both.sum()),
        "r1_moments": int(r1.sum()),
    }
    results["checks"] = {"final_formula_max_error": formula_error, "device": device, "length": length}
    results["runs"] = per_run
    results["seconds"] = round(time.time() - t0, 1)
    return results


def verdict(results: dict) -> dict:
    """PREDICTIONS.md's rules, applied as written."""
    d = results["differences"]

    def beats(x):
        return x["value"] > 0 and x["ci_excludes_0"]

    def about(x):
        return not x["ci_excludes_0"]

    def worse(x):
        return x["value"] < 0 and x["ci_excludes_0"]

    p1 = d["a-b"]["P1_strict"]
    if beats(p1):
        p1_v = "supported" if p1["value"] >= 0.005 else "supported but not useful (below +0.005)"
    else:
        p1_v = "refuted (the CI includes 0)" if about(p1) else "refuted ((a) - (b) < 0)"
    flips, comf = d["a-b"]["P2a_flip"], d["a-b"]["P2b_comfort"]
    p2a = "(a) better than (b)" if flips["value"] < 0 else "(a) not better than (b)"
    p2b = "(a) better than (b)" if comf["value"] > 0 else "(a) not better than (b)"
    ab, bc, ac = d["a-b"]["P1_strict"], d["b-c"]["P1_strict"], d["a-c"]["P1_strict"]
    rules = []
    if beats(ab) and about(bc):
        rules.append("(a) beats (b), and (b) ~ (c): memory helps; go on to the gate")
    if about(ab) and beats(bc) and beats(ac):
        rules.append("(a) ~ (b), and both beat (c): capacity or training; rethink")
    if about(ab) and about(bc) and about(ac):
        rules.append("all equal: stop, and write it up")
    if worse(ac) or worse(bc):
        rules.append("an arm is worse than (c): stop, or fix")
    if not beats(ab):
        rules.append("P1 fails: not supported, whatever P2 shows")
    elif not (flips["value"] < 0 and comf["value"] > 0):
        rules.append("P1 passes and P2 fails: supported on grade; comfort not shown")
    if not rules:
        rules.append("no pre-registered rule matches exactly; see the differences")
    return {
        "P1": p1_v,
        "P2a": p2a,
        "P2b": p2b,
        "P3": "not run (gate)",
        "P4": "not run (gate)",
        "rules": rules,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h15", type=Path, help="default {data.navtrain.cache}/h15")
    args = ap.parse_args()

    import torch

    from gallops.config import SiteConfig

    torch.set_num_threads(2)
    cfg = SiteConfig.load()
    h15 = (args.h15 or Path(cfg.resolve("{data.navtrain.cache}")) / "h15").resolve()
    results = run(cfg, h15)
    (h15 / "evaluate.json").write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({"differences": results["differences"]["a-b"], "verdict": results["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
