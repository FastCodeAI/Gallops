"""Extra comparisons Sid asked for (2026-10-16), from saved dumps only -- no model runs.

    ./run.sh src/g_extra.py         # writes out/g_extra.json

Before = zero LoRA, after = the chosen pass.  Selection: both sides are full
dumps (32 proposals, grades, pick).  Part F: "before" is the August pickle,
which stores only the chosen trajectory, so the pick index and best-of-32 are
not available on that side (running zero LoRA on val/overlap is not allowed by
the Stage 2 data rules).

Definitions
- ADE: RUN.displacement_errors, 3 s horizon (RUN:119-136), as everywhere else.
- best-of-32 ADE: the same ADE for each of the 32 proposals, minimum taken
  (the candidate closest to the human).  trajectory == proposals[chosen]
  exactly in every dump checked, so the pick is one of these 32.
- pick distance: mean over the 8 waypoints of the xy distance between the old
  and new chosen trajectory (m).
- plan speed: path length through the 8 waypoints from the ego origin / 4 s;
  end distance: |xy| of waypoint 8 (4 s ahead).
"""

from __future__ import annotations

import json
import pickle

import numpy as np

import common as C
from s2common import OUT, write_json

RUN = C.kit_runner_module()


def ade3(poses, human, dt):
    return RUN.displacement_errors(poses, human, dt, 3.0)["ade"]


def plan_stats(traj):
    xy = np.vstack([[0.0, 0.0], np.asarray(traj, dtype=np.float64)[:, :2]])
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() / 4.0), float(np.linalg.norm(xy[-1]))


def s(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v))} if len(v) else {"n": 0}


def compare(before: dict, after: dict, human: dict, dt: float, before_has_menu: bool) -> dict:
    """before/after: key -> {"trajectory", optional "proposals", "chosen", "pdm_score"}."""
    keys = sorted(after)
    ab = {k: ade3(before[k]["trajectory"], human[k], dt) for k in keys}
    aa = {k: ade3(after[k]["trajectory"], human[k], dt) for k in keys}
    d = np.array([aa[k] - ab[k] for k in keys])
    out = {
        "1_ade": {"before": s(list(ab.values())), "after": s(list(aa.values())),
                  "better_by_more_than_10cm": int((d < -0.10).sum()), "worse_by_more_than_10cm": int((d > 0.10).sum()),
                  "within_10cm": int((np.abs(d) <= 0.10).sum()),
                  "better_by_more_than_1cm": int((d < -0.01).sum()), "worse_by_more_than_1cm": int((d > 0.01).sum())},
    }
    dist = {k: float(np.linalg.norm(np.asarray(before[k]["trajectory"], dtype=np.float64)[:, :2]
                                     - np.asarray(after[k]["trajectory"], dtype=np.float64)[:, :2], axis=1).mean()) for k in keys}
    if before_has_menu:
        changed = [k for k in keys if before[k]["chosen"] != after[k]["chosen"]]
        out["2_pick_distance_m"] = {"moments_pick_changed": len(changed), "changed": s([dist[k] for k in changed]),
                                    "unchanged_pick_same_index": s([dist[k] for k in keys if k not in changed]),
                                    "ade_change_where_pick_changed": s([aa[k] - ab[k] for k in changed]),
                                    "ade_change_where_pick_same": s([aa[k] - ab[k] for k in keys if k not in changed])}
    else:
        out["2_pick_distance_m"] = {"note": "August pickle has no pick index; distance shown for every moment",
                                    "all": s(list(dist.values())),
                                    "moments_moved_more_than_1cm": int(sum(v > 0.01 for v in dist.values())),
                                    "moved_more_than_1cm": s([v for v in dist.values() if v > 0.01])}
    bo_a = {k: min(ade3(p, human[k], dt) for p in after[k]["proposals"]) for k in keys}
    b3 = {"after": s(list(bo_a.values())),
          "after_regret_pick_minus_best": s([aa[k] - bo_a[k] for k in keys])}
    if before_has_menu:
        bo_b = {k: min(ade3(p, human[k], dt) for p in before[k]["proposals"]) for k in keys}
        db = np.array([bo_a[k] - bo_b[k] for k in keys])
        b3.update({"before": s(list(bo_b.values())),
                   "before_regret_pick_minus_best": s([ab[k] - bo_b[k] for k in keys]),
                   "best_of_32_better_by_more_than_10cm": int((db < -0.10).sum()),
                   "best_of_32_worse_by_more_than_10cm": int((db > 0.10).sum())})
    else:
        b3["before"] = "UNAVAILABLE: August pickle stores only the chosen trajectory"
    out["3_best_of_32_ade"] = b3
    sp_b = [plan_stats(before[k]["trajectory"]) for k in keys]
    sp_a = [plan_stats(after[k]["trajectory"]) for k in keys]
    hum = [plan_stats(human[k][:8]) for k in keys if len(human[k]) >= 8]
    out["4_plan"] = {"speed_mps": {"before": s([x[0] for x in sp_b]), "after": s([x[0] for x in sp_a]),
                                   "human": s([x[0] for x in hum])},
                     "end_distance_m": {"before": s([x[1] for x in sp_b]), "after": s([x[1] for x in sp_a]),
                                        "human": s([x[1] for x in hum])},
                     "after_faster_share": float(np.mean([a[0] > b[0] for a, b in zip(sp_a, sp_b)]))}
    # Is the gain "just faster"?  ADE change split by whether the plan sped up, and
    # the error at 3 s (waypoint 6) split into ego-forward x and lateral y.
    faster = [a[0] > b[0] for a, b in zip(sp_a, sp_b)]
    i3 = int(round(3.0 / dt)) - 1
    ex = lambda tr, k: abs(float(np.asarray(tr)[i3, 0] - human[k][i3, 0]))  # noqa: E731
    ey = lambda tr, k: abs(float(np.asarray(tr)[i3, 1] - human[k][i3, 1]))  # noqa: E731
    out["4_plan"].update({
        "ade_change_where_plan_faster": s([aa[k] - ab[k] for k, f in zip(keys, faster) if f]),
        "ade_change_where_plan_not_faster": s([aa[k] - ab[k] for k, f in zip(keys, faster) if not f]),
        "corr_speed_change_vs_ade_change": float(np.corrcoef([a[0] - b[0] for a, b in zip(sp_a, sp_b)], d)[0, 1]),
        "error_at_3s_forward_x_m": {"before": s([ex(before[k]["trajectory"], k) for k in keys]),
                                    "after": s([ex(after[k]["trajectory"], k) for k in keys])},
        "error_at_3s_lateral_y_m": {"before": s([ey(before[k]["trajectory"], k) for k in keys]),
                                    "after": s([ey(after[k]["trajectory"], k) for k in keys])},
    })
    return out


def main() -> None:
    hist = json.loads((OUT / "e_passes.json").read_text())
    best = hist["best_pass"]
    with open(C.BASELINE_PKL, "rb") as fh:
        aug = pickle.load(fh)
    dts = {float(r["interval_length"]) for r in aug}
    assert len(dts) == 1
    dt = dts.pop()
    res = {"chosen_pass": best, "interval_s": dt}

    zero = pickle.load(open(OUT / "c_selection_zero_dump.pkl", "rb"))
    rows = pickle.load(open(OUT / "c_selection_zero_rows.pkl", "rb"))
    human = {(r["scene_id"], r["keyframe_index"]): np.asarray(r["human_poses"]) for r in rows}
    assert all(float(np.abs(zero[k]["trajectory"] - zero[k]["proposals"][zero[k]["chosen"]]).max()) == 0.0 for k in zero)
    after = pickle.load(open(OUT / f"e_dump_pass_{best:02d}.pkl", "rb"))
    res["selection_103"] = compare(zero, after, human, dt, True)

    per_pass, prev = [], zero
    for p in hist["passes"]:
        dmp = pickle.load(open(OUT / f"e_dump_pass_{p['pass']:02d}.pkl", "rb"))
        per_pass.append({
            "pass": p["pass"],
            "mean_abs_grade_change_vs_zero": float(np.mean([np.abs(dmp[k]["pdm_score"] - zero[k]["pdm_score"]).mean() for k in dmp])),
            "mean_abs_grade_change_vs_previous_pass": float(np.mean([np.abs(dmp[k]["pdm_score"] - prev[k]["pdm_score"]).mean() for k in dmp])),
            "best_of_32_ade_mean": float(np.mean([min(ade3(q, human[k], dt) for q in dmp[k]["proposals"]) for k in dmp])),
            "picks_changed_vs_previous": int(sum(dmp[k]["chosen"] != prev[k]["chosen"] for k in dmp)),
        })
        prev = dmp
    res["per_pass"] = per_pass
    res["zero_lora_best_of_32_ade_mean"] = float(np.mean([min(ade3(q, human[k], dt) for q in zero[k]["proposals"]) for k in zero]))

    f_dump = OUT / "f_final_dump.pkl"
    if f_dump.exists():
        fd = pickle.load(open(f_dump, "rb"))
        frows = pickle.load(open(OUT / "f_final_rows.pkl", "rb"))
        split = {(r["scene_id"], r["keyframe_index"]): r["split"] for r in frows}
        a = {(r["scene_id"], int(r["keyframe_index"])): r for r in aug}
        hum = {k: np.asarray(a[k]["human_poses"]) for k in fd}
        before = {k: {"trajectory": np.asarray(a[k]["predicted_poses"])} for k in fd}
        res["final_223"] = {}
        for name, keep in (("val", {"val"}), ("overlap_train_val", {"overlap_train_val"}), ("all", {"val", "overlap_train_val"})):
            ks = [k for k in fd if split[k] in keep]
            res["final_223"][name] = compare({k: before[k] for k in ks}, {k: fd[k] for k in ks}, {k: hum[k] for k in ks}, dt, False)
    write_json(OUT / "g_extra.json", res)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
