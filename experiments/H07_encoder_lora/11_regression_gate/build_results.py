"""Read-only: collect every number in RESULTS.md from the finished gate job into results.json.

Reads the job folder, the per-sample records on the NAS and our own held-out dump. Writes only results.json beside
this file. Run with the Paddock venv: python build_results.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JOB = Path("{paddock_site}/queue/jobs/20261027T215457-siddesh-94f0")
RUNS = Path("{paddock_runs}")
CAND = Path("{paddock_site}/queue/checkouts/9585f816c05ffc5631c950fff47ee90b383970c8/results")
OURS_PKL = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s5/heldout223.pkl")
SAMPLE_LIST = Path("{naren_navsim_workspace}/exp/inference/kitscenes_e2e_final.pkl")

RUN_IDS = {
    "kitscenes": ("20261023T195836-e90d98", "20261027T221934-41f714"),
    "nuscenes": ("20261023T200508-261359", "20261027T220152-404b46"),
    "navsim": ("20261023T202240-de2f00", "20261027T215542-59375a"),
    "neuroncap_quick": ("20261026T195633-2a9a76", "20261027T220137-fc8654"),
    "bench2drive_quick": ("20261026T233957-56d38c", "20261027T220735-ccf106"),
    "alpasim_quick": ("20261027T005941-d3f828", "20261027T221606-98bbe7"),
    "neuroncap": ("20261026T232704-7bcf0e", "20261027T222610-aff320"),
    "bench2drive": ("20261026T233957-56d38c", "20261027T222440-49bb8c"),
    "alpasim": ("20261027T005941-d3f828", "20261027T230105-487c55"),
}
CHECKOUTS = {
    "06870801c": "{paddock_site}/queue/checkouts/06870801c05a8e43b81887a469d869ce468935ab/results",
    "1561b66ac": "{paddock_site}/queue/checkouts/1561b66ac3dfa1b718e6aa962045f2f398cf6018/results",
}


def r4(x):
    return None if x is None else (round(float(x), 4) if isinstance(x, (int, float, np.floating)) else x)


def samples(bench: str, run: str) -> dict:
    return {d["sample_id"]: d for d in map(json.loads, open(RUNS / bench / "drivejepa" / run / "samples.jsonl"))}


def part_a() -> dict:
    v = json.load(open(JOB / "verdict.json"))
    subsets = [s["name"] for s in v["steps"]]
    pairs = []
    for name, p in zip(subsets, v["pairs"]):
        row = {k: r4(p[k]) for k in ("bench", "metric", "verdict", "baseline_mean", "candidate_mean", "diff", "se",
                                       "t", "t_crit", "mde", "units", "unit_name", "matched", "only_baseline",
                                       "only_candidate", "can_ship", "crash_check")}
        row["subset"] = name
        row["baseline_run"], row["candidate_run"] = p["baseline"], p["candidate"]
        row["notes"] = p["notes"]
        row["pdm"] = [{k: r4(q[k]) for k in ("slot", "metric", "baseline_mean", "candidate_mean", "diff", "se", "t",
                                              "t_crit", "verdict")} for q in p["pdm"]]
        row["pdm_worse_beyond_t_crit"] = [q["metric"] for q in p["pdm"]
                                          if q["t"] is not None and q["t"] <= -q["t_crit"]]
        pairs.append(row)
    return {"overall": v["overall"], "unfinished": v["unfinished"], "pairs": pairs}


def b2d_routes(run: str) -> list:
    out = []
    root = RUNS / "bench2drive" / "drivejepa" / run
    motion = {}
    for f in glob.glob(str(root / "metric_info" / "*" / "metric_info.json")):
        d = json.load(open(f))
        loc = [d[k]["location"] for k in sorted(d, key=int)]
        sp = [math.dist(a[:2], b[:2]) / 0.05 for a, b in zip(loc, loc[1:])]
        rid = os.path.basename(os.path.dirname(f)).split("_rep0_")[0].replace("RouteScenario_", "")
        motion[rid] = {"distance_m": round(sum(s * 0.05 for s in sp), 1), "top_speed_mps": round(max(sp), 2),
                       "stopped_s": round(sum(s < 0.1 for s in sp) * 0.05, 1), "frames": len(loc)}
    for rj in sorted(glob.glob(str(root / "*" / "results.json"))):
        d = json.load(open(rj))
        rec = d["_checkpoint"]["records"][0]
        name = os.path.basename(os.path.dirname(rj))
        rid = name.split("-")[0]
        sample = next(s for s in samples("bench2drive", run).values() if s["native_id"]["scene"] == rid)
        out.append({"route": name, "status": rec["status"], "entry_status": d["entry_status"],
                    "route_completion": r4(rec["scores"]["score_route"]),
                    "infraction_penalty": r4(rec["scores"]["score_penalty"]),
                    "driving_score": r4(rec["scores"]["score_composed"]),
                    "infractions": {k: len(x) for k, x in rec["infractions"].items() if x},
                    "game_s": rec["meta"]["duration_game"], "wall_s": sample["metrics"]["duration_s"],
                    **motion[rid]})
    return out


def part_b() -> dict:
    out = {}
    for label, run in (("baseline", "20261026T233957-56d38c"), ("candidate", "20261027T222440-49bb8c"),
                       ("candidate_crash_check", "20261027T220735-ccf106")):
        out[label] = {"run": run, "routes": b2d_routes(run)}
    return out


def replan_stats(run: str, clip: str) -> dict | None:
    f = RUNS / "alpasim" / "drivejepa" / run / f"{clip}.replans.jsonl"
    if not f.exists():
        return None
    rp = [r for r in map(json.loads, open(f)) if r["kind"] == "replan"]
    idx = [r["extras"]["drivejepa_pick"]["chosen_index"] for r in rp]
    first = next(r for r in rp if r["phase"] == "after handover")
    t = np.asarray(first["trajectory_rig"])[:, :2]
    return {"replans": len(rp), "warm_up": sum(r["phase"] == "warm-up" for r in rp),
            "momentum_fed": sum(r["extras"]["drivejepa_pick"]["momentum_fed"] for r in rp),
            "pick_index_changes": sum(a != b for a, b in zip(idx, idx[1:])),
            "first_after_warmup_replan": first["replan"], "speed_sent_mps": round(first["speed_sent"], 2),
            "plan_speed_first_0p5s_mps": round(float(np.linalg.norm(t[0]) / 0.5), 2),
            "plan_speed_avg_3s_mps": round(float(np.linalg.norm(t[5]) / 3.0), 2), "command": first["command"]}


def part_c() -> dict:
    b_run, c_run = RUN_IDS["alpasim"]
    b, c = samples("alpasim", b_run), samples("alpasim", c_run)
    keys = ("scene_score", "progress", "collision_at_fault", "collision_rate", "offroad", "dist_to_gt_trajectory",
            "duration_frac_20s")
    scenes = []
    for sid in c:
        clip = c[sid]["native_id"]["scene"]
        bm, cm = b[sid]["metrics"], c[sid]["metrics"]
        asl = glob.glob(str(RUNS / "alpasim" / "drivejepa" / c_run / clip / "rollouts" / "*" / "*" / "rollout.asl"))
        diff = cm["scene_score"] - bm["scene_score"]
        scenes.append({"clip": clip, "baseline": {k: r4(bm[k]) for k in keys}, "candidate": {k: r4(cm[k]) for k in keys},
                       "scene_score_diff": r4(diff),
                       "class": "worse" if diff < -0.05 else "better" if diff > 0.05 else "same",
                       "moved_more_than_0p55": abs(diff) > 0.55,
                       "replans_baseline": replan_stats(b_run, clip), "replans_candidate": replan_stats(c_run, clip),
                       "rollout_asl": asl[0] if asl else None,
                       "replans_jsonl": str(RUNS / "alpasim" / "drivejepa" / c_run / f"{clip}.replans.jsonl")})
    mc = {}
    for label, path in (("candidate", CAND / "alpasim/drivejepa" / c_run / "run.json"),
                        ("baseline", Path(CHECKOUTS["1561b66ac"]) / "alpasim/drivejepa" / b_run / "run.json")):
        mc[label] = json.load(open(path))["provenance"]["policy"].get("momentum_counts")
    return {"baseline_run": b_run, "candidate_run": c_run, "scenes": scenes, "momentum_counts_run_level": mc}


def part_d() -> dict:
    b_run, c_run = RUN_IDS["neuroncap"]
    b, c = samples("neuroncap", b_run), samples("neuroncap", c_run)
    ncap = []
    for sid in b:
        nid = b[sid]["native_id"]
        row = {"scenario": f"{nid['scenario']}-{nid['scene']}"}
        for label, rec in (("baseline", b[sid]), ("candidate", c[sid])):
            m = rec["metrics"]
            row[label] = {"ncap_score": r4(m["ncap_score"]), "collision": bool(m["collision_rate"]),
                          "impact_speed_mps": r4(m["impact_speed"]), "reference_speed_mps": r4(m["reference_speed"])}
        ncap.append(row)
    b_run, c_run = RUN_IDS["navsim"]
    b, c = samples("navsim", b_run), samples("navsim", c_run)
    names = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
             "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
             "history_comfort", "epdms"]
    nav = {}
    for k in names:
        bv = np.array([b[s]["metrics"][k] for s in b], float)
        cv = np.array([c[s]["metrics"][k] for s in b], float)
        nav[k] = {"baseline_mean": r4(bv.mean()), "candidate_mean": r4(cv.mean()),
                  "baseline_below_1": int((bv < 1).sum()), "candidate_below_1": int((cv < 1).sum()),
                  "baseline_zero": int((bv == 0).sum()), "candidate_zero": int((cv == 0).sum())}
    hard = names[:4]
    for label, rec in (("baseline", b), ("candidate", c)):
        nav[f"{label}_any_hard_gate_fail"] = sum(any(rec[s]["metrics"][h] < 1 for h in hard) for s in rec)
        nav[f"{label}_epdms_zero_all_terms_one"] = sum(
            rec[s]["metrics"]["epdms"] == 0 and all(rec[s]["metrics"][h] == 1 for h in hard) for s in rec)
    nav["n"] = len(b)
    return {"neuroncap": ncap, "navsim": nav}


def part_e() -> dict:
    d = pickle.load(open(OURS_PKL, "rb"))
    ours = {(r["scene_id"], int(r["keyframe_index"])): r for r in d["rows"][5]}
    dumps = d["dumps"][5]
    key = lambda r: (r["native_id"]["scene_id"], int(r["native_id"]["keyframe_index"]))
    gate = {key(r): r for r in samples("kitscenes", RUN_IDS["kitscenes"][1]).values()}
    base = {key(r): r for r in samples("kitscenes", RUN_IDS["kitscenes"][0]).values()}
    cmd = {(r["scene_id"], int(r["keyframe_index"])): ["left", "straight", "right", "unknown"][int(np.argmax(r["driving_command"]))]
           for r in pickle.load(open(SAMPLE_LIST, "rb"))}
    assert set(ours) == set(gate) == set(base) == set(cmd)
    gt_diff = max(float(np.abs(np.asarray(gate[k]["ground_truth"])[:6, :2] - ours[k]["human_poses"][:6, :2]).max())
                  for k in ours)

    def group(ks):
        o = np.array([ours[k]["ade"] for k in ks])
        g = np.array([gate[k]["metrics"]["ade"] for k in ks])
        r = np.array([base[k]["metrics"]["ade"] for k in ks])
        return {"n": len(ks), "ours_ade3s": r4(o.mean()), "gate_ade3s": r4(g.mean()), "released_gate_ade3s": r4(r.mean()),
                "gap": r4(g.mean() - o.mean()), "gate_worse_count": int((g > o).sum()),
                "same_pick_index": int(sum(dumps[k]["chosen"] == gate[k]["extras"]["drivejepa_pick"]["chosen_index"] for k in ks)),
                "corr": r4(np.corrcoef(o, g)[0, 1]) if len(ks) > 2 else None}

    by_split = {sp: group([k for k in ours if ours[k]["split"] == sp]) for sp in ("val", "overlap_train_val")}
    by_split["all"] = group(list(ours))
    by_cmd = {c: group([k for k in ours if cmd[k] == c]) for c in ("left", "straight", "right") if any(cmd[k] == c for k in ours)}
    return {"max_ground_truth_diff_m": gt_diff, "by_split": by_split, "by_true_command": by_cmd,
            "gate_momentum_fed": sum(bool(gate[k]["extras"]["drivejepa_pick"]["momentum_fed"]) for k in gate)}


def main() -> None:
    res = {"job": JOB.name, "run_ids": {k: {"baseline": v[0], "candidate": v[1]} for k, v in RUN_IDS.items()},
           "A": part_a(), "B": part_b(), "C": part_c(), "D": part_d(), "E": part_e()}
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("wrote", HERE / "results.json")


if __name__ == "__main__":
    main()
