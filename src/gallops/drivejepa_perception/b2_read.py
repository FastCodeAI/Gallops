"""B-2's reads: R-clean's health and noise floor, navtest read once per phase, and the decision chain.

    python -m gallops.drivejepa_perception.b2_read rclean --runs-root {out}/part-b/b2/runs --out FILE
    python -m gallops.drivejepa_perception.b2_read navtest --phase 1 --runs-root {out}/part-b/b2/runs \
        --out {out}/part-b/b2/navtest-phase1
    python -m gallops.drivejepa_perception.b2_read nusc --runs-root {out}/part-b/b2/runs --out FILE
    python -m gallops.drivejepa_perception.b2_read swaps --runs-root {out}/part-b/b2/runs --out FILE
    python -m gallops.drivejepa_perception.b2_read analyse --runs-root {out}/part-b/b2/runs --out FILE

PREDICTIONS.md, Part B · B-2 (committed before any run). A run is named `b2-<v>-<arm>-s<seed>` (v: 1 joint, 2
two stages with LoRA, 3 two stages full; a two-stage run's name is its stage 3's) and lives in
`{runs-root}/<name>/`, with its dumps `dump-navtest`, `dump-fresh` and `dump-select` beside its weights.

**R-clean** is read as B-1 read it: each pick's vendored final under the human filter (`veto.filtered_final`),
on the 2,624 known-command windows (D25), reweighted to navtest's cities. C's seed range there is the noise
floor: over 0.3 points, C, D and HD under B-2-1 run seeds 3 and 4 too.

**Navtest** is read once per phase. Each run's picks are replayed through the official scorer (B-0-2's
pipeline, as B-1 read them) and paired per scene with R and with the arm the chain names, SE clustered by log.
The chain (K0 to K5) is judged by PREDICTIONS.md's rules, written in `verdicts`.

**nuScenes** is read as the first pass read its refiners there (`ood.read_b2`): each run's pick against the
human at 1, 2 and 3 s, its collisions with the labelled boxes within 3 s, and the best of 32 by ADE over 3 s,
each minus the release's on the same keyframes, SE clustered by scene. Reported, never claimed.

**The analysis** is B-0-4 (`b_analysis`) on every run's navtest dump against R: the 32's spread, minADE and
misses, the pick's steadiness between frames (an EC proxy), best of 32 and the share passing NC and TTC; and
from R, the Hungarian-matched moves (by R's score rank) and the pick's switches.

**The swaps** (B-2's Q7) read each run again on navtest with its input replaced at test (`train_b2 dump
--swap`): none, another log's, another fold's side-car, labelled agents (D and HD). Vendored, against the
run's own dump: how often the pick changes; how often the picked plan's 4 s point moves more than MOVED_M
(added on 13 Nov, after the smoke run showed the release's scorer near-tied: its top two scores within 0.01
on 88.5% of navtest, so an index can change while the plan barely does); and the pick's final, swapped minus
own.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PREDICTIONS_SECTION = "## Part B · B-2, the side-car's hidden layer into the refiner"
NOISE_LIMIT = 0.3
#: The test-time swaps each arm is read with (`train_b2 dump --swap`), and the move that counts as a new plan.
SWAPS = {
    "O": ("none", "log"),
    "D": ("none", "log", "fold", "truth"),
    "HD": ("none", "log", "fold", "truth"),
    "H6": ("none", "log", "fold"),
    "H3": ("none", "log", "fold"),
}
MOVED_M = 1.0
#: Each phase's runs: (sub-version, arm, seeds). Phase 2's H3 is a seed-0 pilot (D29, decision (c)).
PHASES = {
    "1": [("1", "C", (0, 1, 2)), ("1", "O", (0, 1, 2))],
    "2": [
        ("1", "D", (0, 1, 2)),
        ("1", "HD", (0, 1, 2)),
        ("1", "H6", (0, 1, 2)),
        ("1", "H3", (0,)),
        ("2", "C", (0, 1, 2)),
        ("2", "D", (0, 1, 2)),
        ("2", "HD", (0, 1, 2)),
        ("3", "C", (0, 1, 2)),
        ("3", "D", (0, 1, 2)),
        ("3", "HD", (0, 1, 2)),
    ],
}


def name(version: str, arm: str, seed: int) -> str:
    return f"b2-{version}-{arm}-s{seed}"


def committed(rel: str, needle: str) -> list:
    """Problems unless `rel` holds `needle` and is committed as it stands."""
    text = (REPO / rel).read_text()
    problems = [] if needle in text else [f"{rel} has no section {needle!r}"]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", rel], cwd=REPO, capture_output=True, text=True
    )
    if status.stdout.strip():
        problems.append(f"{rel} has uncommitted changes")
    return problems


# --------------------------------------------------------------------------------------------- R-clean


def rclean_context(out_root: Path) -> dict:
    """R-clean's known-command rows, their logs and city weights, and the human's vendored labels (B-1's)."""
    from gallops.drivejepa_perception import veto

    data = veto.load_split(out_root, "fresh")
    navtest = np.load(Path(out_root) / veto.O_DIRS["navtest"][1] / "labels.npz")
    nmanifest = json.loads((Path(out_root) / veto.O_DIRS["navtest"][1] / "manifest.json").read_text())
    weights, record = veto.city_weights(data["city"], np.array(nmanifest["cities"])[navtest["log_id"]])
    release = veto.filtered_final(data["pdm"][np.arange(len(data["rows"])), data["pick"]], data["human"])
    value, se = veto.weighted_mean_se(release, data["log"], weights)
    return {
        "rows": data["rows"],
        "log": data["log"],
        "human": data["human"],
        "weights": weights,
        "cities": record,
        "release": {"value": value * 100, "se": se * 100},
    }


def rclean_value(ctx: dict, dump_dir: Path) -> dict:
    """A run's pick on R-clean's known-command windows, by B-1's value, against the release's."""
    from gallops.drivejepa_perception import veto

    dump = np.load(Path(dump_dir) / "dump.npz")
    labels = np.load(Path(dump_dir) / "labels.npz")["candidates"]
    at = {int(r): i for i, r in enumerate(dump["rows"])}
    idx = np.array([at[int(r)] for r in ctx["rows"]])
    pick = dump["pick"][idx]
    chosen = labels[idx, pick]
    values = veto.filtered_final(chosen, ctx["human"])
    value, se = veto.weighted_mean_se(values, ctx["log"], ctx["weights"])
    return {"value": value * 100, "se": se * 100, "minus_release": value * 100 - ctx["release"]["value"]}


def rclean(cfg, args) -> dict:
    """Every finished run's R-clean value, and C's noise floor under B-2-1 (the rule above)."""
    out_root = Path(cfg.resolve("{runs}/drivejepa_perception"))
    ctx = rclean_context(out_root)
    root = Path(args.runs_root)
    values = {}
    for run in sorted(root.glob("b2-*")):
        dump = run / "dump-fresh"
        if (dump / "labels.npz").exists():
            values[run.name] = rclean_value(ctx, dump)
    c = [values[name("1", "C", s)]["value"] for s in (0, 1, 2) if name("1", "C", s) in values]
    record = {
        "release": ctx["release"],
        "cities": ctx["cities"],
        "rows": int(len(ctx["rows"])),
        "runs": values,
        "noise_floor": None
        if len(c) < 3
        else {
            "c_seed_range": float(max(c) - min(c)),
            "limit": NOISE_LIMIT,
            "more_seeds": max(c) - min(c) > NOISE_LIMIT,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("release", "noise_floor")}, indent=1))
    return record


# --------------------------------------------------------------------------------------------- navtest


def readouts(dump_dir: Path, release_pick: np.ndarray) -> dict:
    """Vendored readouts of a run's navtest 32: the pick's final, best of 32, the share passing NC and TTC,
    the lateral spread at 4 s, and how often the pick differs from the release's."""
    dump = np.load(Path(dump_dir) / "dump.npz")
    cand = np.load(Path(dump_dir) / "labels.npz")["candidates"]
    n = np.arange(len(dump["pick"]))
    safe = (cand[..., 0] == 1) & (cand[..., 5] == 1)
    lateral = dump["proposals"][:, :, -1, 1]
    return {
        "pick_final": float(cand[n, dump["pick"], -1].mean() * 100),
        "best_of_32": float(cand[..., -1].max(1).mean() * 100),
        "share_safe": float(safe.mean() * 100),
        "lateral_spread_4s_m": float((lateral.max(1) - lateral.min(1)).mean()),
        "switch_from_release": float((dump["pick"] != release_pick[dump["rows"]]).mean()),
    }


def judge(pairs: dict, ref_spread: float) -> dict:
    """H15's rule on per-seed paired differences: supports if every seed is beyond 2 SE and beyond the
    reference arm's seed range; refutes if any seed is below 0 beyond 2 SE."""
    d = [(p["epdms_hf"]["a_minus_b"], p["epdms_hf"]["se"]) for p in pairs.values()]
    return {
        "per_seed": {
            k: {"diff": p["epdms_hf"]["a_minus_b"], "se": p["epdms_hf"]["se"]} for k, p in pairs.items()
        },
        "reference_seed_range": ref_spread,
        "supports": all(x > 2 * se and x > ref_spread for x, se in d),
        "refutes": any(x < -2 * se for x, se in d),
    }


def verdicts(phase: str, pairs: dict, vs_r: dict) -> dict:
    """PREDICTIONS.md's chain, on what this phase read."""

    def spread(version: str, arm: str) -> float:
        # Every seed read: 0-2, or 0-4 when the noise floor fired (PREDICTIONS.md: "H15's rule reads five").
        xs = [vs_r[name(version, arm, s)]["epdms_hf"]["a"] for s in range(5) if name(version, arm, s) in vs_r]
        return float(max(xs) - min(xs)) if len(xs) > 1 else 0.0

    out = {}
    if phase == "1":
        k0 = {k: v for k, v in pairs.items() if k.startswith("O-C")}
        diffs = [v["epdms_hf"]["a_minus_b"] for v in k0.values()]
        out["K0"] = {
            **judge(k0, spread("1", "C")),
            "gate_stops_b2": bool(diffs) and all(x <= 0 for x in diffs),
        }
        return out
    chain = [
        ("K1", "HD-D", spread("1", "D")),
        ("K2", "HD-C", spread("1", "C")),
        ("K4", "H6-HD", spread("1", "HD")),
    ]
    holds = True
    for key, prefix, ref in chain:
        got = {k: v for k, v in pairs.items() if k.startswith(prefix + " ")}
        if not got:
            continue
        verdict = judge(got, ref)
        verdict["read_as_claim"] = holds
        out[key] = verdict
        if key == "K2":
            k3 = {k: v for k, v in vs_r.items() if k.startswith(name("1", "HD", 0)[:-1])}
            lows = [v["epdms_hf"]["a_minus_b"] - 1.96 * v["epdms_hf"]["se"] for v in k3.values()]
            highs = [v["epdms_hf"]["a_minus_b"] + 1.96 * v["epdms_hf"]["se"] for v in k3.values()]
            out["K3"] = {
                "per_seed": {
                    k: {"diff": v["epdms_hf"]["a_minus_b"], "se": v["epdms_hf"]["se"]} for k, v in k3.items()
                },
                "meets": bool(lows) and all(x >= -0.4 for x in lows),
                "beats": bool(lows) and all(x > 0 for x in lows),
                "refutes": any(x < 0 for x in highs),
                "read_as_claim": holds and verdict["supports"],
            }
        holds = holds and verdict["supports"]
    k5 = {k: v for k, v in pairs.items() if k.startswith("H3-H6 ")}
    if k5:
        out["K5"] = {**judge(k5, 0.0), "note": "a seed-0 pilot: reported"}
    return out


def navtest_read(cfg, args) -> dict:
    """The phase's runs on navtest, once (module docstring)."""
    from gallops.drivejepa_perception import navtest

    out_root = Path(cfg.resolve("{runs}/drivejepa_perception"))
    root = Path(args.runs_root)
    out = Path(args.out)
    problems = committed("experiments/H15_perception_sidecar/PREDICTIONS.md", PREDICTIONS_SECTION)
    anchor = json.loads((out_root / "part-b/b0/anchor-d26/anchor.json").read_text())
    if not all(anchor["gates"].values()):
        problems.append(f"B-0's anchor does not hold: {anchor['gates']}")
    if (out / "navtest.json").exists():
        problems.append(f"{out} holds a finished read: navtest is read once per phase")
    runs = [name(v, arm, s) for v, arm, seeds in PHASES[args.phase] for s in seeds]
    if args.phase == "2" and args.extra_seeds:
        runs += [name("1", arm, s) for arm in ("C", "D", "HD") for s in (3, 4)]
    missing = [r for r in runs if not (root / r / "dump-navtest" / "labels.npz").exists()]
    if missing:
        problems.append(f"runs without a labelled navtest dump: {missing}")
    if problems:
        raise SystemExit("B-2 may not read navtest:\n- " + "\n- ".join(problems))
    out.mkdir(parents=True, exist_ok=True)  # resumes run by run: a scored run is never scored again
    labels_dir = out_root / "labels-navtest-v1"
    labels = np.load(labels_dir / "labels.npz")
    tokens = [str(t) for t in labels["token"]]
    release_pick = np.load(out_root / "tokens-navtest-v1" / "pick.npy").astype(np.int64)
    verified = out_root / "part-b/b0/navtest-cache-verify"
    r_dir = out_root / "part-b/b0/navtest-R"
    scored, vs_r, read = {}, {}, {}
    previous = Path(args.previous) if args.previous else None
    for run in runs:
        dump = np.load(root / run / "dump-navtest" / "dump.npz")
        if not np.array_equal(dump["rows"], np.arange(len(tokens))):
            raise SystemExit(f"{run}'s navtest dump does not cover navtest in order")
        plans = out / f"plans-{run}"
        if not plans.exists():
            meta = {
                "run": run,
                "weights_sha256": json.loads((root / run / "run.json").read_text())["weights_sha256"],
            }
            navtest.write_plans(
                plans, tokens, dump["proposals"][dump["rows"], dump["pick"]], meta, pick=dump["pick"]
            )
        scored[run] = out / f"score-{run}"
        if not scored[run].exists():
            navtest.score_plans(cfg, plans, verified, scored[run], args.workers, run)
        vs_r[run] = navtest.paired(scored[run], r_dir, labels_dir)
        read[run] = readouts(root / run / "dump-navtest", release_pick)
    # The arms the chain pairs against, read in phase 1 (C) or this phase.
    for v, arm, seeds in [("1", "C", (0, 1, 2))]:
        for s in seeds:
            r = name(v, arm, s)
            if r not in scored and previous is not None:
                scored[r] = previous / f"score-{r}"
                vs_r[r] = navtest.paired(scored[r], r_dir, labels_dir)
    pairs = {}
    seeds = (0, 1, 2) + ((3, 4) if args.extra_seeds else ())
    combos = [
        ("O-C", "1", "O", "1", "C"),
        ("HD-D", "1", "HD", "1", "D"),
        ("HD-C", "1", "HD", "1", "C"),
        ("H6-HD", "1", "H6", "1", "HD"),
        ("D-C", "1", "D", "1", "C"),
        ("H3-H6", "1", "H3", "1", "H6"),
    ]
    for v in ("2", "3"):
        combos += [
            (f"HD-D/B-2-{v}", v, "HD", v, "D"),
            (f"HD-C/B-2-{v}", v, "HD", v, "C"),
            (f"HD B-2-{v} - B-2-1", v, "HD", "1", "HD"),
        ]
    for label, va, a, vb, b in combos:
        for s in seeds:
            x, y = name(va, a, s), name(vb, b, s)
            if x in scored and y in scored:
                pairs[f"{label} s{s}"] = navtest.paired(scored[x], scored[y], labels_dir)
    record = {
        "pass": "b2-navtest",
        "phase": args.phase,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "r": str(r_dir),
        "vs_r": vs_r,
        "pairs": pairs,
        "readouts": read,
        "verdicts": verdicts(args.phase, pairs, vs_r),
    }
    (out / "navtest.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record["verdicts"], indent=1))
    return record


# -------------------------------------------------------------------------------------------- nuScenes


def nusc_numbers(proposals: np.ndarray, pick: np.ndarray, expert: np.ndarray, boxes: list) -> dict:
    """Per keyframe: the pick's L2 to the human at 1, 2 and 3 s, its collision within 3 s, and min ADE over 3
    s of the 32 (`ood.read_b2`'s `numbers`)."""
    from gallops.drivejepa_perception import nuscenes_data as nd

    plans = proposals[np.arange(len(pick)), pick]
    gap = np.linalg.norm(plans[:, :, :2] - expert[:, :, :2], axis=-1)
    return {
        "l2_1s": gap[:, 1],
        "l2_2s": gap[:, 3],
        "l2_3s": gap[:, 5],
        "collision_3s": nd.collided(plans, boxes, steps=6).astype(np.float64),
        "min_ade_3s": nd.min_ade(proposals, expert, steps=6),
    }


def nusc_read(cfg, args) -> dict:
    """Every run's nuScenes dump against the release on the same keyframes (module docstring)."""
    import pickle

    from gallops.drivejepa_perception import b_report

    out_root = Path(cfg.resolve("{runs}/drivejepa_perception"))
    nusc = out_root / "nusc-v1"
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a readout never overwrites one")
    outputs = np.load(nusc / "outputs.npz")
    rows = np.flatnonzero(outputs["built"])
    scenes = outputs["scene"][rows]
    expert = np.load(nusc / "labels.npz")["ego_future"][rows]
    with open(nusc / "future_boxes.pkl", "rb") as fh:
        future = pickle.load(fh)
    boxes = [future[r] for r in rows]
    release = nusc_numbers(outputs["proposals"][rows], outputs["pick"][rows], expert, boxes)
    result = {
        "pass": "b2-nusc",
        "rows": int(len(rows)),
        "scenes": int(len(set(scenes.tolist()))),
        "release": {k: float(v.mean()) for k, v in release.items()},
        "runs": {},
    }
    for run in sorted(Path(args.runs_root).glob("b2-*")):
        path = run / "dump-nusc" / "dump.npz"
        if not path.exists():
            continue
        dump = np.load(path)
        if not np.array_equal(dump["rows"], rows):
            raise SystemExit(f"{path} does not cover the built keyframes in order")
        mine = nusc_numbers(dump["proposals"], dump["pick"], expert, boxes)
        result["runs"][run.name] = {
            **{k: float(v.mean()) for k, v in mine.items()},
            "minus_release": {
                k: {
                    "mean": float((mine[k] - release[k]).mean()),
                    "se": b_report.clustered_se(mine[k] - release[k], scenes),
                }
                for k in mine
            },
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v["minus_release"]["l2_3s"] for k, v in result["runs"].items()}, indent=1))
    return result


# -------------------------------------------------------------------------------------------- analysis


def analyse(cfg, args) -> dict:
    """B-0-4 on every run's navtest dump, against R (module docstring)."""
    from gallops.drivejepa_perception import b_analysis, tokens, veto

    out_root = Path(cfg.resolve("{runs}/drivejepa_perception"))
    lab = out_root / veto.O_DIRS["navtest"][1]
    labels = np.load(lab / "labels.npz")
    logs = np.array(json.loads((lab / "manifest.json").read_text())["logs"])[labels["log_id"]]
    human, stamps = labels["ego_future"].astype(np.float64), labels["timestamp"]
    r_dir = out_root / "tokens-navtest-v1"
    r = {
        "proposals": np.load(r_dir / "proposals.npy"),
        "pick": np.load(r_dir / "pick.npy").astype(np.int64),
        "scores": tokens.release_scores(r_dir),
        "pdm": np.load(out_root / "pdm-navtest-v1" / "pdm.npz")["candidates"],
    }
    n = np.arange(len(r["pick"]))
    result = {
        "R": b_analysis.arm_summary(r["proposals"], r["pick"], human, logs, stamps, human[:, 0], r["pdm"]),
        "runs": {},
    }
    for run in sorted(Path(args.runs_root).glob("b2-*")):
        d = run / "dump-navtest"
        if not (d / "labels.npz").exists():
            continue
        dump = np.load(d / "dump.npz")
        if not np.array_equal(dump["rows"], n):
            raise SystemExit(f"{d} does not cover navtest in order")
        pdm = np.load(d / "labels.npz")["candidates"]
        moves = b_analysis.hungarian_moves(r["proposals"], dump["proposals"], r["scores"])
        pick_moved = np.linalg.norm(
            r["proposals"][n, r["pick"], :, :2] - dump["proposals"][n, dump["pick"], :, :2], axis=-1
        ).mean(-1)
        result["runs"][run.name] = {
            **b_analysis.arm_summary(dump["proposals"], dump["pick"], human, logs, stamps, human[:, 0], pdm),
            "from_R": {
                "matched_ade_m": moves["matched_ade_m"],
                "matched_ade_by_rank_quarter_m": moves["matched_ade_by_rank_quarter_m"],
                "pick_switch_rate": float((dump["pick"] != r["pick"]).mean()),
                "pick_moved_m": {
                    "mean": float(pick_moved.mean()),
                    "p99": float(np.quantile(pick_moved, 0.99)),
                },
            },
        }
        print(run.name, json.dumps(result["runs"][run.name]["from_R"]), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return result


# ----------------------------------------------------------------------------------------------- swaps


def swaps(cfg, args) -> dict:
    """Every run's test-time swaps against its own navtest dump (module docstring)."""
    from gallops.drivejepa_perception.b_report import clustered_se

    out_root = Path(cfg.resolve("{runs}/drivejepa_perception"))
    labels_dir = out_root / "labels-navtest-v1"
    labels = np.load(labels_dir / "labels.npz")
    logs = np.array(json.loads((labels_dir / "manifest.json").read_text())["logs"])[labels["log_id"]]
    result = {"moved_m": MOVED_M, "runs": {}}
    for run in sorted(Path(args.runs_root).glob("b2-*")):
        own_dir = run / "dump-navtest"
        if not (own_dir / "labels.npz").exists():
            continue
        own = np.load(own_dir / "dump.npz")
        n = np.arange(len(own["pick"]))
        own_final = np.load(own_dir / "labels.npz")["candidates"][n, own["pick"], -1]
        own_end = own["proposals"][n, own["pick"], -1, :2]
        entry = {}
        for kind in ("none", "log", "fold", "truth"):
            d = run / f"dump-navtest-swap-{kind}"
            if not (d / "labels.npz").exists():
                continue
            sw = np.load(d / "dump.npz")
            final = np.load(d / "labels.npz")["candidates"][n, sw["pick"], -1]
            moved = np.linalg.norm(sw["proposals"][n, sw["pick"], -1, :2] - own_end, axis=-1) > MOVED_M
            diff = (final - own_final) * 100
            entry[kind] = {
                "switch": float((sw["pick"] != own["pick"]).mean()),
                "moved": float(moved.mean()),
                "final_minus_own": float(diff.mean()),
                "se": clustered_se(diff, logs),
            }
        if entry:
            result["runs"][run.name] = entry
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["runs"], indent=1))
    return result


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.b2_read")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rclean")
    p.add_argument("--runs-root", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("navtest")
    p.add_argument("--phase", required=True, choices=tuple(PHASES))
    p.add_argument("--runs-root", required=True)
    p.add_argument(
        "--previous", default="", help="phase 2: phase 1's read, whose C runs the chain pairs with"
    )
    p.add_argument("--extra-seeds", action="store_true", help="the noise-floor rule fired: seeds 3 and 4 too")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--out", required=True)
    p = sub.add_parser("nusc")
    p.add_argument("--runs-root", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("swaps")
    p.add_argument("--runs-root", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("analyse")
    p.add_argument("--runs-root", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "swaps":
        swaps(SiteConfig.load(), args)
        return
    if args.cmd == "analyse":
        analyse(SiteConfig.load(), args)
        return
    if args.cmd == "rclean":
        rclean(SiteConfig.load(), args)
        return
    if args.cmd == "nusc":
        nusc_read(SiteConfig.load(), args)
        return
    navtest_read(SiteConfig.load(), args)


if __name__ == "__main__":
    main()
