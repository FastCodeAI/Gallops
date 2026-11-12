"""B-0-4, the proposal analysis every Part B experiment uses: how two arms' 32 proposals and picks differ.

    python -m gallops.drivejepa_perception.b_analysis compare --root {out} --split navtest \
        --a R=tokens-navtest-v1 --b R-tilde=rtilde-navtest-v1 --pdm-a pdm-navtest-v1 \
        --out {out}/part-b/b0/analysis-R-Rtilde.json

docs/drivejepa-perception-sidecar/part-b/B-0-foundations.md, B-0-4. An arm is a directory holding its 32
proposals and its pick per row (`proposals.npy`, `pick.npy`: the token pass for R, `cached_release` for
R-tilde, a trained arm's dump later). Per arm, on one split:
- **best of 32** and the share of the 32 that pass NC and TTC, where the arm's proposals have PDM labels;
- **the 32's spread:** the endpoints' lateral and longitudinal spread;
- **minADE and the miss rate** to the human's plan, by `train_generator.evaluate`'s definitions (ADE over the
  first 6 poses, the best of 32; a miss over 2 m);
- **the pick's steadiness between consecutive frames** (`steadiness`), a proxy for EC.
Between the two arms:
- **which proposals moved:** a Hungarian match of one arm's 32 to the other's, by ADE (`hungarian_moves`),
  with the matched displacement overall and by the first arm's score rank;
- **the pick:** how often it switches, and the matched displacement of the pick.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NC, TTC, FINAL = 0, 5, 8
#: train_generator.evaluate: ADE over the first 6 poses (3 s), a miss beyond 2 m.
ADE_POSES, MISS_M = 6, 2.0
#: Consecutive frames of a log are 0.5 s apart; a gap is a pair further apart than this.
NEXT_FRAME_US = 550_000


def best_of_k(pdm: np.ndarray) -> float:
    return float(pdm[..., FINAL].max(1).mean())


def safe_share(pdm: np.ndarray) -> float:
    return float(((pdm[..., NC] == 1) & (pdm[..., TTC] == 1)).mean())


def spread(proposals: np.ndarray) -> dict:
    """The mean over rows of the 32 endpoints' standard deviation, longitudinal and lateral, metres."""
    end = np.asarray(proposals)[:, :, -1, :2]
    return {"longitudinal_m": float(end[..., 0].std(1).mean()), "lateral_m": float(end[..., 1].std(1).mean())}


def min_ade_miss(proposals: np.ndarray, human: np.ndarray) -> dict:
    """minADE over the first ADE_POSES poses to the human's plan, and the share of rows missing by MISS_M."""
    d = np.linalg.norm(np.asarray(proposals)[..., :ADE_POSES, :2] - human[:, None, :ADE_POSES, :2], axis=-1)
    ade = d.mean(-1).min(1)
    return {"min_ade_m": float(ade.mean()), "miss_rate": float((ade > MISS_M).mean())}


def hungarian_moves(props_a: np.ndarray, props_b: np.ndarray, scores_a: np.ndarray) -> dict:
    """Match each row's 32 of arm A to arm B's by ADE (one to one), and how far they moved: overall, and by
    A's score rank in quarters (first: A's best eight)."""
    from scipy.optimize import linear_sum_assignment

    moved, by_rank = [], [[] for _ in range(4)]
    match = np.zeros(props_a.shape[:2], np.int64)
    for r in range(len(props_a)):
        cost = np.linalg.norm(props_a[r][:, None, :, :2] - props_b[r][None, :, :, :2], axis=-1).mean(-1)
        rows, cols = linear_sum_assignment(cost)
        match[r, rows] = cols
        d = cost[rows, cols]
        moved.append(d.mean())
        rank = np.argsort(np.argsort(-scores_a[r]))
        for q in range(4):
            by_rank[q].append(d[(rank >= 8 * q) & (rank < 8 * (q + 1))].mean())
    return {
        "matched_ade_m": float(np.mean(moved)),
        "matched_ade_by_rank_quarter_m": [float(np.mean(x)) for x in by_rank],
        "match": match,
    }


def steadiness(plans: np.ndarray, logs: np.ndarray, stamps: np.ndarray, ego_next: np.ndarray) -> dict:
    """How far a frame's plan moves at the next frame of its log, where the two overlap (an EC proxy).

    Frame t's plan, poses at t + 0.5 k, is set against frame t+1's, moved into frame t through the ego's own
    pose at t + 0.5 s (`ego_next`, (N, 3) x y yaw in frame t): pose k + 1 of t against pose k of t+1."""
    plans, ego_next = np.asarray(plans, np.float64), np.asarray(ego_next, np.float64)
    gaps = []
    for log in np.unique(logs):
        index = np.flatnonzero(logs == log)
        index = index[np.argsort(stamps[index], kind="stable")]
        for a, b in zip(index[:-1], index[1:]):  # noqa: B905 -- Python 3.9
            if stamps[b] - stamps[a] >= NEXT_FRAME_US:
                continue
            x, y, yaw = ego_next[a]
            c, s = np.cos(yaw), np.sin(yaw)
            later = plans[b][:-1, :2] @ np.array([[c, s], [-s, c]]) + np.array([x, y])
            gaps.append(np.linalg.norm(plans[a][1:, :2] - later, axis=-1).mean())
    gaps = np.asarray(gaps)
    return {"pairs": int(len(gaps)), "mean_m": float(gaps.mean()) if len(gaps) else float("nan")}


def arm_summary(proposals, pick, human, logs, stamps, ego_next, pdm=None) -> dict:
    n = np.arange(len(pick))
    out = {"spread": spread(proposals), **min_ade_miss(proposals, human)}
    out["pick_steadiness"] = steadiness(np.asarray(proposals)[n, pick], logs, stamps, ego_next)
    if pdm is not None:
        out["best_of_32"] = best_of_k(pdm)
        out["safe_share"] = safe_share(pdm)
        out["pick_final"] = float(pdm[n, pick, FINAL].mean())
    return out


def compare(root: Path, split: str, a: tuple, b: tuple, pdm_a=None, pdm_b=None) -> dict:
    """Two arms on a split (module docstring). `a`, `b`: (name, directory under `root`)."""
    from gallops.drivejepa_perception import tokens, veto

    root = Path(root)
    lab = veto.O_DIRS[split][1]
    labels = np.load(root / lab / "labels.npz")
    manifest = json.loads((root / lab / "manifest.json").read_text())
    logs = np.array(manifest["logs"])[labels["log_id"]]
    human, stamps = labels["ego_future"].astype(np.float64), labels["timestamp"]
    arms = {}
    for (name, directory), pdm_dir in ((a, pdm_a), (b, pdm_b)):
        d = root / directory
        pdm = None if pdm_dir is None else np.load(root / pdm_dir / "pdm.npz")["candidates"]
        arms[name] = {
            "proposals": np.load(d / "proposals.npy"),
            "pick": np.load(d / "pick.npy").astype(np.int64),
            "scores": tokens.release_scores(d),
            "pdm": pdm,
        }
    (na, A), (nb, B) = arms.items()
    n = np.arange(len(A["pick"]))
    moves = hungarian_moves(A["proposals"], B["proposals"], A["scores"])
    pick_moved = np.linalg.norm(
        A["proposals"][n, A["pick"], :, :2] - B["proposals"][n, B["pick"], :, :2], axis=-1
    ).mean(-1)
    return {
        "split": split,
        "rows": int(len(n)),
        "arms": {
            name: arm_summary(x["proposals"], x["pick"], human, logs, stamps, human[:, 0], x["pdm"])
            for name, x in arms.items()
        },
        f"{na}_to_{nb}": {
            "matched_ade_m": moves["matched_ade_m"],
            "matched_ade_by_rank_quarter_m": moves["matched_ade_by_rank_quarter_m"],
            "pick_switch_rate": float((A["pick"] != B["pick"]).mean()),
            "pick_moved_m": {"mean": float(pick_moved.mean()), "p99": float(np.quantile(pick_moved, 0.99))},
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.b_analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--root", required=True)
    p.add_argument("--split", choices=("navtest", "fresh"), required=True)
    p.add_argument("--a", required=True, help="NAME=DIR under --root, with proposals.npy and pick.npy")
    p.add_argument("--b", required=True)
    p.add_argument("--pdm-a", default="")
    p.add_argument("--pdm-b", default="")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    record = compare(
        Path(args.root),
        args.split,
        tuple(args.a.split("=", 1)),
        tuple(args.b.split("=", 1)),
        args.pdm_a or None,
        args.pdm_b or None,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
