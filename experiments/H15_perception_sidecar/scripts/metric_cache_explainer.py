"""The numbers and figures behind docs/navsim-metric-cache.md (H15 S0.6). One-off: rerun to refresh.

    {python.drivejepa} experiments/H15_perception_sidecar/scripts/metric_cache_explainer.py \
        --labels LABELS --coverage-labels {runs}/drivejepa_perception/labels-v1 --tokens TOKENS --out OUT

Writes OUT_DIR/explainer.json (coverage, one record's fields, three worked examples) and
OUT_DIR/example_*.png (BEV figures). The scoring is `pdm_labels.score`, the code phase B uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))


def coverage(root: Path, logs: list, tokens: list) -> dict:
    from gallops.drivejepa_perception.pdm_labels import metric_cache_path

    paths = [metric_cache_path(root, log, token) for log, token in zip(logs, tokens)]  # noqa: B905
    with ThreadPoolExecutor(32) as pool:
        exists = np.array(list(pool.map(lambda p: p.exists(), paths)))
    return {"navtrain_samples": len(paths), "with_metric_cache": int(exists.sum()), "share": float(exists.mean())}


FIELDS = (
    "trajectory", "human_trajectory", "past_human_trajectory", "ego_state", "observation", "centerline",
    "route_lane_ids", "drivable_area_map", "past_detections_tracks", "current_tracked_objects",
    "future_tracked_objects", "map_parameters",
)


def describe(cache) -> dict:
    """What one record holds, and which of the dataclass's fields the pruning left empty."""
    observation = cache.observation
    return {
        "fields_present": {name: getattr(cache, name) is not None for name in FIELDS},
        "log_name": cache.log_name,
        "scene_type": str(cache.scene_type),
        "ego_rear_axle": [cache.ego_state.rear_axle.x, cache.ego_state.rear_axle.y, cache.ego_state.rear_axle.heading],
        "ego_speed_mps": float(cache.ego_state.dynamic_car_state.speed),
        "pdm_closed_reference_poses": len(cache.trajectory.get_sampled_trajectory()),
        "past_human_poses": len(cache.past_human_trajectory.get_sampled_trajectory()),
        "observation_steps_10hz": len(observation._occupancy_maps),
        "objects_in_observation_at_t": len(observation[0].tokens),
        "route_lane_ids": len(cache.route_lane_ids),
        "centerline_points": len(cache.centerline.discrete_path),
        "drivable_area_polygons": len(cache.drivable_area_map.tokens),
        "map": cache.map_parameters.map_name,
    }


def figure(cache, proposals, scores, human, pick, collided, path: Path) -> None:
    """BEV in the ego frame: drivable area, centerline, agents now and their next 4 s, the 32 candidates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gallops.drivejepa_perception.navsim_data import to_local, track_id

    ego = np.array([cache.ego_state.rear_axle.x, cache.ego_state.rear_axle.y, cache.ego_state.rear_axle.heading])
    fig, ax = plt.subplots(figsize=(7, 7))
    for polygon in cache.drivable_area_map._geometries:
        xs, ys = polygon.exterior.xy
        local = to_local(np.c_[xs, ys], ego)
        ax.fill(local[:, 1] * -1, local[:, 0], color="#e5e7eb", zorder=0)
    centre = to_local(np.array([[p.x, p.y] for p in cache.centerline.discrete_path]), ego)
    ax.plot(-centre[:, 1], centre[:, 0], color="#9ca3af", lw=1, ls="--", zorder=1, label="route centerline")
    now = cache.observation[0]
    for token, polygon in zip(now.tokens, now._geometries):  # noqa: B905 -- Python 3.9
        xs, ys = polygon.exterior.xy
        corners = to_local(np.c_[xs, ys], ego)
        is_hit = track_id(token) in collided
        ax.fill(-corners[:, 1], corners[:, 0], color="#dc2626" if is_hit else "#2563eb", alpha=0.8, zorder=3)
    for k in np.argsort(scores[:, -1]):
        colour = plt.cm.RdYlGn(float(scores[k, -1]))
        ax.plot(-proposals[k, :, 1], proposals[k, :, 0], color=colour, lw=2.5 if k == pick else 1, zorder=4)
    ax.plot(-human[:, 1], human[:, 0], color="black", lw=2, ls=":", zorder=5, label="human")
    ax.plot([0], [0], marker="^", color="black", ms=10, zorder=6)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-10, 50)
    ax.set_aspect("equal")
    ax.set_xlabel("← left      lateral position (m)      right →")
    ax.set_ylabel("x forward (m)")
    ax.set_title("final score: red 0 → green 1 · thick: the release's pick · red box: collided with", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    from gallops.config import SiteConfig
    from gallops.drivejepa_perception import env, pdm_labels

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="the labels the --tokens rows were built from")
    parser.add_argument("--coverage-labels", required=True, help="the full navtrain labels, for coverage")
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = SiteConfig.load()
    env.enter(cfg)
    from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as cns

    root = Path(cfg.resolve("{data.navsim_metric_cache}"))
    labels = np.load(Path(args.labels) / "labels.npz")
    logs = json.loads((Path(args.labels) / "manifest.json").read_text())["logs"]
    token_logs = [logs[i] for i in labels["log_id"]]
    tokens = [str(t) for t in labels["token"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full = np.load(Path(args.coverage_labels) / "labels.npz")
    full_logs = json.loads((Path(args.coverage_labels) / "manifest.json").read_text())["logs"]
    report = {"coverage": coverage(root, [full_logs[i] for i in full["log_id"]], [str(t) for t in full["token"]])}

    proposals = np.load(Path(args.tokens) / "proposals.npy")
    picks = np.load(Path(args.tokens) / "pick.npy")
    rows = range(len(proposals))
    scored = []
    for row in rows:
        path = pdm_labels.metric_cache_path(root, token_logs[row], tokens[row])
        if not path.exists():
            continue
        scores, agents, steps = pdm_labels.score(path, proposals[row].astype(np.float64))
        human, _, _ = pdm_labels.score(path, labels["ego_future"][row][None].astype(np.float64))
        scored.append((row, path, scores, agents, steps, human[0]))
    # Example 1: the row whose candidates' final scores spread most; example 3: a collision verdict.
    spread = max(scored, key=lambda s: float(np.ptp(s[2][:, -1])))
    collisions = [s for s in scored if (s[3] >= 0).any()]
    examples = {"spread": spread}
    if collisions:
        examples["collision"] = max(collisions, key=lambda s: int((s[3] >= 0).sum()))
    report["record"] = describe(cns.load_metric_cache(spread[1]))
    report["examples"] = {}
    for name, (row, path, scores, agents, steps, human) in examples.items():
        cache = cns.load_metric_cache(path)
        report["examples"][name] = {
            "token": tokens[row],
            "log": token_logs[row],
            "release_pick": int(picks[row]),
            "best_of_32": int(np.argmax(scores[:, -1])),
            "columns": list(pdm_labels.COLUMNS),
            "candidates": np.round(scores, 3).tolist(),
            "human": np.round(human, 3).tolist(),
            "collision_agent": [int(a) for a in agents],
            "collision_step_10hz": [int(s) for s in steps],
        }
        figure(cache, proposals[row], scores, labels["ego_future"][row], int(picks[row]), set(agents[agents >= 0].tolist()), out / f"example_{name}.png")
    (out / "explainer.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"coverage": report["coverage"], "record": report["record"]}, indent=1))


if __name__ == "__main__":
    main()
