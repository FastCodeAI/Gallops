"""Turn collected episodes into a training set: what to imitate, and what to grade.

Step 3 of the finetune. Every tick of every collected episode becomes one example
carrying both halves of what Drive-JEPA has to learn:

* **the expert's trajectory**, for the winner-takes-all loss the vendor already
  uses (`trajectory_loss_anchors` in `drive_jepa_agent.py`): the distance from
  the target to the CLOSEST of the 32 proposals, which pulls one proposal toward
  a swerve without collapsing the other 31;
* **a verdict on each of the 32 candidates the model proposed**, for the scorer.

Nothing here re-renders or re-runs the model. The engine already wrote the frames
the model read, its own trajectory, the actors' full paths and the logged route;
the dump adapter wrote all 32 candidates per tick. They join on the timestamp
that names the frame file.

Episodes are paired to their dumps by tick timestamps, then to their frames by
the same. That pairing is ambiguous only when two seeds drive identically until
one collides a tick earlier, which cannot happen here: each seed jitters the
actor, so the dumps differ from the first tick.

The output is one JSONL per episode, and deliberately holds PATHS to frames
rather than pixels -- the frames are already on disk in the form the model reads,
and copying them into a dataset would triple the storage for nothing.

    python -m gallops.expert.label --collect <dir> --dumps <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .draw import teach
from .drivable import DrivableArea
from .grader import first_hit, grade


def episode_dirs(collect: Path):
    for directory in sorted(collect.glob("*-*/run_*")) + sorted(collect.glob("*-*")):
        if (directory / "trajectories.json").exists() and (directory / "metrics.json").exists():
            yield directory


def match_dump(directory: Path, dumps: Path) -> list[dict] | None:
    """The dump whose ticks are this episode's.

    A dump holds the episode's ticks plus the engine's priming calls, and the
    impact tick when it collided -- the last is not in `trajectories.json`.
    """
    plans = json.loads((directory / "trajectories.json").read_text())
    stamps = {int(t) for t in plans}
    for path in sorted(dumps.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if rows and stamps and stamps <= {r["timestamp"] for r in rows}:
            drift = max(
                np.abs(np.asarray(r["trajectory"])[:6, :2] - np.asarray(plans[str(r["timestamp"])])).max()
                for r in rows
                if str(r["timestamp"]) in plans
            )
            if drift < 1e-3:
                return rows
    return None


def label_episode(directory: Path, rows: list[dict], area=None) -> list[dict]:
    actors = json.loads((directory / "actors.json").read_text())
    targets = [a for a in actors if a["is_target_actor"]]
    poses = json.loads((directory / "ego_poses.json").read_text())
    logged = json.loads((directory / "reference_trajectory.json").read_text())
    reference = np.asarray([v for _, v in sorted(logged.items(), key=lambda kv: int(kv[0]))])
    frames = {int(p.stem): p for p in (directory / "CAM_FRONT").glob("*.jpg")}

    out = []
    for row in rows:
        stamp = row["timestamp"]
        if str(stamp) not in poses:
            continue  # a priming call or the impact tick: no pose was logged
        ego2world = np.asarray(poses[str(stamp)])
        plan = teach(ego2world, stamp, float(row["speed"]), reference, targets, area=area)
        proposals = np.asarray(row["proposals"])
        # The scorer's labels come from the SAME judgement the expert was held to,
        # the road included: a candidate that dodges onto the pavement is not a
        # good option, and grading it clear would teach exactly that.
        graded = [grade(p, ego2world, stamp, targets, reference, area) for p in proposals]
        verdicts = [v.hit_at for v in graded]
        driven = first_hit(np.asarray(row["trajectory"]), ego2world, stamp, targets)
        out.append(
            {
                "episode": f"{directory.parent.name}/{directory.name}",
                "timestamp": stamp,
                "frame": str(frames[stamp]) if stamp in frames else None,
                "speed": row["speed"],
                "accel": row["accel"],
                "command": row["command"],
                # what to imitate
                "expert": np.asarray(plan.waypoints).round(4).tolist(),
                "expert_kind": plan.kind,
                "expert_clear": plan.clear,
                "expert_offset_m": round(plan.offset_m, 3),
                # what to grade: None means that candidate stays clear
                "candidate_hits": verdicts,
                "candidate_off_road": [v.off_road_at for v in graded],
                "candidate_clear": [v.clear for v in graded],
                "driven_hits": driven,
                "pdm_score": row.get("pdm_score"),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", type=Path, required=True)
    parser.add_argument("--dumps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drivable", type=Path, help="directory of scene-*.npz rasters")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    areas: dict[str, DrivableArea] = {}
    if args.drivable:
        for path in sorted(args.drivable.glob("scene-*.npz")):
            areas[path.stem.removeprefix("scene-")] = DrivableArea.load(path)
        print(f"  drivable-area rasters for {len(areas)} scenes")
    totals = {"episodes": 0, "ticks": 0, "expert_clear": 0, "driven_hits": 0, "rescued": 0, "no_frame": 0}
    totals["candidates_clear"] = 0
    totals["candidates"] = 0
    kinds: dict[str, int] = {}
    unmatched = []
    for directory in episode_dirs(args.collect):
        rows = match_dump(directory, args.dumps)
        if rows is None:
            unmatched.append(str(directory))
            continue
        # "frontal-0075/run_3" -> the raster built for scene-0075
        scene = directory.parent.name.rsplit("-", 1)[-1]
        labelled = label_episode(directory, rows, areas.get(scene))
        if not labelled:
            continue
        name = f"{directory.parent.name}__{directory.name}.jsonl"
        with (args.out / name).open("w") as handle:
            for record in labelled:
                handle.write(json.dumps(record) + "\n")
        totals["episodes"] += 1
        totals["ticks"] += len(labelled)
        for record in labelled:
            totals["expert_clear"] += record["expert_clear"]
            totals["driven_hits"] += record["driven_hits"] is not None
            totals["rescued"] += record["driven_hits"] is not None and record["expert_clear"]
            totals["no_frame"] += record["frame"] is None
            totals["candidates"] += len(record["candidate_clear"])
            totals["candidates_clear"] += sum(record["candidate_clear"])
            kinds[record["expert_kind"]] = kinds.get(record["expert_kind"], 0) + 1

    ticks = max(totals["ticks"], 1)
    print(f"  {totals['episodes']} episodes, {totals['ticks']} ticks -> {args.out}")
    print(f"  the model hits on            {totals['driven_hits'] / ticks:6.1%} of ticks")
    print(f"  the expert finds a clear plan{totals['expert_clear'] / ticks:6.1%}")
    print(f"  it rescues                   {totals['rescued']:5} of {totals['driven_hits']} hitting ticks")
    print(f"  ticks with no saved frame    {totals['no_frame']:5}  (unusable for training)")
    share = totals["candidates_clear"] / max(totals["candidates"], 1)
    print(f"  of the model's own candidates, {share:6.1%} are clear")
    print(f"  what the expert chose: {kinds}")
    if unmatched:
        print(f"  UNMATCHED episodes (no dump): {len(unmatched)}")
        for path in unmatched[:5]:
            print(f"    {path}")


if __name__ == "__main__":
    main()
