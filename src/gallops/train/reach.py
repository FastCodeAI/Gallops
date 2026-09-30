"""Did stage A actually move the refiner? Measure the 32, before and after.

Stage A exists to answer one question: the released model's widest proposal reaches
0.70 m sideways on a frontal tick, which is not an evasion of anything. Can the
refiner be moved to offer a real one while almost everything stays frozen?

A falling training loss does not answer that -- it is the min over 32 proposals, so
it falls when ONE proposal creeps toward the target. What answers it is lateral
reach on the ticks where the expert swerves, and, just as importantly, NO new reach
on the ticks where the expert says carry on. A model that swerves everywhere has
not learned to evade; it has learned to swerve.

Three numbers per checkpoint, on a scene held out of training:

    proposal reach   max |y| over the 32 x 8 poses -- what the refiner can offer
    selected reach   max |y| of the one the scorer picks -- what would be DRIVEN
    escape rate      is any of the 32 clear of the hazard and on the road

The gap between the first two is the stage B case: if the refiner offers an escape
and the frozen scorer keeps choosing a straight line, the scorer is what to train
next. If the refiner never offers one, stage B is pointless.

    <drive-jepa python> -m gallops.train.reach --labelled <dir> --run <dir> [--limit 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gallops.config import SiteConfig
from gallops.expert.drivable import DrivableArea
from gallops.expert.grader import grade


def sample(ticks: list[dict], limit: int) -> list[dict]:
    """Evenly spaced, so the sample spans whole episodes instead of their first seconds."""
    if not limit or limit >= len(ticks):
        return ticks
    return [ticks[i] for i in np.linspace(0, len(ticks) - 1, limit).round().astype(int)]


def episode_context(tick: dict, drivable: Path | None) -> tuple:
    """The actors and pose needed to grade a proposal, from the episode on disk."""
    directory = Path(tick["frame"]).parent.parent
    poses = json.loads((directory / "ego_poses.json").read_text())
    actors = json.loads((directory / "actors.json").read_text())
    area = None
    if drivable:
        scene = "scene-" + directory.parent.name.rsplit("-", 1)[-1]
        path = drivable / f"{scene}.npz"
        if path.exists():
            area = DrivableArea.load(path)
    return (
        np.asarray(poses[str(int(tick["timestamp"]))]),
        [a for a in actors if a["is_target_actor"]],
        area,
    )


def measure(model, dataset, ticks, torch, drivable: Path | None, batch: int = 8) -> dict:
    """Reach and escape rate over `ticks`, split by what the expert asked for."""
    rows = []
    contexts: dict[str, tuple] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ticks), batch):
            chunk = list(range(start, min(start + batch, len(ticks))))
            features, _ = dataset.batch(chunk)
            if features is None:
                continue
            output = model(features)
            proposals = output["proposals"].float().cpu().numpy()
            selected = output["trajectory"].float().cpu().numpy()
            for offset, index in enumerate(chunk):
                if offset >= len(proposals):
                    break
                tick = ticks[index]
                key = str(Path(tick["frame"]).parent.parent)
                if key not in contexts:
                    contexts[key] = episode_context(tick, drivable)
                ego2world, targets, area = contexts[key]
                stamp = int(tick["timestamp"])
                verdicts = [grade(p, ego2world, stamp, targets, None, area) for p in proposals[offset]]
                rows.append(
                    {
                        "kind": tick["expert_kind"],
                        "proposal_reach": float(np.abs(proposals[offset][:, :, 1]).max()),
                        "selected_reach": float(np.abs(selected[offset][:, 1]).max()),
                        "escape": any(v.clear for v in verdicts),
                        "expert_offset": abs(tick["expert_offset_m"]),
                    }
                )
    return {"rows": rows}


def report(name: str, rows: list[dict]) -> None:
    swerve = [r for r in rows if r["kind"] == "swerve"]
    carry_on = [r for r in rows if r["kind"] == "continue"]
    print(f"\n  {name}")
    print(f"    {'ticks':<10}{'n':>5}{'proposals':>11}{'selected':>10}{'escape':>9}{'expert':>9}")
    for label, group in (("swerve", swerve), ("continue", carry_on), ("all", rows)):
        if not group:
            continue
        print(
            f"    {label:<10}{len(group):>5}"
            f"{np.median([r['proposal_reach'] for r in group]):>11.2f}"
            f"{np.median([r['selected_reach'] for r in group]):>10.2f}"
            f"{np.mean([r['escape'] for r in group]):>9.0%}"
            f"{np.median([r['expert_offset'] for r in group]):>9.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled", type=Path, required=True)
    parser.add_argument("--run", type=Path, help="a run directory of step_* checkpoints")
    parser.add_argument("--held-out", default="0058")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--drivable", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from gallops.train.swerve import build_model, load_ticks

    cfg = SiteConfig.load()
    import os

    for name, template in (
        ("NAVSIM_DEVKIT_ROOT", "{data.drivejepa_repo}"),
        ("NAVSIM_EXP_ROOT", "{data.navsim.exp}"),
        ("OPENSCENE_DATA_ROOT", "{data.navsim}"),
        ("NUPLAN_MAPS_ROOT", "{data.navsim}/maps"),
    ):
        os.environ.setdefault(name, cfg.resolve(template))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")

    # The held-out scene, so this measures a model on ticks it never trained on.
    ticks = [t for t in load_ticks(args.labelled) if args.held_out in t["episode"]]
    ticks = sample(ticks, args.limit)
    print(f"  {len(ticks)} held-out ticks from scene {args.held_out}")

    released = cfg.resolve(
        "{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
    )
    _agent, model, torch = build_model(released, args.device, cfg.resolve("{data.drivejepa_repo}"))
    from gallops.train.swerve_data import TickDataset

    dataset = TickDataset(ticks, args.device)

    results = {}
    baseline = measure(model, dataset, ticks, torch, args.drivable, args.batch)
    report("released (what the benchmark measured)", baseline["rows"])
    results["released"] = baseline["rows"]

    for directory in sorted((args.run or Path()).glob("step_*")) if args.run else []:
        state = torch.load(directory / "model.pt", map_location=args.device, weights_only=True)
        model.load_state_dict(state)
        rows = measure(model, dataset, ticks, torch, args.drivable, args.batch)["rows"]
        report(directory.name, rows)
        results[directory.name] = rows

    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
