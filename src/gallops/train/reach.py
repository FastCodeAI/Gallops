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
    expert reach     max |y| of the plan it is being taught, which is the only
                     honest thing to compare the selected reach against
    slowing          how much shorter the driven plan is than carrying on at the
                     current speed, beside the same figure for the expert's plan
    escape rate      is any of the 32 clear of the hazard and on the road

**Lateral reach cannot see a brake.** It was the right measure while the question
was whether the refiner would ever offer a sideways plan at all. Once the expert
prefers braking for anything not closing on us -- which is most of the corpus --
a model doing exactly the right thing scores nothing on it. `slowing` is the
other half, and the two together say what the plan actually is.

**Compare against the expert's geometry, not against zero.** An earlier version
reported `expert_offset_m`, the offset the ladder COMMANDED, which is 0 for
`continue` by construction. That is not the shape of the plan: scene 0053 curves,
so a correct `continue` plan there sweeps 1.93 m sideways in the ego frame while
scene 0075, which is straight, sweeps 0.17 m. Read against a constant 0 the first
looks like a model drifting badly and the second like one behaving; both readings
are artefacts of the wrong baseline. This cost a wrong diagnosis once already.

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
from gallops.expert.grader import grade
from gallops.train.swerve_data import episode_context


def sample(ticks: list[dict], limit: int) -> list[dict]:
    """Evenly spaced, so the sample spans whole episodes instead of their first seconds."""
    if not limit or limit >= len(ticks):
        return ticks
    return [ticks[i] for i in np.linspace(0, len(ticks) - 1, limit).round().astype(int)]


def measure(model, dataset, ticks, torch, drivable: Path | None, batch: int = 8) -> dict:
    """Reach and escape rate over `ticks`, split by what the expert asked for."""
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ticks), batch):
            chunk = list(range(start, min(start + batch, len(ticks))))
            features, _, used = dataset.batch(chunk)
            if features is None:
                continue
            output = model(features)
            proposals = output["proposals"].float().cpu().numpy()
            selected = output["trajectory"].float().cpu().numpy()
            # `used`, not `chunk`: batch skips a tick it cannot rebuild, and pairing
            # by position would then grade one tick's plans against the next.
            for offset, tick in enumerate(used):
                ego2world, targets, area = episode_context(tick, drivable)
                stamp = int(tick["timestamp"])
                verdicts = [grade(p, ego2world, stamp, targets, None, area) for p in proposals[offset]]
                # 0 means it holds its speed; 1 means it stops dead.
                free_running = max(float(tick["speed"]) * 4.0, 1e-6)
                rows.append(
                    {
                        "kind": tick["expert_kind"],
                        "slowing": 1.0 - _length(selected[offset]) / free_running,
                        "expert_slowing": 1.0 - _length(tick["expert"]) / free_running,
                        "proposal_reach": float(np.abs(proposals[offset][:, :, 1]).max()),
                        "selected_reach": float(np.abs(selected[offset][:, 1]).max()),
                        "escape": any(v.clear for v in verdicts),
                        "expert_offset": abs(tick["expert_offset_m"]),
                        "expert_reach": float(np.abs(np.asarray(tick["expert"])[:, 1]).max()),
                    }
                )
    return {"rows": rows}


def _length(waypoints) -> float:
    """Path length of a plan from the ego, in metres."""
    points = np.vstack([[0.0, 0.0], np.asarray(waypoints)[:, :2]])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def report(name: str, rows: list[dict]) -> None:
    print(f"\n  {name}")
    print(
        f"    {'ticks':<12}{'n':>5}{'proposals':>11}{'sideways':>10}{'expert':>9}"
        f"{'slows':>8}{'expert':>8}{'escape':>9}"
    )
    kinds = ("swerve", "brake", "continue", "least-impact")
    groups = [(k, [r for r in rows if r["kind"] == k]) for k in kinds] + [("all", rows)]
    for label, group in groups:
        if not group:
            continue
        print(
            f"    {label:<12}{len(group):>5}"
            f"{np.median([r['proposal_reach'] for r in group]):>11.2f}"
            f"{np.median([r['selected_reach'] for r in group]):>10.2f}"
            f"{np.median([r.get('expert_reach', np.nan) for r in group]):>9.2f}"
            f"{np.median([r.get('slowing', np.nan) for r in group]):>8.0%}"
            f"{np.median([r.get('expert_slowing', np.nan) for r in group]):>8.0%}"
            f"{np.mean([r['escape'] for r in group]):>9.0%}"
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
