"""Does a checkpoint pick a safe plan, on moments recorded from real collection runs?

The open-loop instrument behind the l5, s1, s2 and s3 conclusions. For one scene
and one hazard type it replays two groups of labelled ticks through each model:

    creep     ego under 3 m/s and the plan it actually drove hits the car --
              the state behind every neuroNCAP stationary collision (1.4-2.5 m/s)
    approach  ego at 3 m/s or more and the expert's choice is one of --kind
              (default brake; --shape frontal --kind swerve for oncoming cars)

Per tick and model: is the picked plan clear, why not (hits the inserted car,
leaves the road, strays more than 4 m from the lane reference, or hits ANY other
car in the scene), how fast it ends, how far it goes against the expert, and where
the scorer ranks the best clear plan. The labeller and trainer grade against the
inserted car only; the any-car column is the one place a parked car is seen.

GUARD. The released model's 32 plans, rebuilt here, are graded exactly as
labelling graded them (lane reference included) and compared with the verdicts
labelling recorded. Below GUARD_MIN_AGREEMENT the grading here is not the
grading the labels and the trainer use, so the results are moved aside and not
reported. Until 2026-10-08 the ops copy of this check graded against the logged
human path while the labels used the lane centre: 99.8% agreement on 0053, which
looked fine, and 96.1% on 0070, which exposed it.

    <drive-jepa python> -m gallops.train.safety_check --labelled <dir> --drivable <dir> \\
        --scene 0053 --checkpoint s1=<model.pt> --out check.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from gallops.config import SiteConfig
from gallops.expert.grader import first_hit, grade
from gallops.expert.lanes import load_references, reference_at
from gallops.train.swerve import build_model, load_ticks
from gallops.train.swerve_data import TickDataset, episode_context

#: A plan that ends below this speed counts as coming to rest.
REST_MPS = 0.5
#: Above this speed a tick is an approach; below it, a creep if the driven plan hits.
CREEP_BELOW_MPS = 3.0
#: Approach ticks are thinned evenly to at most this many per scene and shape.
MAX_APPROACH = 120
#: The share of the released model's candidate verdicts that must match the labels
#: before any result is reported. The trainer's preflight demands an exact match on
#: its sample; across a whole scene a few verdicts flip on rebuilt inputs.
GUARD_MIN_AGREEMENT = 0.995


def length(waypoints) -> float:
    """Path length from the ego through the waypoints, metres."""
    points = np.vstack([[0.0, 0.0], np.asarray(waypoints)[:, :2]])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def end_speed(waypoints) -> float:
    """Speed over the last 0.5 s step of a plan, m/s."""
    w = np.asarray(waypoints)[:, :2]
    return float(np.linalg.norm(w[-1] - w[-2]) / 0.5)


def groups(ticks: list[dict], scene: str, shape: str = "stationary", kinds=("brake",)) -> dict:
    """The creep and approach ticks of one scene and hazard type."""
    mine = [t for t in ticks if t["episode"].startswith(f"{shape}-{scene}")]
    creep = [t for t in mine if t["speed"] < CREEP_BELOW_MPS and t["driven_hits"] is not None]
    approach = [t for t in mine if t["speed"] >= CREEP_BELOW_MPS and t["expert_kind"] in kinds]
    if len(approach) > MAX_APPROACH:
        keep = np.linspace(0, len(approach) - 1, MAX_APPROACH).round().astype(int)
        approach = [approach[i] for i in keep]
    return {"creep": creep, "approach": approach}


def guard(rows: list[dict]) -> dict:
    """How far the grading here agrees with what labelling recorded, on the released model's rows."""
    pairs = same_ticks = hits_match = 0
    agree = 0
    for r in rows:
        recomputed, recorded = r["recomputed_clear"], r["recorded_clear"]
        # by index: the training interpreter is 3.9, where zip has no strict=
        for i in range(len(recomputed)):
            agree += recomputed[i] == recorded[i]
            pairs += 1
        same_ticks += list(recomputed) == list(recorded)
        hits_match += (r["pick_hit_at"] is None) == (r["recorded_driven_hits"] is None)
    n = max(1, len(rows))
    return {
        "verdicts_agree": agree / max(1, pairs),
        "ticks_identical": same_ticks / n,
        "pick_hits_iff_driven_hit": hits_match / n,
    }


def measure(model, dataset, ticks: list[dict], torch, drivable: Path | None, batch: int = 8) -> list[dict]:
    """One row per rebuildable tick: what the model picked and what that pick does."""
    rows, contexts = [], {}
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
            final = output["pred_logit"][..., -1].float().cpu().numpy()
            # `used`, not `chunk`: batch skips a tick it cannot rebuild, and pairing
            # by position would then grade the wrong tick.
            for offset, tick in enumerate(used):
                directory = Path(tick["frame"]).parent.parent
                key = str(directory)
                # Only what is constant across an episode is cached. The pose is not:
                # caching it per episode is the v1-v4 trainer bug (14-16 m off).
                if key not in contexts:
                    _, targets_, area_ = episode_context(tick, drivable)
                    actors = json.loads((directory / "actors.json").read_text())
                    contexts[key] = (
                        targets_,
                        area_,
                        load_references(directory),
                        json.loads((directory / "ego_poses.json").read_text()),
                        [a for a in actors if not a["is_target_actor"]],
                    )
                targets, area, (logged, lanes), poses, others = contexts[key]
                stamp = int(tick["timestamp"])
                reference = reference_at(stamp, logged, lanes)[0]
                ego2world = np.asarray(poses[str(stamp)])
                props = proposals[offset]
                verdicts = [grade(p, ego2world, stamp, targets, reference, area) for p in props]
                pick = int(np.argmin(np.abs(props - selected[offset]).reshape(len(props), -1).sum(1)))
                order = np.argsort(-final[offset])  # rank 1 = what the scorer likes best
                rank = {int(i): r + 1 for r, i in enumerate(order)}
                clear = [i for i, v in enumerate(verdicts) if v.clear]
                stops = [i for i in clear if end_speed(props[i]) <= REST_MPS]
                rows.append(
                    {
                        "episode": tick["episode"],
                        "timestamp": stamp,
                        "speed": float(tick["speed"]),
                        "pick_is_argmax": pick == int(order[0]),
                        "pick_clear": verdicts[pick].clear,
                        "pick_hit_at": verdicts[pick].hit_at,
                        "pick_hits_other_at": first_hit(props[pick], ego2world, stamp, others),
                        "pick_off_path_at": verdicts[pick].off_path_at,
                        "pick_off_road_at": verdicts[pick].off_road_at,
                        "pick_max_deviation": float(verdicts[pick].max_deviation),
                        "pick_end_lateral": float(props[pick][-1, 1]),
                        "pick_end_speed": end_speed(props[pick]),
                        "pick_travel": length(props[pick]),
                        "expert_travel": length(tick["expert"]),
                        "expert_end_speed": end_speed(tick["expert"]),
                        "n_clear": len(clear),
                        "n_clear_stops": len(stops),
                        "best_clear_rank": min((rank[i] for i in clear), default=None),
                        "best_stop_rank": min((rank[i] for i in stops), default=None),
                        "recomputed_clear": [v.clear for v in verdicts],
                        "recorded_clear": tick["candidate_clear"],
                        "recorded_driven_hits": tick["driven_hits"],
                    }
                )
    return rows


def summary(rows: list[dict]) -> dict:
    """The table's numbers for one model and group."""
    if not rows:
        return {"n": 0}

    def median(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.median(values)) if values else float("nan")

    def share(predicate):
        return float(np.mean([predicate(r) for r in rows]))

    return {
        "n": len(rows),
        "safe": share(lambda r: r["pick_clear"]),
        "hit": share(lambda r: r["pick_hit_at"] is not None),
        "off_road": share(lambda r: r["pick_hit_at"] is None and r["pick_off_road_at"] is not None),
        "off_lane": share(
            lambda r: (
                r["pick_hit_at"] is None
                and r["pick_off_road_at"] is None
                and r["pick_off_path_at"] is not None
            )
        ),
        "hits_other_car": share(lambda r: r["pick_hits_other_at"] is not None),
        "rests": share(lambda r: r["pick_end_speed"] <= REST_MPS),
        "end_speed": median("pick_end_speed"),
        "travel": median("pick_travel"),
        "expert_travel": median("expert_travel"),
        "safe_option_exists": share(lambda r: r["n_clear"] > 0),
        "best_safe_rank": median("best_clear_rank"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labelled", type=Path, required=True, help="the label set the ticks come from")
    parser.add_argument("--drivable", type=Path, required=True, help="directory of scene-*.npz rasters")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", default="0053", help="0053 is held out of training")
    parser.add_argument("--shape", default="stationary", help="stationary, frontal or side")
    parser.add_argument(
        "--kind", action="append", help="expert kinds for the approach group (default: brake)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=MODEL_PT",
        help="a finetune to measure beside the released model; repeatable",
    )
    args = parser.parse_args()

    cfg = SiteConfig.load()
    for name, template in (
        ("NAVSIM_DEVKIT_ROOT", "{data.drivejepa_repo}"),
        ("NAVSIM_EXP_ROOT", "{data.navsim.exp}"),
        ("OPENSCENE_DATA_ROOT", "{data.navsim}"),
        ("NUPLAN_MAPS_ROOT", "{data.navsim}/maps"),
    ):
        os.environ.setdefault(name, cfg.resolve(template))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")

    # blind=True: the off-camera filter is about imitation targets; a car on the path is in frame anyway.
    sets = groups(
        load_ticks(args.labelled, blind=True), args.scene, args.shape, tuple(args.kind or ["brake"])
    )
    for name, ticks in sets.items():
        print(f"  {name}: {len(ticks)} ticks from {len({t['episode'] for t in ticks})} episodes")

    released = cfg.resolve(
        "{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
    )
    _agent, model, torch = build_model(released, args.device, cfg.resolve("{data.drivejepa_repo}"))

    results: dict = {}
    for name, path in [("released", None), *(c.split("=", 1) for c in args.checkpoint)]:
        if path:
            model.load_state_dict(torch.load(path, map_location=args.device, weights_only=True))
        results[name] = {
            g: measure(model, TickDataset(ts, args.device), ts, torch, args.drivable)
            for g, ts in sets.items()
        }
        args.out.write_text(json.dumps(results, indent=1))  # after every model: nothing lost

    check = guard([r for rows in results["released"].values() for r in rows])
    print(
        f"\n  GUARD released vs labelling: candidate verdicts agree {check['verdicts_agree']:.1%}, "
        f"ticks identical {check['ticks_identical']:.1%}, "
        f"picked plan hits iff driven plan hit {check['pick_hits_iff_driven_hit']:.1%}"
    )
    if check["verdicts_agree"] < GUARD_MIN_AGREEMENT:
        failed = args.out.with_suffix(args.out.suffix + ".guard-failed")
        args.out.rename(failed)
        raise SystemExit(
            f"  GUARD FAILED: candidate verdicts agree {check['verdicts_agree']:.2%} "
            f"< {GUARD_MIN_AGREEMENT:.1%} with the labels; results moved to {failed} and not reported"
        )

    columns = (
        "n",
        "safe",
        "hit",
        "off_road",
        "off_lane",
        "hits_other_car",
        "end_speed",
        "travel",
        "expert_travel",
    )
    print("\n  " + f"{'model':<10}{'group':<10}" + "".join(f"{c:>15}" for c in columns))
    for name, by_group in results.items():
        for group, rows in by_group.items():
            s = summary(rows)
            cells = "".join(
                f"{s.get(c, float('nan')):>15.0%}"
                if c in ("safe", "hit", "off_road", "off_lane", "hits_other_car")
                else f"{s.get(c, float('nan')):>15.1f}"
                if c != "n"
                else f"{s['n']:>15}"
                for c in columns
            )
            print(f"  {name:<10}{group:<10}{cells}")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
