"""Before the first step of an aux run: is the code committed, and does the trainer compute what
was recorded?

`gallops.train.preflight` is the swerve trainer's, and its lesson carries over whole. v1-v4 trained
on targets graded from the wrong pose, and every check that existed passed, because each tested a
part that was right on its own. What catches that kind of defect is comparing what the TRAINER
computes with what was RECORDED, through the trainer's own code. An aux run reads three recordings
-- the encoder's tokens, the release's answer and the labels -- and each is held to the path that
would reproduce it:

    code      src/ and tests/ committed (the sha every sidecar of the run carries), and the suite
              passing in this interpreter: swerve's `code_checks`, imported
    data      held-out and train share no scene; every scene is a devkit train scene, none is in
              `val` or behind neuroNCAP; at most MAX_UNREBUILDABLE of the rows lack tokens
    inputs    a sample of train keyframes rebuilt live from nuScenes, through the input builder and
              the vendor's own forward at batch 1: tokens, l2i, img_shape, ego status, the human's
              plan and the release's 32 proposals equal the cache's (measured: exactly)
    trainer   the trainer's path -- `cache.batch`, then `step.cached_forward` over the sample as ONE
              batch with the batch fix -- reproduces the live batch-1 proposals within
              PROPOSAL_TOLERANCE_M (measured 5.3e-5 m at batch 32) and the pick on PICK_AGREEMENT of
              rows (measured 76 of 76); and on that batch the planning loss equals the agent's own
              `trajectory_loss_anchors` with the batch's teachers as its anchors
    labels    the labeller, re-run on the sample, gives what `cache.batch` feeds the heads, exactly;
              boxes in the ego frame equal the devkit's own transform; every box the labels call
              seen projects into the model's crop of the front image by the devkit's projection;
              and over the train split the human's path lies on the rasterised road for at least
              EXPERT_ON_ROAD of its waypoints in view
    heads     upstream's heads as the trainer runs them equal `Scorer.forward` in training mode
              (`step.check_heads`, measured 0.0)
    sight     blanking the front camera moves the release's plan: swerve's `sight_check`, imported

The sample is spread across scenes and visited LATE FIRST, with each sampled scene's earliest
keyframe added: a per-scene cache filled by a late keyframe is then read back for an early one,
and the first keyframe of a scene is where a missing history frame would show.

**Not checked: that these inputs equal paddock's.** The input builder is swerve_data's TickDataset,
which swerve's fidelity check holds to paddock's adapter on neuroNCAP dumps; the feed from the
nuScenes tables (`inputs.TablesTickDataset`) is new and has not been compared with paddock's
nuScenes harness, because paddock's candidate dump takes `--rows` only for KITScenes. The report
says so (`paddock_parity`), and no checkpoint of these runs is a ship candidate until it is.
"""

from __future__ import annotations

import json
import math
import random
from functools import partial

import numpy as np

#: Live tokens may differ from cached ones by this much: a different kernel choice in a new process
#: moves the last bits, a wrong frame or calibration moves them by O(1). Measured: 0.
TOKEN_TOLERANCE = 1e-4
#: Proposals, live against cached (batch 1), and the trainer's batched path against the live record.
#: Measured 0 and 5.3e-5 m; a wrong history frame or a batch-mixing refiner is metres.
PROPOSAL_TOLERANCE_M = 1e-3
#: Rows whose pick may flip between the two paths: a near-tie, nothing else (measured 0 of 76).
PICK_AGREEMENT = 0.95
#: The planning loss against the agent's own, relative (`train_generator`'s first-batch check).
LOSS_PARITY_RTOL = 1e-4
#: The labeller re-run against the stored labels: equal, up to float32 storage.
LABEL_TOLERANCE = 1e-5
#: Boxes against the devkit's transform: position in metres, heading in radians. nusc takes the
#: heading from the 2D yaws; the devkit from the full rotation, which the ego's small pitch and
#: roll move by a few thousandths.
DEVKIT_POSITION_M = 1e-4
DEVKIT_HEADING_RAD = 1e-2
#: The human's logged path in view must lie on the rasterised road for this share of its waypoints.
#: Measured 436 of 436 on the first 3 scenes; below it the map or the poses are misplaced.
EXPERT_ON_ROAD = 0.99
#: `step.check_heads`, measured 0.0.
HEADS_TOLERANCE = 1e-6
PADDOCK_PARITY = (
    "not checked: paddock's candidate dump (tools/models/drivejepa/paddock_candidates.py) takes --rows "
    "for KITScenes only, so the nuScenes-table input feed is held to this repo's builder, not paddock's"
)


def sample_rows(scenes, keyframes, pool, n_scenes: int = 12, per_scene: int = 4, seed: int = 0) -> list[int]:
    """Rows of `pool` across `n_scenes` scenes, `per_scene` of each at fixed fractions of its length,
    LATE FIRST, then each sampled scene's earliest row (swerve's `sample_ticks` and
    `earliest_per_episode`)."""
    by_scene: dict = {}
    for i in pool:
        by_scene.setdefault(str(scenes[i]), []).append(int(i))
    names = sorted(by_scene)
    random.Random(seed).shuffle(names)
    chosen, earliest = [], []
    for name in names[:n_scenes]:
        rows = sorted(by_scene[name], key=lambda i: int(keyframes[i]))
        picks = sorted({round(f * (len(rows) - 1)) for f in np.linspace(0.1, 0.95, per_scene)}, reverse=True)
        chosen.extend(rows[j] for j in picks)
        if rows[0] not in chosen:
            earliest.append(rows[0])
    return chosen + earliest


def _max(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both = np.isnan(a) & np.isnan(b)
    gap = np.abs(np.where(both, 0.0, a) - np.where(both, 0.0, b))
    return float(np.nanmax(np.where(np.isnan(gap), np.inf, gap))) if gap.size else 0.0


def data_checks(data: dict, train_rows, held_rows) -> tuple[dict, list[str]]:
    from nuscenes.utils import splits

    from gallops.drivejepa_aux import nusc
    from gallops.train.preflight import MAX_UNREBUILDABLE

    scenes, built = data["labels"]["scene"], data["inputs"]["built"]
    train, held = set(scenes[train_rows]), set(scenes[held_rows])
    problems = []
    if train & held:
        problems.append(
            f"data: {len(train & held)} scenes both train and are held out, e.g. {sorted(train & held)[:3]}"
        )
    outside = (train | held) - set(nusc.train_scene_names())
    if outside:
        problems.append(
            f"data: {len(outside)} scenes are not trainable train scenes, e.g. {sorted(outside)[:3]}"
        )
    if (train | held) & set(splits.val):
        problems.append("data: a nuScenes val scene is in the cache")
    unbuilt = int((~built).sum())
    if unbuilt > MAX_UNREBUILDABLE * len(built):
        problems.append(
            f"data: {unbuilt} of {len(built)} rows have no tokens (limit {MAX_UNREBUILDABLE:.0%})"
        )
    report = {
        "train_scenes": len(train),
        "held_out_scenes": len(held),
        "train_rows": int(len(train_rows)),
        "held_out_rows": int(len(held_rows)),
        "rows_without_tokens": unbuilt,
    }
    return report, problems


def live_checks(model, data: dict, rows: list, ticks: list, dataset, device: str) -> tuple[dict, list[str]]:
    """The cache against a live rebuild through the vendor's forward at batch 1."""
    import torch

    from gallops.drivejepa_aux import step

    inputs, worst, picks, compared = data["inputs"], {}, 0, 0
    problems = []
    for j, row in enumerate(rows):
        features, target, _ = dataset.batch([j])
        if features is None:
            problems.append(
                f"inputs: row {row} ({ticks[j]['scene']} keyframe {ticks[j]['keyframe']}) does not rebuild"
            )
            continue
        with torch.no_grad():
            result = step.forward(model, None, features, keep=True)
        out = result["out"]
        gaps = {
            "tokens": _max(result["tokens"][0, :, 0].float().cpu().numpy(), data["tokens"][row]),
            "l2i": _max(features["lidar2img"][0, 1:2].cpu().numpy(), inputs["l2i"][row]),
            "img_shape": _max(features["img_shape"][0].cpu().numpy(), inputs["img_shape"][row]),
            "ego": _max(features["ego_status"][0, -1].cpu().numpy(), inputs["ego"][row]),
            "target": _max(target[0].cpu().numpy(), inputs["target"][row]),
            "proposals_m": _max(out["proposals"][0].float().cpu().numpy(), inputs["proposals"][row]),
        }
        for key, gap in gaps.items():
            worst[key] = max(worst.get(key, 0.0), gap)
        picks += int(int(out["pdm_score"][0].argmax()) == int(inputs["score"][row].argmax()))
        compared += 1
    for key, limit in (("tokens", TOKEN_TOLERANCE), ("proposals_m", PROPOSAL_TOLERANCE_M)):
        if worst.get(key, 0.0) > limit:
            problems.append(
                f"inputs: live {key} differ from the cache by up to {worst[key]:.2e} (limit {limit})"
            )
    for key in ("l2i", "img_shape", "ego", "target"):
        if worst.get(key, 0.0) > LABEL_TOLERANCE:
            problems.append(f"inputs: live {key} differs from the cache by up to {worst[key]:.2e}")
    if compared and picks < PICK_AGREEMENT * compared:
        problems.append(f"inputs: the live pick matches the cache's on {picks} of {compared} rows")
    report = {"compared": compared, "picks_agree": picks, **{f"max_{k}": float(v) for k, v in worst.items()}}
    return report, problems


def trainer_checks(model, agent, config, data: dict, rows: list, spatial, level, device: str):
    """The trainer's batched path against the live record, and its loss against the agent's own."""
    import torch

    from gallops.drivejepa_aux import cache, step
    from gallops.drivejepa_b2d.train_generator import mtd_loss

    inputs = data["inputs"]
    fed = cache.batch(data, rows, device, with_labels=False)
    with torch.no_grad():
        result = step.cached_forward(model, None, fed["inputs"], spatial, level, score=True)
    out = result["out"]
    gap = _max(out["proposals"].float().cpu().numpy(), inputs["proposals"][rows])
    picks = int((out["pdm_score"].argmax(1).cpu().numpy() == inputs["score"][rows].argmax(1)).sum())
    problems = []
    if gap > PROPOSAL_TOLERANCE_M:
        problems.append(
            f"trainer: the batched cached path is {gap:.2e} m off the live record "
            f"(limit {PROPOSAL_TOLERANCE_M})"
        )
    if picks < PICK_AGREEMENT * len(rows):
        problems.append(
            f"trainer: the batched cached path picks the recorded plan on {picks} of {len(rows)} rows"
        )

    # train_generator's first-batch check: the batch's teachers as the agent's anchors, so its own
    # trajectory_loss_anchors computes winner-takes-all + 0.5 x the teacher term over the 4 passes
    teachers = inputs["teachers"][rows]
    k = teachers.shape[1]
    with torch.no_grad():
        ours = float(
            mtd_loss(
                out["proposal_list"], fed["target"], fed["teachers"], fed["n_teachers"], config.prev_weight
            )
        )
        saved = agent.anchors
        try:
            agent.anchors = teachers.reshape(-1, *teachers.shape[2:])
            index = [np.arange(i * k, (i + 1) * k) for i in range(len(rows))]
            theirs = float(
                agent.trajectory_loss_anchors(out["proposal_list"], fed["target"], config, index)[0]
            )
        finally:
            agent.anchors = saved
    if abs(ours - theirs) > LOSS_PARITY_RTOL * max(1.0, abs(theirs)):
        problems.append(
            f"trainer: planning loss {ours:.6f} != the agent's trajectory_loss_anchors {theirs:.6f}"
        )
    report = {
        "batch": len(rows),
        "max_proposal_gap_m": gap,
        "picks_agree": picks,
        "planning_loss": ours,
        "agent_trajectory_loss_anchors": theirs,
    }
    return report, problems


def label_checks(data: dict, rows: list, ticks: list, extract: dict, root, train_rows, device: str):
    """The labeller re-run against what the trainer feeds; the boxes against the devkit."""
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import view_points
    from pyquaternion import Quaternion

    from gallops.drivejepa_aux import cache, heads, nusc

    maps: dict = {}
    records, problems = [], []
    position = heading = 0.0
    projected = outside = 0
    for tick in ticks:
        scene = extract["scenes"][tick["scene"]]
        if scene["location"] not in maps:
            maps[scene["location"]] = nusc.MapView(nusc.devkit_map(root, scene["location"]))
        records.append(cache.keyframe_record(scene, tick, extract["annotations"], maps[scene["location"]]))

        # boxes: nusc's ego-frame transform against the devkit's Box.translate / rotate
        ego2global = np.asarray(scene["keyframes"][tick["keyframe"]]["ego"])
        chosen = [
            a
            for a in extract["annotations"].get(tick["sample"], [])
            if a["category"].startswith(nusc.VEHICLE)
        ]
        ours, centres, _ = nusc._boxes(chosen, ego2global, nusc.VEHICLE)
        ego_rotation = Quaternion(matrix=ego2global[:3, :3])
        for annotation, box in zip(chosen, ours):  # noqa: B905 -- Python 3.9
            devkit = Box(
                annotation["translation"], annotation["size_wlh"], Quaternion(annotation["rotation"])
            )
            devkit.translate(-ego2global[:3, 3])
            devkit.rotate(ego_rotation.inverse)
            position = max(position, float(np.abs(devkit.center[:2] - box[:2]).max()))
            turn = (devkit.orientation.yaw_pitch_roll[0] - box[2] + math.pi) % (2 * math.pi) - math.pi
            heading = max(heading, abs(turn))

        # every box called seen projects into the model's crop, by the devkit's projection
        camera = scene["calibration"]["CAM_FRONT"]
        intrinsics, cam2ego = np.asarray(camera["intrinsics"]), np.asarray(camera["pose"])
        width, height = scene["image_size"]
        labels = nusc.keyframe_labels(
            scene, tick, extract["annotations"].get(tick["sample"], []), maps[scene["location"]]
        )
        for centre, seen in zip(centres, labels["vehicles_seen"]):  # noqa: B905 -- Python 3.9
            if not seen:
                continue
            in_camera = (np.linalg.inv(cam2ego) @ np.r_[centre, 1.0])[:3]
            u, v, _ = view_points(in_camera[:, None], intrinsics, normalize=True)[:, 0]
            inside = (
                in_camera[2] > 0
                and 0 <= u < width
                and nusc.MODEL_CROP_ROWS <= v < height - nusc.MODEL_CROP_ROWS
            )
            projected += 1
            outside += int(not inside)

    # the labeller's output, collated as the heads' losses take it, against what the trainer feeds
    relabelled = heads.collate(records, device)
    fed = cache.batch(data, rows, device)["labels"]
    differs = []
    for key, value in relabelled.items():
        gap = _max(value.float().cpu().numpy(), fed[key].float().cpu().numpy())
        if gap > LABEL_TOLERANCE:
            differs.append(f"{key} ({gap:.2e})")
    stored = data["labels"]
    road_gap = int((cache.road(stored, rows) != np.stack([r["road"] for r in records])).sum())
    box_gap = 0.0
    for record, row in zip(records, rows):  # noqa: B905 -- Python 3.9
        relabelled_boxes = np.concatenate(record["future_boxes"])
        stored_boxes = np.concatenate(cache.future_boxes(stored, row))
        same = relabelled_boxes.shape == stored_boxes.shape
        box_gap = max(box_gap, _max(relabelled_boxes, stored_boxes) if same else math.inf)
    if differs:
        problems.append(
            f"labels: the labeller re-run differs from what the trainer feeds: {', '.join(differs)}"
        )
    if road_gap or box_gap > LABEL_TOLERANCE:
        problems.append(f"labels: re-run road differs on {road_gap} cells, future boxes by {box_gap:.2e}")
    if position > DEVKIT_POSITION_M or heading > DEVKIT_HEADING_RAD:
        problems.append(
            f"labels: boxes differ from the devkit's transform by {position:.2e} m, {heading:.2e} rad"
        )
    if outside:
        problems.append(
            f"labels: {outside} of {projected} boxes called seen project outside the model's crop"
        )
    judged = int(stored["expert_judged"][train_rows].sum())
    on = int(stored["expert_on_road"][train_rows].sum())
    if judged and on < EXPERT_ON_ROAD * judged:
        problems.append(
            f"labels: the human's path in view is on the rasterised road for {on} of {judged} waypoints"
        )
    report = {
        "relabelled_rows": len(rows),
        "devkit_max_position_m": position,
        "devkit_max_heading_rad": heading,
        "seen_boxes_projected": projected,
        "seen_boxes_outside_crop": outside,
        "expert_waypoints_on_road": f"{on}/{judged}",
    }
    return report, problems


def run(
    repo,
    model,
    scene_heads,
    agent,
    config,
    data: dict,
    train_rows,
    held_rows,
    extract: dict,
    root,
    device: str,
):
    """Every check; raises PreflightFailed listing every problem, or returns the report."""
    import torch

    from gallops.drivejepa_aux import inputs, step
    from gallops.train import preflight as swerve_preflight

    report, problems = {}, []
    report["code"], found = swerve_preflight.code_checks(repo)
    problems += found
    report["data"], found = data_checks(data, train_rows, held_rows)
    problems += found

    labels = data["labels"]
    built = train_rows[data["inputs"]["built"][train_rows]]
    rows = sample_rows(labels["scene"], labels["keyframe"], built)
    by_key = {(t["scene"], t["keyframe"]): t for name in extract["ticks"] for t in extract["ticks"][name]}
    ticks = [by_key[(str(labels["scene"][r]), int(labels["keyframe"][r]))] for r in rows]
    if [t["sample"] for t in ticks] != [str(s) for s in labels["sample"][rows]]:
        problems.append("data: the cache's rows do not name the extract's keyframes")
    dataset_cls = partial(inputs.TablesTickDataset, scenes=extract["scenes"], nuscenes_root=str(root))

    report["inputs"], found = live_checks(model, data, rows, ticks, dataset_cls(ticks, device), device)
    problems += found
    spatial = torch.as_tensor(data["inputs"]["spatial"], device=device)
    level = torch.as_tensor(data["inputs"]["level"], device=device)
    report["trainer"], found = trainer_checks(model, agent, config, data, rows, spatial, level, device)
    problems += found
    report["labels"], found = label_checks(data, rows, ticks, extract, root, train_rows, device)
    problems += found
    if getattr(model.scorer, "bev_map", False):
        features, _, _ = dataset_cls(ticks[:1], device).batch([0])
        with torch.no_grad():
            gap = step.check_heads(model, step.forward(model, scene_heads, features))
        report["heads"] = {"max_difference": gap}
        if gap > HEADS_TOLERANCE:
            problems.append(f"heads: upstream's heads as trained differ from Scorer.forward by {gap:.2e}")
    report["sight"], found = swerve_preflight.sight_check(model, dataset_cls, ticks, torch, device)
    problems += found
    report["paddock_parity"] = PADDOCK_PARITY
    print("  preflight: " + json.dumps(report), flush=True)
    if problems:
        raise swerve_preflight.PreflightFailed(problems)
    return report
