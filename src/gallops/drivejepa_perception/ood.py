"""OOD (PREDICTIONS.md P15): the side-car and the B1 scorers on nuScenes, which nothing here saw.

    python -m gallops.drivejepa_perception.nuscenes_data draw --n N --seed S       # once, committed
    python -m gallops.drivejepa_perception.ood extract --out {out}/nusc-extract          # CPU, once
    python -m gallops.drivejepa_perception.ood pass --extract {out}/nusc-extract --out {out}/nusc-v1
    python -m gallops.drivejepa_perception.ood read --nusc {out}/nusc-v1 --sidecar {out}/runs/a1-s0 \
        --b1 {out}/b1 --out {out}/ood.json
    python -m gallops.drivejepa_perception.ood read-b2 --nusc {out}/nusc-v1 --sidecar {out}/runs/a1-s0 \
        --b2 {out}/b2 --out {out}/ood-b2.json

**Rows.** Every usable keyframe (`nuscenes_data.ticks`: the keyframe 0.5 s back and 4 s of future
keyframes exist) of H15's own seeded draw of nuScenes train scenes (`nuscenes_data.OOD_SCENES`).
The release trained on NAVSIM and H15 on navtrain, so every number here is out of sample for all
of them: the comparison with the release that navtrain cannot give fairly.

**Inputs** are swerve_data's TickDataset fed from the extract (`nuscenes_data.keyframe_dataset`:
the vendor's feature builder, nuScenes' cameras on the ego frame, the 28-row crop). The release
runs at batch 1 in eval mode, so no batch_fix is involved, and the pass keeps what H15's readers
read: the ViT's output (the side-car's tap), the scorer's pooled input, the 32 proposals, the 6
logits and the pick, and lidar2img, img_shape and the ego status.

**Labels**, in H15's layout (`navsim_data.pack`), from the annotations at the keyframe:
- vehicles, pedestrians and two-wheelers (`class_of`), each with at least one lidar point:
  nuScenes' detection convention, since an annotation no sensor saw is no target for a camera;
- boxes in the keyframe's ego frame and ground velocity in its axes (`nuscenes_data.boxes`);
- each agent's path at +0.5 s and +1.0 s: its instance at the next two keyframes, which are 0.5 s
  apart, in this keyframe's frame.
Every annotated box at the next 8 keyframes (`nuscenes_data.future_boxes`) gives the collision
rate (`nuscenes_data.collided`: the ego's footprint at a waypoint overlaps a box, within 3 s).

**The projection.** The vendor's lidar2img maps ego points to 0.4x the camera's pixels; the tokens
see the image after a 28-row crop, resized to 512x256. `targets.to_512(width, height)` composes
the two: at NAVSIM's 1920x1080 it is TO_512, at nuScenes' 1600x900 it is not. Before writing a
row, the pass holds `to_512(1600, 900) @ lidar2img` to the projection made directly from each
scene's calibration (intrinsics, camera pose, crop, resize) on a grid of points ahead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gallops.drivejepa_perception import navsim_data
from gallops.drivejepa_perception import nuscenes_data as nd
from gallops.drivejepa_perception import targets as tg

#: nuScenes category prefix -> H15's class (navsim_data.CLASSES). Two-wheelers are the cyclist class,
#: as NAVSIM's "bicycle"; static objects are kept for analysis, as in the navtrain labels.
CATEGORY_CLASS = (
    ("vehicle.bicycle", 2),
    ("vehicle.motorcycle", 2),
    ("vehicle.", 0),
    ("human.pedestrian.", 1),
    ("movable_object.barrier", 3),
    ("movable_object.trafficcone", 3),
)
MIN_LIDAR_POINTS = 1
PATH_KEYFRAMES = (1, 2)  # +0.5 s and +1.0 s: nuScenes keyframes are 0.5 s apart
#: The largest gap, in token-image pixels, allowed between the two projections on any scene.
PROJECTION_LIMIT_PX = 0.5
NUSC_SIZE = (1600, 900)


def class_of(category: str) -> int:
    """H15's class of a nuScenes category, or -1 for one it does not keep."""
    for prefix, cls in CATEGORY_CLASS:
        if category.startswith(prefix):
            return cls
    return -1


def _to_frame(ego2global: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N, 3) global points into the frame of `ego2global`."""
    inverse = np.linalg.inv(ego2global)
    return (np.c_[points, np.ones(len(points))] @ inverse.T)[:, :3]


def keyframe_agents(scene: dict, tick: dict, annotations: dict) -> dict:
    """One keyframe's agents in H15's label layout (navsim_data.AGENT_KEYS)."""
    frames = scene["keyframes"]
    k = tick["keyframe"]
    ego2global = np.asarray(frames[k]["ego"])
    chosen = [
        a
        for a in annotations.get(frames[k]["sample"], [])
        if class_of(a["category"]) >= 0 and a["lidar_points"] >= MIN_LIDAR_POINTS
    ]
    n = len(chosen)
    out = {
        "cls": np.array([class_of(a["category"]) for a in chosen], np.int8),
        "name": np.full(n, -1, np.int8),
        "box": np.zeros((n, 7), np.float32),
        "velocity": np.zeros((n, 2), np.float32),
        "path": np.zeros((n, 2, 2), np.float32),
        "path_ok": np.zeros((n, 2), bool),
        "size_change": np.zeros((n, 2), np.float32),
        "track": np.arange(n, dtype=np.int64),
    }
    if not n:
        return out
    b = nd.boxes(chosen, ego2global)
    # box: x, y, z, length, width, height, yaw (navsim_data's order)
    out["box"] = np.stack(
        [b["xyz"][:, 0], b["xyz"][:, 1], b["xyz"][:, 2], b["length"], b["width"], b["height"], b["yaw"]],
        axis=1,
    ).astype(np.float32)
    out["velocity"] = np.nan_to_num(b["vel"]).astype(np.float32)
    for j, step in enumerate(PATH_KEYFRAMES):
        later = {a["instance"]: a for a in annotations.get(frames[k + step]["sample"], [])}
        for i, a in enumerate(chosen):
            match = later.get(a["instance"])
            if match is not None:
                out["path"][i, j] = _to_frame(ego2global, np.asarray([match["translation"]]))[0, :2]
                out["path_ok"][i, j] = True
    return out


def to_512_direct(calibration: dict, width: int, height: int) -> np.ndarray:
    """Ego -> token-image projection straight from a scene's calibration: K, the camera's pose on
    the ego frame, the 28-row crop and the resize to 512x256. The route the pass holds the vendor's to."""
    camera = calibration["CAM_FRONT"]
    k = np.eye(4)
    k[:3, :3] = np.asarray(camera["intrinsics"])
    ego2cam = np.linalg.inv(np.asarray(camera["pose"]))
    crop = 28
    resize = np.diag([512 / width, 256 / (height - 2 * crop), 1.0, 1.0])
    resize[1, 2] = -crop * 256 / (height - 2 * crop)
    return resize @ k @ ego2cam


def projection_gap(a: np.ndarray, b: np.ndarray) -> float:
    """The largest pixel gap between two ego -> image projections over a grid of points ahead."""
    xs, ys = np.meshgrid(np.linspace(5, 50, 10), np.linspace(-10, 10, 9))
    points = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 0.5)]
    pixels = []
    for p in (a, b):
        h = np.c_[points, np.ones(len(points))] @ np.asarray(p, dtype=np.float64).T
        pixels.append(h[:, :2] / h[:, 2:3])
    inside = tg.seen(a, points) | tg.seen(b, points)
    return float(np.abs(pixels[0] - pixels[1])[inside].max()) if inside.any() else float("nan")


def ood_ticks(extract: dict) -> list:
    """Every usable keyframe of H15's scenes, scenes sorted, keyframes in order."""
    return [t for name in sorted(nd.load_scenes()) for t in extract["ticks"].get(name, [])]


def run_extract(cfg, out: Path) -> None:
    """H15's scenes from the nuScenes tables, with their calibration in swerve_data's layout."""
    import pickle

    out = Path(out)
    if (out / "extract.pkl").exists():
        raise SystemExit(f"{out}/extract.pkl exists; an extract is written once")
    root = Path(cfg.resolve("{data.nuscenes}"))
    names = nd.load_scenes()
    started = time.time()
    data = nd.extract(root, names)
    data["ticks"] = {name: nd.ticks(name, scene, root / "can_bus") for name, scene in data["scenes"].items()}
    out.mkdir(parents=True, exist_ok=True)
    nd.write_calibration(data["scenes"], out / "calib")
    with open(out / "extract.pkl", "wb") as fh:
        pickle.dump(data, fh, protocol=4)
    counts = {"scenes": len(names), "keyframes": sum(len(v) for v in data["ticks"].values())}
    print(json.dumps({**counts, "seconds": round(time.time() - started)}))


def run_pass(cfg, extract_dir: Path, out: Path, workers: int) -> None:
    """The release at batch 1 over every keyframe of H15's scenes, and the keyframes' labels."""
    import pickle
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started, sha = time.time(), git_sha()
    entered = env.enter(cfg)
    with open(Path(extract_dir) / "extract.pkl", "rb") as fh:
        data = pickle.load(fh)
    root = Path(cfg.resolve("{data.nuscenes}"))
    ticks = ood_ticks(data)
    n = len(ticks)
    first: dict = {}
    for i, t in enumerate(ticks):
        first.setdefault(t["scene"], i)

    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.cuda().eval()
    captured: dict = {}
    model._backbone.register_forward_hook(lambda m, i, o: captured.__setitem__("tokens", o[0]))
    model._backbone.img_backbone.register_forward_hook(lambda m, i, o: captured.__setitem__("vit_out", o))
    model.scorer.pred_score.register_forward_pre_hook(lambda m, i: captured.__setitem__("pooled", i[0]))

    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    vit = np.lib.format.open_memmap(
        staging / "vit_out.npy", mode="w+", dtype=np.float16, shape=(n, 512, 1024)
    )
    arrays = {
        "built": np.zeros(n, bool),
        "tokens": np.zeros((n, 512, 256), np.float16),
        "pooled": np.zeros((n, 32, 256), np.float16),
        "proposals": np.zeros((n, 32, 8, 3), np.float32),
        "logits": np.zeros((n, 32, 6), np.float32),
        "pick": np.zeros(n, np.int64),
        "lidar2img": np.zeros((n, 4, 4, 4), np.float32),
        "img_shape": np.zeros((n, 4, 3), np.float32),
        "ego_status": np.zeros((n, 11), np.float32),
    }
    projection_gaps = {}

    def build_scene(name: str) -> list:
        dataset = nd.keyframe_dataset(
            data["ticks"][name], data["scenes"], str(root), Path(extract_dir) / "calib"
        )
        return [dataset.batch([j])[0] for j in range(len(dataset))]

    names = sorted({t["scene"] for t in ticks})
    torch.set_num_threads(4)
    done = 0
    with ThreadPoolExecutor(workers) as pool, torch.no_grad():
        queue, pending = deque(), iter(names)
        for name in pending:
            queue.append((name, pool.submit(build_scene, name)))
            if len(queue) >= 2 * workers:
                break
        while queue:
            name, future = queue.popleft()
            built = future.result()
            following = next(pending, None)
            if following is not None:
                queue.append((following, pool.submit(build_scene, following)))
            for j, features in enumerate(built):
                i = first[name] + j
                if features is None:
                    continue
                if name not in projection_gaps:
                    vendor = tg.to_512(*NUSC_SIZE) @ features["lidar2img"][0, tg.FRONT_SLOT].double().numpy()
                    direct = to_512_direct(data["scenes"][name]["calibration"], *NUSC_SIZE)
                    projection_gaps[name] = projection_gap(vendor, direct)
                    if not projection_gaps[name] <= PROJECTION_LIMIT_PX:
                        raise SystemExit(f"{name}: the vendor's projection is {projection_gaps[name]} px off")
                out_ = model({k: v.cuda() if torch.is_tensor(v) else v for k, v in features.items()})
                if tuple(captured["tokens"].shape) != (1, 512, 1, 256):
                    raise SystemExit(f"unexpected token shape {tuple(captured['tokens'].shape)}")
                vit[i] = captured["vit_out"].reshape(512, 1024).half().cpu().numpy()
                arrays["tokens"][i] = captured["tokens"][0, :, 0].half().cpu().numpy()
                arrays["pooled"][i] = captured["pooled"][0].half().cpu().numpy()
                arrays["proposals"][i] = out_["proposals"][0].float().cpu().numpy()
                arrays["logits"][i] = out_["pred_logit"][0].float().cpu().numpy()
                arrays["pick"][i] = int(out_["pdm_score"][0].argmax())
                arrays["lidar2img"][i] = features["lidar2img"][0].numpy()
                arrays["img_shape"][i] = features["img_shape"][0].numpy()
                arrays["ego_status"][i] = features["ego_status"][0, -1].numpy()
                arrays["built"][i] = True
                done += 1
            if done and (first[name] // 500) != ((first[name] + len(built)) // 500):
                rate = done / (time.time() - started)
                print(f"{done}/{n} keyframes, {rate:.1f}/s", flush=True)
    vit.flush()
    del vit
    samples = [keyframe_agents(data["scenes"][t["scene"]], t, data["annotations"]) for t in ticks]
    for s, t in zip(samples, ticks):  # noqa: B905 -- Python 3.9
        s["ego_future"] = np.asarray(t["expert"], np.float32)
        s["ego_speed"] = np.full(5, t["speed"], np.float32)
    labels = navsim_data.pack(samples)
    np.savez(staging / "labels.npz", sample=np.array([t["sample"] for t in ticks]), **labels)
    future = [nd.future_boxes(data["scenes"][t["scene"]], t["keyframe"], data["annotations"]) for t in ticks]
    with open(staging / "future_boxes.pkl", "wb") as fh:
        pickle.dump(future, fh, protocol=4)
    np.savez(staging / "outputs.npz", scene=np.array([t["scene"] for t in ticks]), **arrays)
    manifest = {
        "pass": "nuScenes keyframes of H15's scenes, the release at batch 1 (H15 OOD)",
        "gallops_sha": sha,
        "entered": entered,
        "extract": str(extract_dir),
        "rows": n,
        "built": int(arrays["built"].sum()),
        "scenes": len(names),
        "projection_gap_px_max": max(projection_gaps.values()) if projection_gaps else None,
        "agents": int(labels["count"].sum()),
        "seconds": round(time.time() - started),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    staging.rename(out)
    print(json.dumps(manifest, default=str))


def read(cfg, args) -> None:
    """P15's numbers: the side-car zero-shot, and every B1 run's pick against the release's."""
    import pickle

    import torch

    from gallops.drivejepa_b2d.head import Head, head_tensors_from_checkpoint
    from gallops.drivejepa_perception import b_report, env, scorers, train_scorer, train_sidecar

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a readout never overwrites one")
    entered = env.enter(cfg)
    nusc = Path(args.nusc)
    labels = dict(np.load(nusc / "labels.npz"))
    outputs = dict(np.load(nusc / "outputs.npz"))
    vit = np.load(nusc / "vit_out.npy", mmap_mode="r")
    with open(nusc / "future_boxes.pkl", "rb") as fh:
        future = pickle.load(fh)
    rows = np.flatnonzero(outputs["built"])
    scenes = outputs["scene"]
    front = outputs["lidar2img"][:, tg.FRONT_SLOT].astype(np.float64)
    projections = (tg.to_512(*NUSC_SIZE) @ front).astype(np.float32)
    targets = [tg.build(navsim_data.agents_of(labels, r), projections[r]) for r in range(len(scenes))]
    device = torch.device("cuda")
    result = {"rows": int(len(rows)), "scenes": int(len(set(scenes[rows].tolist())))}

    # The side-car, zero-shot: A1's run at its kept epoch, read as on navtrain (the gap prior is
    # the navtrain train mean that run recorded).
    run = Path(args.sidecar)
    meta = json.loads((run / "run.json").read_text())
    car, kept = train_sidecar.load_run(run)
    car = car.to(device)
    dataset = train_sidecar.Rows(vit, int, rows, projections, targets)
    result["sidecar"] = {
        "run": str(run),
        "kept_epoch": kept,
        "readout": train_sidecar.read(car, dataset, targets, rows, device, meta["prior_gap_m"]),
    }
    agents_f = train_sidecar.agent_arrays(car, vit, int, np.arange(len(scenes)), projections, device)
    print(json.dumps({"sidecar_recall_2m_within_30m": result["sidecar"]["readout"]["recall_2m_within_30m"]}))

    # B1: every run under --b1, every seed at its kept epoch, on the same rows as the release.
    data = {
        "pooled": outputs["pooled"],
        "proposals": outputs["proposals"],
        "logits": outputs["logits"],
        "ego": outputs["ego_status"],
        "shape": train_scorer.shape_of(outputs["proposals"]),
        "pdm": np.zeros((len(scenes), 32, 9), np.float32),  # the forward never reads them
        "ade": np.zeros((len(scenes), 32), np.float32),
    }
    by_arm = {
        "C": None,
        "F": agents_f,
        "O": train_scorer.truth_agents_projected(labels, projections),
        "S": {k: v[train_scorer.shuffled_rows(scenes)] for k, v in agents_f.items()},
    }
    expert = labels["ego_future"]
    boxes = [future[r] for r in rows]

    def plan_numbers(pick):
        plans = outputs["proposals"][rows, pick[rows]]
        gap = np.linalg.norm(plans[:, :, :2] - expert[rows][:, :, :2], axis=-1)
        return {
            "l2_1s": gap[:, 1],
            "l2_2s": gap[:, 3],
            "l2_3s": gap[:, 5],
            "collision_3s": nd.collided(plans, boxes, steps=6).astype(np.float64),
        }

    def summary(numbers):
        return {k: float(v.mean()) for k, v in numbers.items()}

    release = plan_numbers(outputs["pick"])
    result["release"] = summary(release)
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    result["b1"] = {}
    for run_dir in sorted(p for p in Path(args.b1).iterdir() if (p / "summary.json").exists()):
        mode, arm = run_dir.name.rsplit("-", 1)
        run_summary = json.loads((run_dir / "summary.json").read_text())
        if arm != "C":
            data["agents"] = by_arm[arm]
        else:
            data.pop("agents", None)
        seeds = {}
        for key in sorted(k for k in run_summary if k.startswith("seed")):
            kept_epoch = run_summary[key]["kept_epoch"]
            model = scorers.B1Model(mode, arm != "C", head)
            state = torch.load(run_dir / key / f"epoch_{kept_epoch:02d}.pt", map_location="cpu")["state_dict"]
            model.load_state_dict(state, strict=False)
            model = model.to(device)
            pick = np.zeros(len(scenes), np.int64)
            pick[rows] = train_scorer.scores(model, data, rows, device).argmax(1)
            numbers = plan_numbers(pick)
            seeds[key] = {
                "kept_epoch": kept_epoch,
                **summary(numbers),
                "minus_release": {
                    k: {
                        "mean": float((numbers[k] - release[k]).mean()),
                        "se": b_report.clustered_se(numbers[k] - release[k], scenes[rows]),
                    }
                    for k in numbers
                },
                "pick_changed_share": float((pick[rows] != outputs["pick"][rows]).mean()),
            }
            result.setdefault("_picks", {})[f"{run_dir.name}/{key}"] = pick
        result["b1"][run_dir.name] = seeds
        print(json.dumps({run_dir.name: {k: round(v["l2_3s"], 3) for k, v in seeds.items()}}), flush=True)
    picks = result.pop("_picks", {})
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out.with_suffix(".picks.npz"),
        rows=rows,
        release=outputs["pick"],
        **{k.replace("/", "__"): v for k, v in picks.items()},
    )
    out.write_text(json.dumps(result, indent=1, default=str))


def read_b2(cfg, args) -> None:
    """B2's refiners on nuScenes: every run under --b2, every seed at its last epoch, against the
    release on the same keyframes. The refiner runs from the pass's post-projector tokens, the
    frozen release scorer picks, and the plans are read as `read` reads them: the pick's L2 at
    1, 2 and 3 s and its collisions within 3 s, plus the best of 32 by ADE over 3 s (`nd.min_ade`).
    F and S read the side-car's agents on these keyframes (S: another scene's).
    """
    import pickle

    import torch

    from gallops.drivejepa_b2d.head import Head, head_tensors_from_checkpoint
    from gallops.drivejepa_b2d.train_generator import refine_from_cache
    from gallops.drivejepa_perception import b_report, env, train_refiner, train_scorer, train_sidecar

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a readout never overwrites one")
    entered = env.enter(cfg)
    nusc = Path(args.nusc)
    labels = dict(np.load(nusc / "labels.npz"))
    outputs = dict(np.load(nusc / "outputs.npz"))
    vit = np.load(nusc / "vit_out.npy", mmap_mode="r")
    with open(nusc / "future_boxes.pkl", "rb") as fh:
        future = pickle.load(fh)
    rows = np.flatnonzero(outputs["built"])
    scenes = outputs["scene"]
    front = outputs["lidar2img"][:, tg.FRONT_SLOT].astype(np.float64)
    projections = (tg.to_512(*NUSC_SIZE) @ front).astype(np.float32)
    device = torch.device("cuda")
    car, _ = train_sidecar.load_run(Path(args.sidecar))
    agents_f = train_sidecar.agent_arrays(car, vit, int, np.arange(len(scenes)), projections, device)
    by_arm = {"F": agents_f, "S": {k: v[train_scorer.shuffled_rows(scenes)] for k, v in agents_f.items()}}
    by_arm["O"] = train_scorer.truth_agents_projected(labels, projections)
    expert = labels["ego_future"]
    boxes = [future[r] for r in rows]
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    head = head.to(device).eval()
    spatial, level = torch.tensor([[16, 32]], dtype=torch.long), torch.tensor([0], dtype=torch.long)

    def numbers(proposals, pick):
        plans = proposals[np.arange(len(rows)), pick]
        gap = np.linalg.norm(plans[:, :, :2] - expert[rows][:, :, :2], axis=-1)
        return {
            "l2_1s": gap[:, 1],
            "l2_2s": gap[:, 3],
            "l2_3s": gap[:, 5],
            "collision_3s": nd.collided(plans, boxes, steps=6).astype(np.float64),
            "min_ade_3s": nd.min_ade(proposals, expert[rows], steps=6),
        }

    release = numbers(outputs["proposals"][rows], outputs["pick"][rows])
    result = {"rows": int(len(rows)), "release": {k: float(v.mean()) for k, v in release.items()}, "b2": {}}
    for run_dir in sorted(p for p in Path(args.b2).iterdir() if (p / "summary.json").exists()):
        mode, arm = run_dir.name.rsplit("-", 1)
        summary = json.loads((run_dir / "summary.json").read_text())
        epochs = summary["args"]["epochs"]
        seeds = {}
        for key in sorted(k for k in summary if k.startswith("seed")):
            _, model, adapter = train_refiner.build(entered, mode, arm != "C")
            state = torch.load(run_dir / key / f"epoch_{epochs:02d}.pt", map_location="cpu")
            adapter.load_state_dict(state["adapter"])
            model.load_state_dict(state["refiner"], strict=False)
            model, adapter = model.to(device).eval(), adapter.to(device).eval()
            model._backbone = model._backbone.cpu()
            proposals = np.zeros((len(rows), 32, 8, 3), np.float32)
            pick = np.zeros(len(rows), np.int64)
            with torch.no_grad():
                for i in range(0, len(rows), train_refiner.BATCH):
                    chunk = rows[i : i + train_refiner.BATCH]
                    if arm != "C":
                        adapter.context = {
                            k: torch.from_numpy(np.asarray(v[chunk])).to(device)
                            for k, v in by_arm[arm].items()
                        }
                        adapter.context["mask"] = adapter.context["mask"].bool()
                    props, bev = refine_from_cache(
                        model,
                        torch.from_numpy(outputs["tokens"][chunk].astype(np.float32)),
                        torch.from_numpy(outputs["lidar2img"][chunk][:, 1:2]),
                        torch.from_numpy(outputs["img_shape"][chunk]),
                        torch.from_numpy(outputs["ego_status"][chunk]),
                        spatial,
                        level,
                    )
                    proposals[i : i + len(chunk)] = props[-1].float().cpu().numpy()
                    pick[i : i + len(chunk)] = (
                        head(bev.reshape(len(chunk), 32, 8, -1).amax(2))[..., -1].argmax(1).cpu().numpy()
                    )
            mine = numbers(proposals, pick)
            seeds[key] = {
                **{k: float(v.mean()) for k, v in mine.items()},
                "minus_release": {
                    k: {
                        "mean": float((mine[k] - release[k]).mean()),
                        "se": b_report.clustered_se(mine[k] - release[k], scenes[rows]),
                    }
                    for k in mine
                },
            }
            del model, adapter
            torch.cuda.empty_cache()
        result["b2"][run_dir.name] = seeds
        print(
            json.dumps({run_dir.name: {k: round(v["min_ade_3s"], 3) for k, v in seeds.items()}}), flush=True
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))


# ------------------------------------------------------------- step 7: the side-car alone, warped or not
#
# The rerun's step 7 (docs/drivejepa-perception-sidecar-A.md §8.2, choice 11.12) reads the side-car alone
# on these keyframes twice: as recorded, and with every frame warped to NAVSIM's CAM_F0. The two cameras
# differ by about a degree in where their rays point and by a few per cent in focal length (§4.1, §8.2);
# a rotation and a change of intrinsics about one centre are undone exactly by a homography, so the warped
# read keeps nuScenes' images and asks only whether the side-car's distance error is the camera's geometry.
# The cameras' centres differ by centimetres; the warp keeps nuScenes' own, and its projection says so.

#: NAVSIM's CAM_F0 frame, which the warp renders into before the vendor's crop and resize.
NAVSIM_SIZE = (1920, 1080)
#: The largest gap, in full-resolution pixels, allowed between a point carried by the homography and the
#: same point projected through the warped camera: the two must agree to rounding.
WARP_LIMIT_PX = 1e-3
#: Rows the identity warp is held to the as-recorded ViT output on, spread over the scenes.
WARP_PARITY_ROWS = 32
#: Scene-resampled interval draws for step 7's headline numbers.
DRAWS = 1000
#: H15's class names, in navsim_data.CLASSES' order.
CLASS_NAMES = ("vehicle", "pedestrian", "cyclist")


def rq3(m: np.ndarray) -> tuple:
    """m = k @ r for a 3x3 m: k upper triangular with a positive diagonal, r orthonormal."""
    flip = np.flipud(np.eye(3))
    q, r = np.linalg.qr((flip @ np.asarray(m, dtype=np.float64)).T)
    k, rot = flip @ r.T @ flip, flip @ q.T
    signs = np.diag(np.sign(np.diag(k)))
    return k @ signs, signs @ rot


def crop_resize(width: int, height: int) -> np.ndarray:
    """The 3x3 that takes a width x height camera's pixels to the 512x256 token image (`to_512_direct`)."""
    plain = {"CAM_FRONT": {"intrinsics": np.eye(3).tolist(), "pose": np.eye(4).tolist()}}
    return to_512_direct(plain, width, height)[:3, :3]


def camera_of(projection: np.ndarray, width: int, height: int) -> dict:
    """K at full resolution, the ego -> camera rotation and the camera's centre, out of an ego ->
    token-image projection made for a width x height camera."""
    p = np.asarray(projection, dtype=np.float64)
    k_tokens, rotation = rq3(p[:3, :3])
    if np.linalg.det(rotation) <= 0:
        raise ValueError("the projection's rotation is not proper")
    k = np.linalg.solve(crop_resize(width, height), k_tokens)
    return {"k": k / k[2, 2], "rotation": rotation, "centre": -np.linalg.solve(p[:3, :3], p[:3, 3])}


def homography(k_from: np.ndarray, rot_from: np.ndarray, k_to: np.ndarray, rot_to: np.ndarray) -> np.ndarray:
    """Pixels of one camera to the pixels of another at the same centre: exact for every scene point,
    since only the rotation (ego -> camera) and the intrinsics change."""
    return k_to @ rot_to @ rot_from.T @ np.linalg.inv(k_from)


def warped_projection(k_to: np.ndarray, rot_to: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Ego -> token-image projection of the warped frame: NAVSIM's K and rotation at nuScenes' centre,
    through NAVSIM's crop and resize."""
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = np.asarray(rot_to).T, centre
    return to_512_direct({"CAM_FRONT": {"intrinsics": k_to, "pose": pose}}, *NAVSIM_SIZE)


def warp_gap(h: np.ndarray, source: dict, target: dict) -> float:
    """The largest full-resolution pixel gap between points ahead carried by `h` from the source camera
    and the same points projected through the target camera (both {k, rotation, centre})."""
    xs, ys, zs = np.meshgrid(np.linspace(5, 50, 10), np.linspace(-10, 10, 9), (-0.5, 0.5, 2.0))
    points = np.c_[xs.ravel(), ys.ravel(), zs.ravel()]

    def pixels(cam, pts):
        h_ = (cam["k"] @ cam["rotation"] @ (pts - cam["centre"]).T).T
        return h_[:, :2] / h_[:, 2:3]

    carried = np.c_[pixels(source, points), np.ones(len(points))] @ h.T
    return float(np.abs(carried[:, :2] / carried[:, 2:3] - pixels(target, points)).max())


def navsim_camera(lidar2img: np.ndarray, rows: np.ndarray) -> dict:
    """The camera step 7 warps to, by a fixed rule: of `rows` (navtrain's train rows), the one whose
    CAM_F0 lands a grid of points ahead closest, in mean pixels, to the per-point median over all of
    them. NAVSIM's camera varies by half a degree (plan §4.1); this picks a typical one, not an average
    that no car carried."""
    projections = tg.front_projection(lidar2img[rows])
    xs, ys = np.meshgrid(np.linspace(5, 50, 10), np.linspace(-10, 10, 9))
    grid = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 0.5), np.ones(xs.size)]
    h = np.einsum("nij,gj->ngi", projections, grid)
    pix = h[..., :2] / h[..., 2:3]
    cost = np.linalg.norm(pix - np.median(pix, axis=0), axis=-1).mean(-1)
    best = int(np.argmin(cost))
    return {
        "row": int(rows[best]),
        "cost_px": float(cost[best]),
        **camera_of(projections[best], *NAVSIM_SIZE),
    }


def warped_dataset(ticks: list, scenes: dict, nuscenes_root: str, calibration_dir: Path, warps: dict):
    """`nd.keyframe_dataset`, its frames warped first: `warps` maps a scene to its homography, or to None
    for the frame as recorded. The crop and resize after the warp are the vendor's (`TickDataset.frame`)."""
    import types

    import cv2
    from PIL import Image

    dataset = nd.keyframe_dataset(ticks, scenes, nuscenes_root, calibration_dir)
    scene_of = {
        str(Path(nuscenes_root) / f["file"]): name
        for name, scene in scenes.items()
        for f in scene["keyframes"]
    }

    def frame(self, path):
        key = str(path)
        if key not in self._frames:
            image = np.asarray(Image.open(path).convert("RGB"))
            h = warps[scene_of[key]]
            if h is not None:
                image = cv2.warpPerspective(image, h, NAVSIM_SIZE, flags=cv2.INTER_LINEAR, borderValue=0)
            self._frames[key] = cv2.resize(image[tg.CROP_ROWS : -tg.CROP_ROWS], (512, 256))
        return self._frames[key]

    # The base class reads frames through self.frame, so an instance method replaces it for this
    # dataset alone, and every other path (history, poses, the camera feature) stays the vendor's.
    dataset.frame = types.MethodType(frame, dataset)
    return dataset


def border_share(h: np.ndarray, source_size: tuple) -> float:
    """The share of the warped frame's cropped rows (what the ViT reads) that fall outside the source
    image and are black."""
    import cv2

    inside = cv2.warpPerspective(
        np.full(source_size[::-1], 255, np.uint8), h, NAVSIM_SIZE, flags=cv2.INTER_NEAREST
    )
    return float((inside[tg.CROP_ROWS : -tg.CROP_ROWS] == 0).mean())


def navtrain_train_rows(labels_dir: Path) -> np.ndarray:
    """navtrain's train rows of a labels pass, by the committed split."""
    labels = np.load(Path(labels_dir) / "labels.npz")
    logs = json.loads((Path(labels_dir) / "manifest.json").read_text())["logs"]
    token_logs = np.array([logs[i] for i in labels["log_id"]])
    splits = json.loads(
        (Path(__file__).resolve().parents[3] / "recipes/drivejepa_perception/splits.json").read_text()
    )
    return np.flatnonzero(np.isin(token_logs, splits["train"]))


def run_warp_pass(cfg, args) -> None:
    """Every built keyframe's ViT output with its frames warped to NAVSIM's camera, and the projection
    the side-car reads them with. Checks first, refusing on failure: the homography and the warped
    projection agree on every scene, and the identity warp reproduces the as-recorded ViT output."""
    import pickle

    import torch

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env, tokens

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started, sha = time.time(), git_sha()
    entered = env.enter(cfg)
    with open(Path(args.extract) / "extract.pkl", "rb") as fh:
        data = pickle.load(fh)
    root = str(cfg.resolve("{data.nuscenes}"))
    ticks = ood_ticks(data)
    nusc = Path(args.nusc)
    outputs = np.load(nusc / "outputs.npz")
    built = outputs["built"]
    if len(built) != len(ticks):
        raise SystemExit(f"{nusc} has {len(built)} rows, the extract {len(ticks)}")

    lidar2img = np.load(Path(args.navsim_tokens) / "lidar2img.npy", mmap_mode="r")
    target = navsim_camera(np.asarray(lidar2img), navtrain_train_rows(Path(args.navsim_labels)))
    scenes = data["scenes"]
    names = sorted({t["scene"] for t in ticks})
    warps, per_scene, projections = {}, {}, {}
    for name in names:
        cam = scenes[name]["calibration"]["CAM_FRONT"]
        pose = np.asarray(cam["pose"], dtype=np.float64)
        source = {
            "k": np.asarray(cam["intrinsics"], dtype=np.float64),
            "rotation": pose[:3, :3].T,
            "centre": pose[:3, 3],
        }
        aimed = {"k": target["k"], "rotation": target["rotation"], "centre": source["centre"]}
        h = homography(source["k"], source["rotation"], target["k"], target["rotation"])
        gap = warp_gap(h, source, aimed)
        if not gap <= WARP_LIMIT_PX:
            raise SystemExit(f"{name}: the homography and the warped camera disagree by {gap} px")
        warps[name] = h
        projections[name] = warped_projection(target["k"], target["rotation"], source["centre"])
        per_scene[name] = {
            "gap_px": gap,
            "black_share": border_share(h, tuple(scenes[name]["image_size"])),
            "turn_deg": float(
                np.degrees(
                    np.arccos(np.clip((np.trace(target["rotation"] @ source["rotation"].T) - 1) / 2, -1, 1))
                )
            ),
        }

    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.cuda().eval()
    captured: dict = {}
    model._backbone.img_backbone.register_forward_hook(lambda m, i, o: captured.__setitem__("vit_out", o))

    def vit_of(features: list) -> np.ndarray:  # tokens.run_vit_out's, on this pass's features
        cam_1 = torch.stack([f["camera_feature_1"] for f in features]).cuda()
        cam_2 = torch.stack([f["camera_feature_2"] for f in features]).cuda()
        camera = torch.cat([model.transform(cam_2)[:, None], model.transform(cam_1)[:, None]], dim=1)
        model._backbone(camera, img_metas={})
        return captured["vit_out"].reshape(len(features), 512, 1024).half().cpu().numpy()

    def camera_features(dataset, i):
        agent_input = dataset.agent_input(ticks[i], resized=True)
        return None if agent_input is None else dataset.camera_feature(agent_input)

    calib = Path(args.extract) / "calib"
    rows = np.flatnonzero(built)
    picked = rows[np.linspace(0, len(rows) - 1, WARP_PARITY_ROWS).astype(int)]
    plain = warped_dataset(ticks, scenes, root, calib, {name: None for name in names})
    reference = np.load(nusc / "vit_out.npy", mmap_mode="r")
    with torch.no_grad():
        mine = vit_of([camera_features(plain, int(i)) for i in picked]).astype(np.float32)
    diff = np.abs(mine - np.asarray(reference[picked], dtype=np.float32))
    parity = {"rows": int(len(picked)), "max_abs": float(diff.max()), "mean_abs": float(diff.mean())}
    if parity["max_abs"] > tokens.VIT_LIMITS["max_abs"] or parity["mean_abs"] > tokens.VIT_LIMITS["mean_abs"]:
        raise SystemExit(f"the identity warp does not reproduce the as-recorded ViT output: {parity}")
    print(f"identity warp equals the as-recorded ViT output on {parity['rows']} rows: {parity}", flush=True)
    del plain

    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    n = len(ticks)
    vit = np.lib.format.open_memmap(
        staging / "vit_out.npy", mode="w+", dtype=np.float16, shape=(n, 512, 1024)
    )
    projection = np.zeros((n, 4, 4), np.float32)
    done = np.zeros(n, bool)
    dataset = warped_dataset(ticks, scenes, root, calib, warps)
    batch = 16
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = [int(i) for i in rows[start : start + batch]]
            feats = [camera_features(dataset, i) for i in chunk]
            kept = [(i, f) for i, f in zip(chunk, feats) if f is not None]  # noqa: B905 -- Python 3.9
            for (i, _), value in zip(kept, vit_of([f for _, f in kept])):  # noqa: B905
                vit[i] = value
                projection[i] = projections[ticks[i]["scene"]]
                done[i] = True
            if start // batch % 40 == 0:
                print(f"{int(done.sum())}/{len(rows)} keyframes warped", flush=True)
    vit.flush()
    del vit
    np.save(staging / "projection.npy", projection)
    np.save(staging / "built.npy", done)
    navsim_labels = Path(args.navsim_labels)
    navsim = np.load(navsim_labels / "labels.npz")
    navsim_logs = json.loads((navsim_labels / "manifest.json").read_text())["logs"]
    manifest = {
        "pass": "H15's nuScenes keyframes warped to NAVSIM's CAM_F0, the release's ViT output (step 7)",
        "gallops_sha": sha,
        "entered": entered,
        "nusc": str(nusc),
        "rows": n,
        "built": int(done.sum()),
        "as_recorded_built": int(built.sum()),
        "target_camera": {
            "rule": "the navtrain train row whose CAM_F0 lands a grid ahead closest to the per-point median",
            "row": target["row"],
            "token": str(navsim["token"][target["row"]]),
            "log": navsim_logs[int(navsim["log_id"][target["row"]])],
            "cost_px": target["cost_px"],
            "k": target["k"].tolist(),
            "rotation": target["rotation"].tolist(),
        },
        "identity_parity": parity,
        "warp_gap_px_max": max(s["gap_px"] for s in per_scene.values()),
        "black_share_max": max(s["black_share"] for s in per_scene.values()),
        "turn_deg_mean": float(np.mean([s["turn_deg"] for s in per_scene.values()])),
        "scenes": per_scene,
        "seconds": round(time.time() - started),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    staging.rename(out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "scenes"}, default=str))


def scene_interval(counts: dict, scenes: np.ndarray, draws: int = DRAWS, seed: int = 0) -> dict:
    """Recall, precision and F1 within the counts' range, with 95% intervals from resampling whole
    scenes (keyframes of one scene are 0.5 s apart and nearly the same)."""
    names, index = np.unique(scenes, return_inverse=True)
    per = {k: np.bincount(index, weights=v, minlength=len(names)) for k, v in counts.items()}

    def rates(take):
        t, h = per["targets"][take].sum(), per["hits"][take].sum()
        d, r = per["detections"][take].sum(), per["right"][take].sum()
        recall = h / t if t else float("nan")
        precision = r / d if d else float("nan")
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        return np.array([recall, precision, f1])

    point = rates(np.arange(len(names)))
    rng = np.random.default_rng(seed)
    draws_ = np.array([rates(rng.integers(0, len(names), len(names))) for _ in range(draws)])
    low, high = np.nanpercentile(draws_, [2.5, 97.5], axis=0)
    out = {}
    for j, name in enumerate(("recall", "precision", "f1")):
        out[name], out[f"{name}_95"] = float(point[j]), [float(low[j]), float(high[j])]
    out["targets"] = int(per["targets"].sum())
    return out


def ray_summary(errors: dict, bins=(0.0, 10.0, 20.0, 30.0, 50.0)) -> list:
    """Median depth and direction errors by target range (notebook 06 §7's table)."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):  # noqa: B905
        sel = (errors["range"] >= lo) & (errors["range"] < hi)
        n = int(sel.sum())
        out.append(
            {
                "range_m": [lo, hi],
                "pairs": n,
                "median_abs_depth_m": float(np.median(np.abs(errors["depth"][sel]))) if n else float("nan"),
                "median_depth_m": float(np.median(errors["depth"][sel])) if n else float("nan"),
                "median_abs_lateral_m": float(np.median(np.abs(errors["lateral"][sel])))
                if n
                else float("nan"),
                "median_abs_direction_deg": float(np.median(np.abs(errors["direction_deg"][sel])))
                if n
                else float("nan"),
                "share_depth_over_2m": float((np.abs(errors["depth"][sel]) > 2).mean())
                if n
                else float("nan"),
            }
        )
    return out


def scene_groups(scenes: dict, row_scenes: np.ndarray) -> dict:
    """Per keyframe, the groups step 7 breaks its numbers down by (plan §8.2): location, traffic side,
    and night and rain from each scene's own description."""
    location = np.array([scenes[s]["location"] for s in row_scenes])
    words = np.array([scenes[s]["description"].lower() for s in row_scenes])
    groups = {f"location={name}": location == name for name in sorted(set(location.tolist()))}
    left = np.array([name.startswith("singapore") for name in location])
    groups.update({"traffic=left": left, "traffic=right": ~left})
    groups.update(
        {"night": np.array(["night" in w for w in words]), "rain": np.array(["rain" in w for w in words])}
    )
    return groups


def selected_f1_check(run: Path, loaded: tuple, device) -> dict:
    """Check (a): the run's kept epoch, read on NAVSIM's select logs with this readout's code, must give
    the select F1 its training recorded (metrics.jsonl), so a nuScenes number is the run's and not the
    reader's. `loaded` is `train_sidecar.load`'s output for the run's tap."""
    from gallops.drivejepa_perception import train_sidecar

    meta = json.loads((run / "run.json").read_text())
    run_args = meta["args"]
    model, kept = train_sidecar.load_run(run)
    model = model.to(device)
    tokens_, slot, rows, projections, targets = loaded
    dataset = train_sidecar.Rows(tokens_, slot, rows["select"], projections, targets)
    blind = bool(run_args.get("blind"))
    readout = train_sidecar.read(
        model, dataset, targets, rows["select"], device, meta["prior_gap_m"], blind=blind
    )
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    recorded = next(r["f1_30m"] for r in records if r["epoch"] == kept)
    mine = train_sidecar.f1(readout)
    return {"kept_epoch": kept, "recorded_f1": recorded, "reread_f1": mine, "gap": abs(mine - recorded)}


def read_sidecar(cfg, args) -> None:
    """Step 7's readout: each named run's side-car, at its kept epoch and its last, on every built
    keyframe, as recorded and (with --warp) warped. Q9 is the vehicles' recall at 2 m within 30 m on
    the frames as recorded (PREDICTIONS.md); everything else is reported, not decided."""
    import pickle

    import torch

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env, evaluate, train_sidecar

    out = Path(args.out)
    if (out / "readout.json").exists():
        raise SystemExit(f"{out}/readout.json exists; a readout never overwrites one")
    entered = env.enter(cfg)
    device = torch.device("cuda")
    runs = [Path(r) for r in args.runs.split(",") if r]
    taps = {
        (a["tap"], a.get("vit_tokens", ""))
        for a in (json.loads((run / "run.json").read_text())["args"] for run in runs)
    }
    if len(taps) != 1:
        raise SystemExit(f"the runs read different taps: {sorted(taps)}")
    tap, vit_tokens = taps.pop()
    loaded = train_sidecar.load(Path(args.navsim_tokens), Path(args.navsim_labels), tap, vit_tokens)
    checks = {}
    for run in runs:
        checks[run.name] = selected_f1_check(run, loaded, device)
        print(json.dumps({"check_a": run.name, **checks[run.name]}), flush=True)
        if not checks[run.name]["gap"] <= 1e-4:
            raise SystemExit(
                f"{run.name}: the readout does not reproduce the recorded select F1: {checks[run.name]}"
            )
    del loaded

    with open(Path(args.extract) / "extract.pkl", "rb") as fh:
        scenes = pickle.load(fh)["scenes"]
    nusc = Path(args.nusc)
    labels = dict(np.load(nusc / "labels.npz"))
    outputs = np.load(nusc / "outputs.npz")
    row_scenes = outputs["scene"]
    groups = scene_groups(scenes, row_scenes)
    variants = {
        "recorded": {
            "tokens": np.load(nusc / "vit_out.npy", mmap_mode="r"),
            "projections": (
                tg.to_512(*NUSC_SIZE) @ outputs["lidar2img"][:, tg.FRONT_SLOT].astype(np.float64)
            ).astype(np.float32),
            "built": outputs["built"],
        }
    }
    if args.warp:
        warp = Path(args.warp)
        variants["warped"] = {
            "tokens": np.load(warp / "vit_out.npy", mmap_mode="r"),
            "projections": np.load(warp / "projection.npy"),
            "built": np.load(warp / "built.npy") & outputs["built"],
        }
    # One set of rows for both reads, so the two differ only in their frames and geometry.
    rows = np.flatnonzero(np.all([v["built"] for v in variants.values()], axis=0))
    result = {
        "gallops_sha": git_sha(),
        "entered": entered,
        "rows": int(len(rows)),
        "scenes": int(len(set(row_scenes[rows].tolist()))),
        "check_a_select_f1": checks,
        "groups": {name: int(mask[rows].sum()) for name, mask in groups.items()},
        "variants": {},
    }
    details = {}
    for vname, v in variants.items():
        projections = v["projections"]
        targets = [tg.build(navsim_data.agents_of(labels, r), projections[r]) for r in range(len(row_scenes))]
        subsets = train_sidecar.importance_subsets(nusc, projections, targets, rows)
        gts = [targets[r] for r in rows]
        result["variants"][vname] = {}
        for run in runs:
            meta = json.loads((run / "run.json").read_text())
            run_args = meta["args"]
            blind = bool(run_args.get("blind"))
            summary = json.loads((run / "summary.json").read_text())
            model = train_sidecar.build_model(run_args).to(device)
            entry = {}
            for which, epoch in (("kept", summary["kept_epoch"]), ("last", summary["last_epoch"])):
                state = torch.load(run / f"epoch_{epoch:02d}.pt", map_location="cpu")["state_dict"]
                model.load_state_dict(state)
                dataset = train_sidecar.Rows(v["tokens"], int, rows, projections, targets, blind=blind)
                preds = train_sidecar.predict(model, dataset, device, blind=blind)
                read_ = evaluate.readout(preds, gts, meta["prior_gap_m"])
                read_.update(evaluate.slot_numbers(preds))
                read_.update(evaluate.recall_of(preds, gts, subsets))
                read_["f1_30m"] = train_sidecar.f1(read_)
                read_["ap_2m_within_30m"] = {
                    cls: evaluate.average_precision(preds, gts, c)["ap"] for c, cls in enumerate(CLASS_NAMES)
                }
                cal = evaluate.calibration(preds, gts)
                read_["calibration"] = {k: np.asarray(val).tolist() for k, val in cal.items()}
                errors = evaluate.ray_errors(preds, gts, [projections[r] for r in rows])
                read_["ray_errors"] = ray_summary(errors)
                vehicles = evaluate.per_sample_counts(preds, gts, cls=0)
                every = evaluate.per_sample_counts(preds, gts)
                read_["q9_vehicles_within_30m"] = scene_interval(vehicles, row_scenes[rows])
                read_["all_classes_within_30m"] = scene_interval(every, row_scenes[rows])
                read_["by_group"] = {}
                for gname, mask in groups.items():
                    sel = mask[rows]
                    if sel.sum() < 20:
                        continue
                    part = {k: c[sel] for k, c in every.items()}
                    part_v = {k: c[sel] for k, c in vehicles.items()}
                    read_["by_group"][gname] = {
                        "keyframes": int(sel.sum()),
                        "scenes": int(len(set(row_scenes[rows][sel].tolist()))),
                        "all_classes": scene_interval(part, row_scenes[rows][sel]),
                        "vehicles": scene_interval(part_v, row_scenes[rows][sel]),
                    }
                entry[which] = {"epoch": int(epoch), **read_}
                key = f"{vname}_{run.name}_{which}"
                details.update({f"{key}_{k}": np.asarray(val) for k, val in every.items()})
                details.update({f"{key}_vehicle_{k}": np.asarray(val) for k, val in vehicles.items()})
                details.update({f"{key}_ray_{k}": np.asarray(val) for k, val in errors.items()})
                print(
                    json.dumps(
                        {
                            "variant": vname,
                            "run": run.name,
                            "which": which,
                            "q9": round(read_["q9_vehicles_within_30m"]["recall"], 4),
                            "f1_30m": round(read_["f1_30m"], 4),
                        }
                    ),
                    flush=True,
                )
            result["variants"][vname][run.name] = entry
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "details.npz", rows=rows, scene=row_scenes[rows], **details)
    (out / "readout.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({"rows": result["rows"], "scenes": result["scenes"]}))


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.ood")
    parser.add_argument(
        "command", choices=("extract", "pass", "read", "read-b2", "warp-pass", "read-sidecar")
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--extract", default="", help="pass, warp-pass, read-sidecar: the extract directory")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--nusc", default="", help="read, warp-pass, read-sidecar: the pass's output")
    parser.add_argument("--sidecar", default="", help="read: the A1 run whose side-car is read")
    parser.add_argument("--b1", default="", help="read: the B1 runs' directory")
    parser.add_argument("--b2", default="", help="read-b2: the B2 runs' directory")
    parser.add_argument(
        "--warp", default="", help="read-sidecar: the warp-pass's output, for the warped read"
    )
    parser.add_argument("--runs", default="", help="read-sidecar: comma-separated side-car runs")
    parser.add_argument("--navsim-tokens", default="", help="warp-pass, read-sidecar: tokens-v1")
    parser.add_argument("--navsim-labels", default="", help="warp-pass, read-sidecar: labels-v1")
    args = parser.parse_args(argv)
    cfg = SiteConfig.load()
    if args.command == "extract":
        run_extract(cfg, Path(args.out))
    elif args.command == "pass":
        run_pass(cfg, Path(args.extract), Path(args.out), args.workers)
    elif args.command == "read-b2":
        read_b2(cfg, args)
    elif args.command == "warp-pass":
        run_warp_pass(cfg, args)
    elif args.command == "read-sidecar":
        read_sidecar(cfg, args)
    else:
        read(cfg, args)


if __name__ == "__main__":
    main()
