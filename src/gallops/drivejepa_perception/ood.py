"""OOD (PREDICTIONS.md P15): the side-car and the B1 scorers on nuScenes, which nothing here saw.

    python -m gallops.drivejepa_perception.ood extract --out {out}/nusc-extract          # CPU, once
    python -m gallops.drivejepa_perception.ood pass --extract {out}/nusc-extract --out {out}/nusc-v1
    python -m gallops.drivejepa_perception.ood read --nusc {out}/nusc-v1 --sidecar {out}/runs/a1-s0 \
        --b1 {out}/b1 --out {out}/ood.json

**Rows.** Every trainable keyframe (H14's `nusc.ticks`: the keyframe 0.5 s back and 4 s of future
keyframes exist) of H14's 102 held-out nuScenes train scenes, the list `drivejepa_aux.split` drew
and H14 committed. The release trained on NAVSIM and H15 on navtrain, so every number here is out
of sample for all of them: the comparison with the release that navtrain cannot give fairly.

**Inputs** are H14's: `drivejepa_aux.inputs.TablesTickDataset`, which is swerve_data's TickDataset
fed from the extract (the vendor's feature builder, nuScenes' cameras on the ego frame, the 28-row
crop). The release runs at batch 1 in eval mode, so no batch_fix is involved, and the pass keeps
what H15's readers read: the ViT's output (the side-car's tap), the scorer's pooled input, the 32
proposals, the 6 logits and the pick, and lidar2img, img_shape and the ego status.

**Labels**, in H15's layout (`navsim_data.pack`), from the annotations at the keyframe:
- vehicles, pedestrians and two-wheelers (`class_of`), each with at least one lidar point:
  nuScenes' detection convention, since an annotation no sensor saw is no target for a camera;
- boxes in the keyframe's ego frame and ground velocity in its axes, as H14's `nusc._boxes`;
- each agent's path at +0.5 s and +1.0 s: its instance at the next two keyframes, which are 0.5 s
  apart, in this keyframe's frame.
Every annotated box at the next 8 keyframes (H14's `nusc.future_boxes`) gives the collision rate
(H14's `metrics.collided`: the ego's footprint at a waypoint overlaps a box, within 3 s).

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
    from gallops.drivejepa_aux import nusc

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
    boxes, centres, velocity = nusc._boxes(chosen, ego2global, ("",))
    height = np.array([a["size_wlh"][2] for a in chosen])
    # box: x, y, z, length, width, height, yaw (navsim_data's order)
    out["box"] = np.stack(
        [boxes[:, 0], boxes[:, 1], centres[:, 2], boxes[:, 3], boxes[:, 4], height, boxes[:, 2]], axis=1
    ).astype(np.float32)
    out["velocity"] = np.nan_to_num(velocity).astype(np.float32)
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


def heldout_ticks(extract: dict) -> list:
    """Every trainable keyframe of H14's held-out scenes, scenes sorted, keyframes in order."""
    from gallops.drivejepa_aux import split

    held = split.load()["held_out"]
    return [t for name in sorted(held) for t in extract["ticks"].get(name, [])]


def run_extract(cfg, out: Path) -> None:
    """H14's extract of the held-out scenes alone, with their calibration (nusc extract's layout)."""
    import pickle

    from gallops.drivejepa_aux import nusc, split

    out = Path(out)
    if (out / "extract.pkl").exists():
        raise SystemExit(f"{out}/extract.pkl exists; an extract is written once")
    root = Path(cfg.resolve("{data.nuscenes}"))
    names = sorted(split.load()["held_out"])
    started = time.time()
    data = nusc.extract(root, names)
    data["ticks"] = {name: nusc.ticks(name, scene, root) for name, scene in data["scenes"].items()}
    out.mkdir(parents=True, exist_ok=True)
    nusc.write_calibration(data["scenes"], out / "calib")
    with open(out / "extract.pkl", "wb") as fh:
        pickle.dump(data, fh, protocol=4)
    counts = {"scenes": len(names), "keyframes": sum(len(v) for v in data["ticks"].values())}
    print(json.dumps({**counts, "seconds": round(time.time() - started)}))


def run_pass(cfg, extract_dir: Path, out: Path, workers: int) -> None:
    """The release at batch 1 over every held-out keyframe, and the keyframes' labels."""
    import pickle
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from gallops.drivejepa_aux import inputs, nusc
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started, sha = time.time(), git_sha()
    entered = env.enter(cfg)
    with open(Path(extract_dir) / "extract.pkl", "rb") as fh:
        data = pickle.load(fh)
    inputs.use_calibration(Path(extract_dir) / "calib")
    root = Path(cfg.resolve("{data.nuscenes}"))
    ticks = heldout_ticks(data)
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
        dataset = inputs.TablesTickDataset(data["ticks"][name], "cpu", data["scenes"], str(root))
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
    future = [nusc.future_boxes(data["scenes"][t["scene"]], t, data["annotations"]) for t in ticks]
    with open(staging / "future_boxes.pkl", "wb") as fh:
        pickle.dump(future, fh, protocol=4)
    np.savez(staging / "outputs.npz", scene=np.array([t["scene"] for t in ticks]), **arrays)
    manifest = {
        "pass": "nuScenes held-out keyframes, the release at batch 1 (H15 OOD)",
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

    from gallops.drivejepa_aux import metrics as aux_metrics
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
            "collision_3s": aux_metrics.collided(plans, boxes, steps=6).astype(np.float64),
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


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.ood")
    parser.add_argument("command", choices=("extract", "pass", "read"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--extract", default="", help="pass: the extract directory")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--nusc", default="", help="read: the pass's output")
    parser.add_argument("--sidecar", default="", help="read: the A1 run whose side-car is read")
    parser.add_argument("--b1", default="", help="read: the B1 runs' directory")
    args = parser.parse_args(argv)
    cfg = SiteConfig.load()
    if args.command == "extract":
        run_extract(cfg, Path(args.out))
    elif args.command == "pass":
        run_pass(cfg, Path(args.extract), Path(args.out), args.workers)
    else:
        read(cfg, args)


if __name__ == "__main__":
    main()
