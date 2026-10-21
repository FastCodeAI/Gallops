"""The cache: every label, and the frozen encoder run once over every keyframe, on disk.

    python -m gallops.drivejepa_aux.cache labels --extract <dir> --out <cache>   # CPU: numpy + the map
    python -m gallops.drivejepa_aux.cache tokens --extract <dir> --out <cache>   # GPU: the released model

**Why tokens can be cached.** The encoder (ViT-L and its projector, 304 M parameters) is frozen in
every arm, so what it makes of a keyframe never changes. Run once, its 512 tokens per keyframe are
what a live pass would give, and each epoch then trains the 2.4 M-parameter refiner path on them in
under a minute. "What a live pass would give" is checked, not assumed: the preflight rebuilds a
sample of keyframes live and compares.

Two passes, each writing its own directory under <cache> whole or not at all (staged, then renamed):

    labels/labels.npz    every target `nusc.keyframe_labels` builds; the full road mask and every
                         box at each future keyframe, for the held-out metrics; and how much of the
                         human's path in view lies on the rasterised road, for the preflight
    tokens/tokens.npy    (N, 512, 256) float32: the encoder's output
    tokens/inputs.npz    the rest of what the refiner reads (l2i, img_shape, ego); the human's 4 s;
                         and the release's own answer -- its 32 proposals, their scores and its top
                         2, the pseudo-teachers
    */manifest.json      what made it

Both are keyed by `rows`: every trainable keyframe of the extract, scenes sorted, keyframes in
order. Each records its rows' sample tokens, and `load` refuses two passes that disagree.

**The release's answer is recorded live, at batch 1,** as it drives -- not through the cached,
batched path the trainer uses -- so the preflight compares two different routes to one number.

Runs in Drive-JEPA's interpreter (Python 3.9). The labels pass forks workers, which share the
extract read once; the tokens pass decodes frames on a thread pool, a scene per task, while the
GPU runs the model one keyframe at a time (input build 144 ms a keyframe, forward 29 ms: measured).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing
import pickle
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np

from gallops.drivejepa_aux import bev, metrics, nusc

#: The release's top-k proposals by its own scorer, kept as the pseudo-teachers (`step.release_teachers`).
TEACHERS = 2
LABELS, TOKENS = "labels", "tokens"


def rows(data: dict) -> list[dict]:
    """Every trainable keyframe of an extract, in the cache's order: scenes sorted, keyframes in order."""
    return [tick for name in sorted(data["ticks"]) for tick in data["ticks"][name]]


def load_extract(extract: Path) -> dict:
    with open(Path(extract) / "extract.pkl", "rb") as f:
        return pickle.load(f)


def _staging(out: Path, part: str) -> Path:
    final = out / part
    if final.exists():
        raise SystemExit(f"{final} exists; a cache pass is written once -- choose a new --out")
    staging = out / f"{part}.partial"
    if staging.exists():
        shutil.rmtree(staging)  # an earlier attempt that died before it was renamed into place
    staging.mkdir(parents=True)
    return staging


def _publish(staging: Path, manifest: dict) -> Path:
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    final = staging.with_name(staging.name[: -len(".partial")])
    staging.rename(final)
    return final


def _provenance(extract: Path) -> dict:
    """What a pass read and ran, taken as it starts: read at the end, the sha named whatever was
    checked out by then (swerve's preflight learned the same: v4fix's sidecar named a later tree)."""
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_b2d.sidecar import sha256

    return {
        "extract": str(extract),
        "extract_sha256": sha256(Path(extract) / "extract.pkl"),
        "git_sha": git_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --- labels ---------------------------------------------------------------------------------------

#: The extract and each worker's maps. Set before the pool forks, so the workers share one copy.
_DATA: dict = {}
_MAPS: dict = {}


def keyframe_record(scene: dict, tick: dict, annotations: dict, view_map) -> dict:
    """One row of labels.npz: the targets, the road, the future boxes, the human's path on the road."""
    labels = nusc.keyframe_labels(scene, tick, annotations.get(tick["sample"], []), view_map)
    judged, on = metrics.on_road(np.asarray(tick["expert"]), labels["road"], labels["view"])
    return {
        "bev": labels["bev"],
        "road": labels["road"],
        "agent_states": labels["agent_states"],
        "agent_valid": labels["agent_valid"],
        "lead_present": labels["lead_present"],
        "lead_gap": labels["lead_gap"],
        "lead_closing": labels["lead_closing"],
        "lane_change": labels["lane_change"],
        "expert_judged": judged,
        "expert_on_road": on,
        "future_boxes": nusc.future_boxes(scene, tick, annotations),
    }


def _scene_records(job: tuple) -> list[dict]:
    name, root = job
    scene = _DATA["scenes"][name]
    location = scene["location"]
    if location not in _MAPS:
        _MAPS[location] = nusc.MapView(nusc.devkit_map(Path(root), location))
    return [keyframe_record(scene, t, _DATA["annotations"], _MAPS[location]) for t in _DATA["ticks"][name]]


def stack_records(records: list[dict], ticks: list[dict]) -> dict:
    """labels.npz's arrays from one record per row."""

    def col(key, dtype):
        return np.asarray([r[key] for r in records], dtype=dtype)

    boxes = [r["future_boxes"] for r in records]
    counts = [sum(len(step) for step in b) for b in boxes]
    flat = [step for b in boxes for step in b]
    return {
        "sample": np.asarray([t["sample"] for t in ticks]),
        "scene": np.asarray([t["scene"] for t in ticks]),
        "keyframe": np.asarray([t["keyframe"] for t in ticks], dtype=np.int32),
        "bev": col("bev", np.uint8),
        "road": np.packbits(col("road", bool), axis=-1),
        "agent_states": col("agent_states", np.float32),
        "agent_valid": col("agent_valid", bool),
        "lead_present": col("lead_present", np.int8),
        "lead_gap": col("lead_gap", np.float32),
        "lead_closing": col("lead_closing", np.float32),
        "lane_change": col("lane_change", np.int8),
        "expert_judged": col("expert_judged", np.int16),
        "expert_on_road": col("expert_on_road", np.int16),
        # every row's future boxes, flattened: row i's are offsets[i]:offsets[i + 1], at step `future_step`
        "future_boxes": np.concatenate(flat).astype(np.float32) if flat else np.zeros((0, 5), np.float32),
        "future_step": (
            np.concatenate([np.full(len(s), k, np.int8) for b in boxes for k, s in enumerate(b)])
            if flat
            else np.zeros(0, np.int8)
        ),
        "future_offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
    }


def build_labels(extract: Path, out: Path, nuscenes_root: Path, workers: int, limit: int = 0) -> Path:
    staging = _staging(out, LABELS)
    provenance = _provenance(extract)  # now, not at the end: the code this process loaded
    started = time.time()
    _DATA.update(load_extract(extract))
    names = [n for n in sorted(_DATA["ticks"]) if _DATA["ticks"][n]]
    if limit:
        names = names[:limit]
    jobs = [(n, str(nuscenes_root)) for n in names]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        per_scene = pool.map(_scene_records, jobs, chunksize=1)
    ticks = [t for n in names for t in _DATA["ticks"][n]]
    arrays = stack_records([r for records in per_scene for r in records], ticks)
    np.savez(staging / "labels.npz", **arrays)
    judged, on = int(arrays["expert_judged"].sum()), int(arrays["expert_on_road"].sum())
    manifest = {
        **provenance,
        "rows": len(ticks),
        "scenes": len(names),
        "limit": limit or None,
        "expert_waypoints_judged": judged,
        "expert_waypoints_on_road": on,
        "seconds": round(time.time() - started, 1),
    }
    final = _publish(staging, manifest)
    seconds = manifest["seconds"]
    print(f"labels for {len(ticks)} keyframes of {len(names)} scenes in {seconds:.0f} s -> {final}")
    share = on / max(judged, 1)
    print(f"  the human's path on the rasterised road: {on} of {judged} waypoints in view ({share:.2%})")
    return final


def future_boxes(labels: dict, i: int) -> list:
    """Row i's boxes per future step (`nusc.future_boxes`), back out of labels.npz."""
    lo, hi = labels["future_offsets"][i], labels["future_offsets"][i + 1]
    boxes, step = labels["future_boxes"][lo:hi], labels["future_step"][lo:hi]
    return [boxes[step == k] for k in range(nusc.FUTURE_STEPS)]


def road(labels: dict, index) -> np.ndarray:
    """The full road mask of the rows `index`, unpacked: (len, ROWS, COLS) bool."""
    return np.unpackbits(labels["road"][index], axis=-1, count=bev.COLS).astype(bool)


# --- tokens ---------------------------------------------------------------------------------------


def build_tokens(
    extract: Path, out: Path, checkpoint: str, nuscenes_root: Path, device: str, workers: int, limit: int = 0
):
    """The release, live at batch 1, over every row: its tokens, its inputs and its answer."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from gallops.drivejepa_aux import inputs, step
    from gallops.drivejepa_b2d.sidecar import sha256

    staging = _staging(out, TOKENS)
    provenance = _provenance(extract)  # now, not at the end: the code this process loaded
    started = time.time()
    data = load_extract(extract)
    inputs.use_calibration(Path(extract) / "calib")
    names = [n for n in sorted(data["ticks"]) if data["ticks"][n]]
    if limit:
        names = names[:limit]
    ticks = [t for n in names for t in data["ticks"][n]]
    n = len(ticks)
    first = {}
    for i, t in enumerate(ticks):
        first.setdefault(t["scene"], i)

    with contextlib.redirect_stdout(io.StringIO()):  # the vendor prints the whole ViT as it builds
        agent, model, _, config = step.build_model(checkpoint, device, aux=False)
    tokens = np.lib.format.open_memmap(
        staging / "tokens.npy", mode="w+", dtype=np.float32, shape=(n, 512, config.tf_d_model)
    )
    arrays = {
        "sample": np.asarray([t["sample"] for t in ticks]),
        "built": np.zeros(n, bool),
        "l2i": np.zeros((n, 1, 4, 4), np.float32),
        "img_shape": np.zeros((n, 4, 3), np.float32),
        "ego": np.zeros((n, 11), np.float32),
        "target": np.asarray([t["expert"] for t in ticks], np.float32),
        "proposals": np.zeros((n, 32, 8, 3), np.float32),
        "score": np.zeros((n, 32), np.float32),
        "teachers": np.zeros((n, TEACHERS, 8, 3), np.float32),
    }
    shapes = {}

    def build_scene(name: str) -> list:
        # its own dataset per scene: TickDataset caches frames and the scene's slow-path features
        dataset = inputs.TablesTickDataset(data["ticks"][name], "cpu", data["scenes"], str(nuscenes_root))
        return [dataset.batch([j])[0] for j in range(len(dataset))]

    torch.set_num_threads(4)  # per op: the workers' decoding is what needs the cores
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        queue = deque()
        pending = iter(names)
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
                    continue  # counted by `built`, dropped by the trainer
                live = {k: v.to(device) if hasattr(v, "to") else v for k, v in features.items()}
                with torch.no_grad():
                    result = step.forward(model, None, live, keep=True)
                out_ = result["out"]
                tokens[i] = result["tokens"][0, :, 0].float().cpu().numpy()
                arrays["l2i"][i] = features["lidar2img"][0, 1:2].numpy()  # the vendor keeps slot 1
                arrays["img_shape"][i] = features["img_shape"][0].numpy()
                arrays["ego"][i] = features["ego_status"][0, -1].numpy()
                arrays["proposals"][i] = out_["proposals"][0].float().cpu().numpy()
                arrays["score"][i] = out_["pdm_score"][0].float().cpu().numpy()
                arrays["teachers"][i] = step.release_teachers(out_, TEACHERS)[0][0].cpu().numpy()
                arrays["built"][i] = True
                shapes.setdefault("spatial", result["spatial"].cpu().numpy())
                shapes.setdefault("level", result["level"].cpu().numpy())
                done += 1
            if done and (first[name] // 2000) != ((first[name] + len(built)) // 2000):
                rate = done / (time.time() - started)
                left = (n - done) / rate / 60
                print(f"  {done} of {n} keyframes, {rate:.1f}/s, {left:.0f} min left", flush=True)
    tokens.flush()
    del tokens
    np.savez(staging / "inputs.npz", spatial=shapes["spatial"], level=shapes["level"], **arrays)
    manifest = {
        **provenance,
        "rows": n,
        "built": int(arrays["built"].sum()),
        "limit": limit or None,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(Path(checkpoint)),
        "pass": "the vendor's forward, eval mode, batch 1, unchanged (step.forward)",
        "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
        "torch": torch.__version__,
        "seconds": round(time.time() - started, 1),
    }
    final = _publish(staging, manifest)
    print(f"tokens for {manifest['built']} of {n} keyframes in {manifest['seconds'] / 60:.1f} min -> {final}")
    return final


# --- reading --------------------------------------------------------------------------------------


def load(cache: Path, tokens: bool = True) -> dict:
    """Both passes, checked to describe the same rows. Tokens are read whole into memory (10 GB for the
    full extract): every epoch visits every row in a new order."""
    cache = Path(cache)
    out = {"labels": dict(np.load(cache / LABELS / "labels.npz"))}
    out["manifest"] = {LABELS: json.loads((cache / LABELS / "manifest.json").read_text())}
    if tokens:
        out["inputs"] = dict(np.load(cache / TOKENS / "inputs.npz"))
        out["manifest"][TOKENS] = json.loads((cache / TOKENS / "manifest.json").read_text())
        if not np.array_equal(out["inputs"]["sample"], out["labels"]["sample"]):
            raise SystemExit(f"{cache}: the labels and tokens passes describe different rows")
        if out["manifest"][TOKENS]["extract_sha256"] != out["manifest"][LABELS]["extract_sha256"]:
            raise SystemExit(f"{cache}: the labels and tokens passes read different extracts")
        out["tokens"] = np.load(cache / TOKENS / "tokens.npy")
    return out


def batch(data: dict, index, device: str, with_labels: bool = True) -> dict:
    """The rows `index` as the trainer feeds them: the refiner's inputs, the human's plan, the
    pseudo-teachers and, with `with_labels`, the heads' targets with `heads.collate`'s keys and
    dtypes. One function for the trainer and the preflight, so the preflight checks this one."""
    import torch

    index = np.asarray(index)
    inputs, labels = data["inputs"], data["labels"]

    def put(array, dtype=torch.float32):
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)

    out = {
        "inputs": {
            "tokens": put(data["tokens"][index]),
            "l2i": put(inputs["l2i"][index]),
            "img_shape": put(inputs["img_shape"][index]),
            "ego": put(inputs["ego"][index]),
        },
        "target": put(inputs["target"][index]),
        "teachers": put(inputs["teachers"][index]),
        "n_teachers": torch.full((len(index),), inputs["teachers"].shape[1], device=device),
        "labels": None,
    }
    if with_labels:
        out["labels"] = {
            "bev": put(labels["bev"][index], torch.long),
            "agent_states": put(labels["agent_states"][index]),
            "agent_valid": put(labels["agent_valid"][index], torch.bool),
            "lead_present": put(labels["lead_present"][index]),
            "lead_gap": put(labels["lead_gap"][index]),
            "lead_closing": put(labels["lead_closing"][index]),
            "lane_change": put(labels["lane_change"][index], torch.long),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=(LABELS, TOKENS))
    ap.add_argument("--extract", type=Path, required=True, help="the `nusc extract` directory")
    ap.add_argument("--out", type=Path, required=True, help="the cache directory; each pass adds one part")
    ap.add_argument("--nuscenes", type=Path, help="default: gallops.toml's {data.nuscenes}")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes (a quick look)")
    args = ap.parse_args()

    from gallops.config import SiteConfig

    cfg = SiteConfig.load()
    extract, out = args.extract.resolve(), args.out.resolve()  # before `step.enter` changes directory
    root = (args.nuscenes or Path(cfg.resolve("{data.nuscenes}"))).resolve()
    if args.part == LABELS:
        build_labels(extract, out, root, args.workers, args.limit)
    else:
        from gallops.drivejepa_aux import step

        checkpoint = step.enter(cfg)
        build_tokens(extract, out, checkpoint, root, args.device, args.workers, args.limit)


if __name__ == "__main__":
    main()
