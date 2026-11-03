"""The released Drive-JEPA run once over every moment, its outputs on disk.

    python -m gallops.memory.cache build [--ticks DIR] [--out DIR] [--limit N] [--workers N]  # GPU, resumable

**Why it can be cached.** In the memory picker the encoder, the proposal generator and the released
scorer are all frozen; only the memory and the new picker head train. So what the release makes of
each moment never changes: run once here, each training epoch reads it from disk.

Per moment, float32, keyed by the row order of the ticks table (`ticks.load`):

    encoder.npy    (N, 512, 256)  the encoder's 16 x 32 map of tokens (`ImgEncoder` output 0)
    scorer_in.npy  (N, 32, 256)   the scorer's input: the last refiner pass's query features, max
                                  over each proposal's 8 waypoints (scorer.py:41)
    proposals.npy  (N, 32, 8, 3)  the 32 proposals (x, y, heading), ego frame
    logits.npy     (N, 32, 6)     the released scorer's six logits per proposal
    ego.npy        (N, 11)        the ego status the model was given for the current frame
    done.npy       (N,)           whether the row was built (a missing image leaves it False)

**Inputs** are paddock's node's (`inputs.agent_input`): what the cache stores is what a paddock run
of the same model would compute, which `checks fidelity` measures on NAVSIM's navtest.

**Resumable per log**: rows are written in log order, and `progress.json` records each finished
log; a restart skips them. The arrays are created whole at the start (np.lib.format.open_memmap),
so a killed run leaves a valid file with some rows unbuilt.

**Decoding in parallel** (`--workers N`, default 1). The five image reads of a moment go to a pool
of N threads, and with N > 1 the next moment's inputs are built while the GPU runs this one. The
forward pass stays batch 1 and in row order; the arrays do not depend on N (`checks workers`).

The forward pass is the vendor's own, at batch 1, in eval mode, on a shallow copy of the features
(it rebinds features["lidar2img"]). Runs in Drive-JEPA's interpreter (Python 3.9, torch 2.7.1).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from gallops.compat import zip_strict

#: The released checkpoint, where paddock's model.yaml finds it.
RELEASE = "{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"

ARRAYS = {
    "encoder": (512, 256),
    "scorer_in": (32, 256),
    "proposals": (32, 8, 3),
    "logits": (32, 6),
    "ego": (11,),
}


# --- the model ---------------------------------------------------------------------------------


def enter(cfg) -> str:
    """Drive-JEPA's environment as paddock's model.yaml gives it to its node; returns the release.

    Changes into navsim_v2/: the agent opens ./data/8192.npy and ./vjepa2/... relative to its cwd.
    (The aux branch's `step.enter`, without paddock on the path: nothing here imports it.)
    """
    navsim_v2 = Path(cfg.resolve("{code.drivejepa}")) / "navsim_v2"
    navsim_data = str(cfg.datasets["navsim"].path)
    os.environ.update(
        NAVSIM_DEVKIT_ROOT=str(navsim_v2),
        NAVSIM_EXP_ROOT=cfg.resolve("{data.navsim.exp}"),
        OPENSCENE_DATA_ROOT=navsim_data,
        NUPLAN_MAPS_ROOT=navsim_data + "/maps",
        NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
        DRIVE_JEPA_MSDA="auto",
    )
    if str(navsim_v2) not in sys.path:
        sys.path.insert(0, str(navsim_v2))
    os.chdir(navsim_v2)
    return cfg.resolve(RELEASE)


def build_model(checkpoint: str, device: str):
    """The released agent as paddock's node loads it (`DriveJepaPolicy.load`): eval mode."""
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    config = DriveJEPAConfig()
    with contextlib.redirect_stdout(io.StringIO()):  # the vendor prints the whole ViT as it builds
        agent = DriveJEPAAgent(config=config, lr=1e-4, checkpoint_path=checkpoint)
        agent.initialize()
    agent.eval().to(device)
    return agent, agent.get_feature_builders()[0]


def run(agent, feature_builder, agent_input, device: str) -> dict:
    """One moment through the release at batch 1: the arrays the cache stores, and the pick."""
    import torch

    features = feature_builder.compute_features(agent_input)
    features = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in features.items()}
    model = agent._pad_model
    captured = {}

    def encoder_out(module, args, output):
        captured["encoder"] = output[0]  # (cameras=1, 512, B, 256)

    def scorer_in(module, args):
        captured["proposals_in"], captured["query"] = args[0], args[1]

    hooks = [
        model._backbone.register_forward_hook(encoder_out),
        model.scorer.register_forward_pre_hook(scorer_in),
    ]
    try:
        with torch.no_grad():
            # the vendor's forward rebinds features["lidar2img"], so it gets a shallow copy
            out = agent.forward(dict(features))
    finally:
        for hook in hooks:
            hook.remove()
    proposals = out["proposals"]
    b, p, t = proposals.shape[:3]
    query = captured["query"].reshape(b, p, t, -1).amax(-2)  # scorer.py:41, the same expression
    logits = out["pred_logit"]
    return {
        "encoder": captured["encoder"][0, :, 0].float().cpu().numpy(),
        "scorer_in": query[0].float().cpu().numpy(),
        "proposals": proposals[0].float().cpu().numpy(),
        "logits": logits[0].float().cpu().numpy(),
        "ego": features["ego_status"][0, -1].float().cpu().numpy(),
        "pick": int(torch.argmax(out["pdm_score"], dim=1)[0]),
        "trajectory": out["trajectory"][0].float().cpu().numpy(),
    }


def released_pick(logits: np.ndarray) -> int:
    """The release's pick from its stored logits: argmax of the final score's sigmoid (model.py:85-90)."""
    return int(np.argmax(1.0 / (1.0 + np.exp(-logits[:, -1].astype(np.float64)))))


# --- the build ---------------------------------------------------------------------------------


def decoders(workers: int):
    """(decode pool, read-ahead thread): both None for 1 worker, the reads then run in the caller."""
    from gallops.memory import inputs

    pool = inputs.decode_pool(workers)
    ahead = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ahead") if workers > 1 else None
    return pool, ahead


def log_inputs(frames: list, frame_idxs: list, roots: list, pool=None, ahead=None):
    """Each moment's AgentInput (or the FileNotFoundError that stopped it), in `frame_idxs`' order.

    With `ahead`, the next moment's inputs are built while the caller uses this one (the GPU
    forward), at most one moment ahead, so memory stays at two moments' images.
    """
    from gallops.memory import inputs

    def make(frame_idx):
        try:
            return inputs.agent_input(frames, frame_idx, roots, pool=pool)
        except FileNotFoundError as exc:
            return exc

    if ahead is None:
        for frame_idx in frame_idxs:
            yield make(frame_idx)
        return
    nxt = ahead.submit(make, frame_idxs[0]) if frame_idxs else None
    for n, _ in enumerate(frame_idxs):
        current = nxt.result()
        nxt = ahead.submit(make, frame_idxs[n + 1]) if n + 1 < len(frame_idxs) else None
        yield current


def open_arrays(out: Path, n: int, mode: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, shape in ARRAYS.items():
        path = out / f"{name}.npy"
        if mode == "w+" and not path.exists():
            arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, *shape))
        else:
            arrays[name] = np.load(path, mmap_mode="r+" if mode != "r" else "r")
    done = out / "done.npy"
    if not done.exists():
        np.save(done, np.zeros(n, bool))
    arrays["done"] = np.load(done, mmap_mode="r+" if mode != "r" else "r")
    return arrays


def build(cfg, ticks_dir: Path, out: Path, device: str, limit: int = 0, rows=None, workers: int = 1) -> dict:
    """Every moment (or `rows`, row indices of the ticks table) through the release; resumable per log."""
    import torch

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_b2d.sidecar import sha256
    from gallops.memory import ticks

    table = ticks.load(ticks_dir)
    n = len(table["log"])
    data = cfg.datasets["navtrain"]
    logs_dir, roots = data.extra["logs"], [data.extra["blobs"], data.extra["extra_blobs"]]
    out = Path(out)
    manifest_path = out / "manifest.json"
    progress_path = out / "progress.json"
    checkpoint = enter(cfg)
    if not manifest_path.exists():
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "git_sha": git_sha(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ticks": str(ticks_dir),
            "ticks_git_sha": table["manifest"].get("git_sha"),
            "rows": n,
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256(Path(checkpoint)),
            "pass": "the vendor's forward, eval mode, batch 1, paddock's NAVSIM inputs (inputs.agent_input)",
            "torch": torch.__version__,
            "dtype": "float32",
            "decode_workers": workers,
        }
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"logs_done": []}
    finished = set(progress["logs_done"])

    arrays = open_arrays(out, n, "w+")
    agent, feature_builder = build_model(checkpoint, device)
    torch.set_num_threads(2)

    wanted = np.arange(n) if rows is None else np.asarray(rows)
    wanted = wanted[table["inputs_ok"][wanted]]
    by_log: dict = {}
    for i in wanted:
        by_log.setdefault(str(table["log"][i]), []).append(int(i))
    names = [name for name in sorted(by_log) if name not in finished]
    if limit:
        names = names[:limit]
    started, built, failed, failed_total = time.time(), 0, [], 0
    pool, ahead = decoders(workers)
    for k, log in enumerate(names):
        with open(logs_dir / f"{log}.pkl", "rb") as f:
            frames = pickle.load(f)
        idx = [int(table["frame_idx"][i]) for i in by_log[log]]
        for i, ai in zip_strict(by_log[log], log_inputs(frames, idx, roots, pool, ahead)):
            if isinstance(ai, FileNotFoundError):
                failed.append({"row": i, "error": str(ai)})
                continue
            result = run(agent, feature_builder, ai, device)
            for name in ARRAYS:
                arrays[name][i] = result[name]
            arrays["done"][i] = True
            built += 1
        for a in arrays.values():
            a.flush()
        if rows is None:
            finished.add(log)
            progress["logs_done"] = sorted(finished)
        progress.update(
            built_this_run=built,
            failed=progress.get("failed", []) + failed,
            updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            seconds_this_run=round(time.time() - started, 1),
            logs_this_run=k + 1,
            logs_left=len(names) - k - 1,
        )
        failed_total += len(failed)
        failed = []
        progress_path.write_text(json.dumps(progress, indent=1) + "\n")
        if (k + 1) % 10 == 0:
            rate = built / (time.time() - started)
            print(
                f"  {k + 1} of {len(names)} logs, {built} moments, {rate:.2f}/s, {failed_total} failed",
                flush=True,
            )
    for executor in (ahead, pool):
        if executor is not None:
            executor.shutdown()
    return {
        "built": built,
        "seconds": round(time.time() - started, 1),
        "logs": len(names),
        "failed": failed_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("build",))
    ap.add_argument("--ticks", type=Path, help="default: {data.navtrain.cache}/ticks")
    ap.add_argument("--out", type=Path, help="default: {data.navtrain.cache}/release")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="only the first N logs (a quick look)")
    ap.add_argument(
        "--workers", type=int, default=1, help="image-decoding threads (the arrays do not change)"
    )
    args = ap.parse_args()

    from gallops.config import SiteConfig

    cfg = SiteConfig.load()
    cache = Path(cfg.resolve("{data.navtrain.cache}"))
    ticks_dir = (args.ticks or cache / "ticks").resolve()
    out = (args.out or cache / "release").resolve()
    if args.device.startswith("cuda"):
        from gallops.memory.checks import wait_for_gpu

        print(f"GPU free at start: {wait_for_gpu()} MiB", flush=True)
    print(json.dumps(build(cfg, ticks_dir, out, args.device, args.limit, workers=args.workers)))


if __name__ == "__main__":
    main()
