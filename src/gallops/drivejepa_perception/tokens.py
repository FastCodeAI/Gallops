"""The token pass: one batch-1 forward of the released model per navtrain sample, cached.

    python -m gallops.drivejepa_perception.tokens --labels {out}/labels-v1 --out {out}/tokens-v1

Rows follow the labels pass's order (`labels.npz`: token, log_id), so row i of every array here is
row i there. Per row:

| file | shape, dtype | what it is | read by |
|---|---|---|---|
| `tokens.npy` | (512, 256) fp16 | post-projector image tokens, which the refiner samples | A, B2 |
| `pooled.npy` | (32, 256) fp16 | the scorer's input: pass-4 queries max-pooled per candidate | B1 |
| `proposals.npy` | (32, 8, 3) fp32 | the release's 32 candidates, the ones it scores | B1, S0.8 |
| `logits.npy` | (32, 6) fp32 | the release's scorer logits; the last one picks | B1, S0.8 |
| `pick.npy` | int16 | argmax of sigmoid(last logit): the release's choice | S0.8 |
| `lidar2img.npy` | (4, 4, 4) fp32 | the cameras' projections; the model keeps slot 1, CAM_F0 | A, B2 |
| `img_shape.npy`, `ego_status.npy` | (4, 3), (4, 11) fp32 | the rest of what the refiner reads | B2 |

A0's taps, for the rows in `tap_rows.npy` only: `tap_vit_out.npy` (the ViT's own output, before the
projector) and `tap_block12.npy` (block 12 of 24), both (512, 1024) fp16. The rows are all the
select logs' samples plus a seeded 10,000 of the train logs'.

**Batched with batch_fix, held to batch 1; eval mode; a fresh feature dict per forward.** The vendor
mixes samples within a batch (drivejepa_b2d/batch_fix.py; a median 1.49 m off at batch 64 without
the fix, measured in GC04) and slices `lidar2img` inside the dict it is given (drive_jepa_model.py).
The plan cached at batch 1 for that reason. On this shared GPU a batch-1 forward took 158 ms, half
of it the refiner's small kernels, so a batch-1 pass would take 4.6 h. The pass instead runs
batched with batch_fix and refuses to start unless, on 32 real rows, batched equals batch 1: every
pick equal, candidates and logits within 1e-3 (`check_batched`; the manifest records the
differences). GridMask is a no-op in eval mode.

The inputs are built by the vendor's own code -- its windows (`filter_scenes`), `AgentInput` and
feature builder -- in loader workers sharded by log, so a log's pickle is read once. The workers
never build the model. One part is swapped: `fast_bev_feature` replaces the builder's
`_get_bev_feature`, which spends most of a sample normalising four full-size camera images into a
tensor the model never reads; the swap is held to the vendor's exact output before the pass starts
(`check_fast_features`), and the manifest records it.

The pass resumes: rows already written are skipped (`done.npy`), and the staging directory becomes
`--out` only when every row is done.

Two more passes run the backbone alone, each held to an earlier cache before it writes a row:
- `--vit-out-all` (A1 onward): the ViT's own output for every row, held to `tap_vit_out`;
- `--block20` (B3): blocks 1-20's output for B3's rows (`block20_rows`), held to the ViT-out cache
  through the frozen last four blocks (vit_lora.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

#: This file's checkout, for the committed splits: the cwd moves into navsim_v2/ (env.enter).
REPO = Path(__file__).resolve().parents[3]
ARRAYS = {
    "tokens": ((512, 256), np.float16),
    "pooled": ((32, 256), np.float16),
    "proposals": ((32, 8, 3), np.float32),
    "logits": ((32, 6), np.float32),
    "pick": ((), np.int16),
    "lidar2img": ((4, 4, 4), np.float32),
    "img_shape": ((4, 3), np.float32),
    "ego_status": ((4, 11), np.float32),
}
TAPS = {"vit_out": ((512, 1024), np.float16), "block12": ((512, 1024), np.float16)}
#: A0's probe trains on this many train-log samples and reads every select-log sample.
TAP_TRAIN_ROWS = 10_000
TAP_SEED = 0
TAP_BLOCK = 11  # zero-based: the 12th of 24 blocks


def tap_rows(token_logs: list, splits: dict) -> np.ndarray:
    """A0's rows: every select-log sample plus a seeded TAP_TRAIN_ROWS of the train logs' samples."""
    select, train = set(splits["select"]), set(splits["train"])
    in_select = np.array([log in select for log in token_logs])
    in_train = np.flatnonzero([log in train for log in token_logs])
    picked = np.random.default_rng(TAP_SEED).choice(
        in_train, min(TAP_TRAIN_ROWS, len(in_train)), replace=False
    )
    return np.sort(np.concatenate([np.flatnonzero(in_select), picked])).astype(np.int64)


#: Rows on which the fast input path is held to the vendor's before the pass starts.
FAST_CHECK_ROWS = 12
#: The vendor's image pipeline: RandomScaleImageMultiViewImage(scales=[0.4]), then padding to a
#: multiple of 32 (bev_feature_build.py).
SCALE, PAD_DIVISOR = 0.4, 32


def fast_bev_feature(agent_input, training: bool = False) -> dict:
    """`bev_feature_build._get_bev_feature`'s geometry, without its pixel work.

    The vendor's function loads, normalises, rescales and pads all four cameras into a
    `camera_feature` tensor that nothing in the model reads (it is built in bev_feature_build.py and
    read nowhere else). Profiled, that pixel work is most of building an input: 1.3 to 5 s of a
    sample's 1.5 to 5.5 s on this host. The model reads `lidar2img` and `img_shape` from it, and
    both are pure geometry of each camera's calibration and image size. Computed here with the
    vendor's arithmetic, line for line; `check_fast_features` holds the result to the vendor's
    function exactly, on real samples, before a pass uses it.
    """
    lidar2img, shapes = [], []
    cameras = agent_input.cameras[-1]
    for cam in [cameras.cam_b0, cameras.cam_f0, cameras.cam_l0, cameras.cam_r0]:
        if cam.image is None:
            continue
        lidar2cam_r = np.linalg.inv(cam.sensor2lidar_rotation)
        lidar2cam_t = cam.sensor2lidar_translation @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        viewpad = np.eye(4)
        viewpad[: cam.intrinsics.shape[0], : cam.intrinsics.shape[1]] = cam.intrinsics
        scale = np.eye(4)
        scale[0, 0] *= SCALE
        scale[1, 1] *= SCALE
        lidar2img.append(scale @ (viewpad @ lidar2cam_rt.T))
        height, width = int(cam.image.shape[0] * SCALE), int(cam.image.shape[1] * SCALE)
        shapes.append(
            (
                int(np.ceil(height / PAD_DIVISOR)) * PAD_DIVISOR,
                int(np.ceil(width / PAD_DIVISOR)) * PAD_DIVISOR,
                cam.image.shape[2],
            )
        )
    return {
        "img_shape": torch.FloatTensor(np.stack(shapes)),
        "lidar2img": torch.FloatTensor(np.stack(lidar2img)),
    }


def use_fast_geometry():
    """Point the vendor's feature builder at `fast_bev_feature`; return what it pointed at before."""
    from navsim.agents.drive_jepa_perception_based import drive_jepa_features

    original = drive_jepa_features._get_bev_feature
    drive_jepa_features._get_bev_feature = fast_bev_feature
    return original


def check_fast_features(samples: list, navsim_root: Path) -> dict:
    """Every input the model reads is the same, exactly, with the fast path and with the vendor's.

    The only key allowed to differ is `camera_feature`, the tensor the fast path does not build.
    Refuses the pass (SystemExit) on any difference.
    """
    from navsim.agents.drive_jepa_perception_based import drive_jepa_features
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.common.dataclasses import AgentInput

    from gallops.drivejepa_perception import navsim_data

    builder = drive_jepa_features.DriveJEPAFeatureBuilder(config=DriveJEPAConfig())
    sensors = DriveJEPAAgent.get_sensor_config(None)
    problems = []
    for log, token in samples:
        frames = navsim_data.log_windows(log, navsim_root / "navsim_logs/trainval", {token})[token]
        agent_input = AgentInput.from_scene_dict_list(
            frames, navsim_root / "sensor_blobs/trainval", navsim_data.HISTORY, sensors
        )
        vendor = builder.compute_features(agent_input)
        original = use_fast_geometry()
        try:
            fast = builder.compute_features(agent_input)
        finally:
            drive_jepa_features._get_bev_feature = original
        if set(vendor) - set(fast) != {"camera_feature"} or set(fast) - set(vendor):
            problems.append(f"{token}: keys differ: vendor {sorted(vendor)}, fast {sorted(fast)}")
            continue
        for key, value in fast.items():
            if not torch.equal(value, vendor[key]):
                problems.append(f"{token}: `{key}` differs from the vendor's")
    if problems:
        raise SystemExit("the fast input path differs from the vendor's:\n- " + "\n- ".join(problems))
    return {"rows": len(samples), "equal": True}


#: Rows on which the batched forward is held to batch 1 before the pass starts, and its limits.
PARITY_ROWS = 32
PARITY_LIMITS = {"proposals": 1e-3, "logits": 1e-3}


def _features(samples: list, navsim_root: Path) -> list:
    """The model inputs of (log, token) samples, built by the vendor's code (fast geometry on)."""
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder
    from navsim.common.dataclasses import AgentInput

    from gallops.drivejepa_perception import navsim_data

    builder = DriveJEPAFeatureBuilder(config=DriveJEPAConfig())
    sensors = DriveJEPAAgent.get_sensor_config(None)
    out = []
    for log, token in samples:
        frames = navsim_data.log_windows(log, navsim_root / "navsim_logs/trainval", {token})[token]
        agent_input = AgentInput.from_scene_dict_list(
            frames, navsim_root / "sensor_blobs/trainval", navsim_data.HISTORY, sensors
        )
        out.append(builder.compute_features(agent_input))
    return out


def _forward(model, captured: dict, items: list, tap_slot=None) -> list:
    """One forward of (row, features) items as a batch; per-item numpy outputs, taps for tap rows."""
    features = [f for _, f in items]
    keys = [k for k, v in features[0].items() if torch.is_tensor(v)]
    batch = {k: torch.stack([f[k] for f in features]).cuda() for k in keys}  # a fresh dict per forward
    out = model(batch)
    size = len(items)
    if tuple(captured["tokens"].shape) != (1, 512, size, 256):
        raise SystemExit(f"unexpected token shape {tuple(captured['tokens'].shape)}")
    arrays = {
        "tokens": captured["tokens"][0].permute(1, 0, 2).half().cpu().numpy(),
        "pooled": captured["pooled"].half().cpu().numpy(),
        "proposals": out["proposals"].float().cpu().numpy(),
        "logits": out["pred_logit"].float().cpu().numpy(),
        "pick": out["pdm_score"].argmax(1).cpu().numpy(),
    }
    results = [{key: value[b] for key, value in arrays.items()} for b in range(size)]
    tapped = [b for b, (row, _) in enumerate(items) if tap_slot is not None and int(row) in tap_slot]
    if tapped:
        index = torch.as_tensor(tapped, device=captured["vit_out"].device)
        for key in TAPS:
            values = captured[key].reshape(size, 512, 1024).index_select(0, index).half().cpu().numpy()
            for b, value in zip(tapped, values):  # noqa: B905 -- Python 3.9
                results[b][key] = value
    return results


def check_batched(model, captured: dict, items: list, batch: int) -> dict:
    """The batched forward (batch_fix on) against batch 1 on real rows; refuses on any material change.

    The plan cached the release's outputs at batch 1 because the vendor mixes samples within a batch.
    batch_fix removes that mixing (H14 measured 4.6e-5 m against batch 1, every pick equal); at batch
    1 on this shared GPU a forward took 158 ms, half of it the refiner's small kernels, so the pass
    runs batched and holds itself to batch 1 here: every pick equal, candidates and logits within
    PARITY_LIMITS. The measured differences go into the manifest.
    """
    with torch.no_grad():
        ones = [_forward(model, captured, [item])[0] for item in items]
        many = []
        for i in range(0, len(items), batch):
            many += _forward(model, captured, items[i : i + batch])
    worst = {
        key: max(
            float(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64)).max())
            for a, b in zip(ones, many)  # noqa: B905 -- Python 3.9
        )
        for key in ("tokens", "pooled", "proposals", "logits")
    }
    picks = sum(int(a["pick"] == b["pick"]) for a, b in zip(ones, many))  # noqa: B905 -- Python 3.9
    problems = [
        f"{key} differ by {worst[key]:.3g} (limit {limit})"
        for key, limit in PARITY_LIMITS.items()
        if worst[key] > limit
    ]
    if picks != len(items):
        problems.append(f"the pick differs on {len(items) - picks} of {len(items)} rows")
    if problems:
        raise SystemExit("the batched forward differs from batch 1:\n- " + "\n- ".join(problems))
    return {"rows": len(items), "batch": batch, "max_abs_diff": worst, "picks_equal": picks}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cfg, labels_dir: Path, out: Path, workers: int, limit: int, batch: int) -> None:
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env, navsim_data

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    sha = git_sha()
    entered = env.enter(cfg)
    labels = np.load(Path(labels_dir) / "labels.npz")
    lmanifest = json.loads((Path(labels_dir) / "manifest.json").read_text())
    logs = lmanifest["logs"]
    token_logs = [logs[i] for i in labels["log_id"]]
    tokens = [str(t) for t in labels["token"]]
    n = len(tokens) if not limit else min(limit, len(tokens))
    splits = navsim_data.load_splits(
        REPO / "recipes/drivejepa_perception/splits.json",
        navsim_data.navtrain_filter(Path(entered["code"]))["log_names"],
    )
    taps = tap_rows(token_logs[:n], splits)

    staging = out.with_name(out.name + ".staging")
    fresh = not staging.exists()
    staging.mkdir(parents=True, exist_ok=True)
    mode = "w+" if fresh else "r+"
    mm = {
        key: np.lib.format.open_memmap(staging / f"{key}.npy", mode=mode, dtype=dtype, shape=(n, *shape))
        for key, (shape, dtype) in ARRAYS.items()
    }
    tap_mm = {
        key: np.lib.format.open_memmap(
            staging / f"tap_{key}.npy", mode=mode, dtype=dtype, shape=(len(taps), *shape)
        )
        for key, (shape, dtype) in TAPS.items()
    }
    np.save(staging / "tap_rows.npy", taps)
    tap_slot = {int(row): i for i, row in enumerate(taps)}
    done_path = staging / "done.npy"
    done = np.load(done_path) if done_path.exists() else np.zeros(n, dtype=bool)
    print(f"{n} rows, {int(done.sum())} already done, {len(taps)} tap rows", flush=True)

    # The fast input path must equal the vendor's on real samples before any row is built with it.
    fast_check = check_fast_features(
        [(token_logs[r], tokens[r]) for r in np.linspace(0, n - 1, FAST_CHECK_ROWS).astype(int)],
        Path(entered["navsim_root"]),
    )
    print(f"fast inputs equal the vendor's on {fast_check['rows']} rows", flush=True)

    # The loader first, so its workers fork from a process that holds no model and no CUDA context.
    plan: dict = {}
    for row in range(n):
        if not done[row]:
            plan.setdefault(token_logs[row], []).append((row, tokens[row]))
    dataset = _Samples(
        sorted(plan.items()),
        Path(entered["navsim_root"]) / "navsim_logs/trainval",
        Path(entered["navsim_root"]) / "sensor_blobs/trainval",
    )
    loader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers, prefetch_factor=4)
        if workers
        else dataset
    )

    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    from gallops.drivejepa_b2d import batch_fix

    batch_fix.apply_all()  # the refiner mixes samples within a batch without it (batch_fix.py)
    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.cuda().eval()
    captured: dict = {}
    model._backbone.register_forward_hook(lambda m, i, o: captured.__setitem__("tokens", o[0]))
    model._backbone.img_backbone.register_forward_hook(lambda m, i, o: captured.__setitem__("vit_out", o))
    model.scorer.pred_score.register_forward_pre_hook(lambda m, i: captured.__setitem__("pooled", i[0]))
    blocks = [module for name, module in model.named_modules() if name.endswith(f"blocks.{TAP_BLOCK}")]
    if len(blocks) != 1:
        raise SystemExit(f"expected one ViT block {TAP_BLOCK}, found {len(blocks)}")
    blocks[0].register_forward_hook(lambda m, i, o: captured.__setitem__("block12", o))

    # Batched with batch_fix must equal batch 1 on real rows before any row is written batched.
    use_fast_geometry()
    parity_rows = np.linspace(0, n - 1, PARITY_ROWS).astype(int)
    parity = check_batched(
        model,
        captured,
        list(
            zip(  # noqa: B905 -- Python 3.9
                parity_rows,
                _features([(token_logs[r], tokens[r]) for r in parity_rows], Path(entered["navsim_root"])),
            )
        ),
        batch,
    )
    print(f"batch {batch} equals batch 1 on {parity['rows']} rows: {parity['max_abs_diff']}", flush=True)

    t0, written, pending = time.time(), 0, []

    def write(items):
        for (row, features), result in zip(items, _forward(model, captured, items, tap_slot)):  # noqa: B905
            for key in ("tokens", "pooled", "proposals", "logits", "pick"):
                mm[key][row] = result[key]
            for key in ("lidar2img", "img_shape", "ego_status"):
                mm[key][row] = features[key].float().numpy()
            for key in TAPS:
                if key in result:
                    tap_mm[key][tap_slot[int(row)]] = result[key]
            done[row] = True
        return len(items)

    with torch.no_grad():
        for item in loader:
            pending.append(item)
            if len(pending) < batch:
                continue
            before, written = written, written + write(pending)
            pending = []
            if written // 2000 > before // 2000:
                for array in [*mm.values(), *tap_mm.values()]:
                    array.flush()
                np.save(done_path, done)
                rate = written / (time.time() - t0)
                print(
                    f"{int(done.sum())}/{n} rows, {rate:.1f}/s, eta {(n - done.sum()) / rate / 60:.0f} min",
                    flush=True,
                )
        if pending:
            written += write(pending)
    for array in [*mm.values(), *tap_mm.values()]:
        array.flush()
    np.save(done_path, done)
    if not done.all():
        raise SystemExit(f"{int((~done).sum())} rows not written; rerun to resume")

    manifest = {
        "pass": "tokens",
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "release_sha256": _sha256(Path(entered["release"])),
        "labels": str(labels_dir),
        "rows": n,
        "arrays": {key: [list(shape), np.dtype(dtype).name] for key, (shape, dtype) in ARRAYS.items()},
        "taps": {key: [list(shape), np.dtype(dtype).name] for key, (shape, dtype) in TAPS.items()},
        "tap_rows": int(len(taps)),
        "tap_rule": f"select-log rows + default_rng({TAP_SEED}) choice of {TAP_TRAIN_ROWS} train-log rows",
        "batch": batch,
        "batch_fix": True,
        "batched_equals_batch1": parity,
        "mode": "eval",
        "inputs": "the vendor's builder with `fast_bev_feature` for its unread 4-camera tensor",
        "fast_inputs_check": fast_check,
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "entered"}, indent=1))


#: The ViT-out pass is held to the tap captured inside the full forward on these many tap rows.
VIT_PARITY_ROWS = 32
VIT_LIMITS = {"max_abs": 0.1, "mean_abs": 1e-3}


def run_vit_out(cfg, labels_dir: Path, tokens_dir: Path, out: Path, workers: int, batch: int) -> None:
    """The tap A0 chose -- the ViT's own output, before the projector -- for every row (A1 onward).

    Only the backbone runs. Its input is assembled as DriveJEPAModel.forward assembles it (both frames
    ImageNet-normalised, previous frame first). Before any row is written, the result is held to the
    `tap_vit_out` the full forward captured in `tokens_dir` on VIT_PARITY_ROWS of its rows.
    """
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started, sha = time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha()
    entered = env.enter(cfg)
    labels = np.load(Path(labels_dir) / "labels.npz")
    logs = json.loads((Path(labels_dir) / "manifest.json").read_text())["logs"]
    token_logs = [logs[i] for i in labels["log_id"]]
    tokens = [str(t) for t in labels["token"]]
    n = len(tokens)
    staging = out.with_name(out.name + ".staging")
    fresh = not staging.exists()
    staging.mkdir(parents=True, exist_ok=True)
    vit = np.lib.format.open_memmap(
        staging / "vit_out.npy", mode="w+" if fresh else "r+", dtype=np.float16, shape=(n, 512, 1024)
    )
    done_path = staging / "done.npy"
    done = np.load(done_path) if done_path.exists() else np.zeros(n, dtype=bool)
    plan: dict = {}
    for row in range(n):
        if not done[row]:
            plan.setdefault(token_logs[row], []).append((row, tokens[row]))
    dataset = _Samples(
        sorted(plan.items()),
        Path(entered["navsim_root"]) / "navsim_logs/trainval",
        Path(entered["navsim_root"]) / "sensor_blobs/trainval",
    )
    loader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers, prefetch_factor=4)
    )

    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.cuda().eval()
    captured: dict = {}
    model._backbone.img_backbone.register_forward_hook(lambda m, i, o: captured.__setitem__("vit_out", o))

    def vit_of(features: list) -> np.ndarray:
        cam_1 = torch.stack([f["camera_feature_1"] for f in features]).cuda()
        cam_2 = torch.stack([f["camera_feature_2"] for f in features]).cuda()
        camera = torch.cat([model.transform(cam_2)[:, None], model.transform(cam_1)[:, None]], dim=1)
        model._backbone(camera, img_metas={})
        return captured["vit_out"].reshape(len(features), 512, 1024).half().cpu().numpy()

    use_fast_geometry()
    tap_rows = np.load(Path(tokens_dir) / "tap_rows.npy")
    reference = np.load(Path(tokens_dir) / "tap_vit_out.npy", mmap_mode="r")
    picked = np.linspace(0, len(tap_rows) - 1, VIT_PARITY_ROWS).astype(int)
    feats = _features(
        [(token_logs[int(tap_rows[i])], tokens[int(tap_rows[i])]) for i in picked],
        Path(entered["navsim_root"]),
    )
    with torch.no_grad():
        mine = np.concatenate([vit_of(feats[i : i + batch]) for i in range(0, len(feats), batch)]).astype(
            np.float32
        )
    diff = np.abs(mine - np.asarray(reference[picked], dtype=np.float32))
    parity = {"rows": int(len(picked)), "max_abs": float(diff.max()), "mean_abs": float(diff.mean())}
    if parity["max_abs"] > VIT_LIMITS["max_abs"] or parity["mean_abs"] > VIT_LIMITS["mean_abs"]:
        raise SystemExit(f"the ViT-out pass differs from the full forward's tap: {parity}")
    print(f"ViT-out equals the full forward's tap on {parity['rows']} rows: {parity}", flush=True)

    t0, written, pending = time.time(), 0, []

    def write(items):
        for (row, _), value in zip(items, vit_of([f for _, f in items])):  # noqa: B905 -- Python 3.9
            vit[row] = value
            done[row] = True
        return len(items)

    with torch.no_grad():
        for item in loader:
            pending.append(item)
            if len(pending) < batch:
                continue
            before, written = written, written + write(pending)
            pending = []
            if written // 4000 > before // 4000:
                vit.flush()
                np.save(done_path, done)
                rate = written / (time.time() - t0)
                print(
                    f"{int(done.sum())}/{n} rows, {rate:.1f}/s, eta {(n - done.sum()) / rate / 60:.0f} min",
                    flush=True,
                )
        if pending:
            written += write(pending)
    vit.flush()
    np.save(done_path, done)
    if not done.all():
        raise SystemExit(f"{int((~done).sum())} rows not written; rerun to resume")
    manifest = {
        "pass": "vit_out",
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "labels": str(labels_dir),
        "reference_tokens": str(tokens_dir),
        "rows": n,
        "array": [[512, 1024], "float16"],
        "batch": batch,
        "parity_with_full_forward_tap": parity,
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "entered"}, indent=1))


#: B3's cache rows: a seeded draw of train rows, then every select and report row (vit_lora.py).
B3_TRAIN_ROWS, B3_SEED = 20_000, 0


def block20_rows(labels_dir: Path) -> np.ndarray:
    """B3's rows, in cache order: B3_TRAIN_ROWS train rows drawn with B3_SEED, then select and report."""
    labels = np.load(Path(labels_dir) / "labels.npz")
    logs = json.loads((Path(labels_dir) / "manifest.json").read_text())["logs"]
    token_logs = np.array([logs[i] for i in labels["log_id"]])
    splits = json.loads((REPO / "recipes/drivejepa_perception/splits.json").read_text())
    train = np.flatnonzero(np.isin(token_logs, splits["train"]))
    picked = np.sort(np.random.default_rng(B3_SEED).choice(train, B3_TRAIN_ROWS, replace=False))
    held = np.flatnonzero(np.isin(token_logs, splits["select"] + splits["report"]))
    return np.concatenate([picked, held])


def run_block20(cfg, labels_dir: Path, vit_dir: Path, out: Path, workers: int, batch: int) -> None:
    """B3's cache: the output of the ViT's first vit_lora.BRANCH_FROM blocks, for `block20_rows`.

    Only the backbone runs, as in `run_vit_out`. Before any row is written, the frozen ViT's last
    blocks and norm (`vit_lora.LateBranch` without LoRA), run on this pass's fp16 output, are held to
    the ViT-out cache in `vit_dir` on VIT_PARITY_ROWS rows, within VIT_LIMITS: the cache plus the
    branch is the tap A1 trained on.
    """
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env, vit_lora

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started, sha = time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha()
    entered = env.enter(cfg)
    labels = np.load(Path(labels_dir) / "labels.npz")
    logs = json.loads((Path(labels_dir) / "manifest.json").read_text())["logs"]
    token_logs = [logs[i] for i in labels["log_id"]]
    tokens = [str(t) for t in labels["token"]]
    rows = block20_rows(labels_dir)
    n = len(rows)
    staging = out.with_name(out.name + ".staging")
    fresh = not staging.exists()
    staging.mkdir(parents=True, exist_ok=True)
    np.save(staging / "rows.npy", rows)
    cache = np.lib.format.open_memmap(
        staging / "block20.npy", mode="w+" if fresh else "r+", dtype=np.float16, shape=(n, 512, 1024)
    )
    done_path = staging / "done.npy"
    done = np.load(done_path) if done_path.exists() else np.zeros(n, dtype=bool)
    plan: dict = {}
    for slot, row in enumerate(rows):
        if not done[slot]:
            plan.setdefault(token_logs[row], []).append((slot, tokens[row]))
    dataset = _Samples(
        sorted(plan.items()),
        Path(entered["navsim_root"]) / "navsim_logs/trainval",
        Path(entered["navsim_root"]) / "sensor_blobs/trainval",
    )
    loader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers, prefetch_factor=4)
    )

    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.cuda().eval()
    vit = model._backbone.img_backbone
    captured: dict = {}
    vit.blocks[vit_lora.BRANCH_FROM - 1].register_forward_hook(lambda m, i, o: captured.__setitem__("x", o))

    def block20_of(features: list) -> np.ndarray:
        cam_1 = torch.stack([f["camera_feature_1"] for f in features]).cuda()
        cam_2 = torch.stack([f["camera_feature_2"] for f in features]).cuda()
        camera = torch.cat([model.transform(cam_2)[:, None], model.transform(cam_1)[:, None]], dim=1)
        model._backbone(camera, img_metas={})
        return captured["x"].reshape(len(features), 512, 1024).half().cpu().numpy()

    use_fast_geometry()
    reference = np.load(Path(vit_dir) / "vit_out.npy", mmap_mode="r")
    picked = np.linspace(0, n - 1, VIT_PARITY_ROWS).astype(int)
    feats = _features(
        [(token_logs[int(rows[i])], tokens[int(rows[i])]) for i in picked], Path(entered["navsim_root"])
    )
    branch = vit_lora.LateBranch(vit, lora=False).cuda().eval()
    with torch.no_grad():
        mine = np.concatenate([block20_of(feats[i : i + batch]) for i in range(0, len(feats), batch)])
        tail = branch(torch.from_numpy(mine).cuda().float()).cpu().numpy()
    diff = np.abs(tail - np.asarray(reference[rows[picked]], dtype=np.float32))
    parity = {"rows": int(len(picked)), "max_abs": float(diff.max()), "mean_abs": float(diff.mean())}
    if parity["max_abs"] > VIT_LIMITS["max_abs"] or parity["mean_abs"] > VIT_LIMITS["mean_abs"]:
        raise SystemExit(f"the frozen tail on this cache differs from the ViT-out cache: {parity}")
    print(
        f"frozen tail on the block-20 cache equals the ViT-out cache on {parity['rows']} rows: {parity}",
        flush=True,
    )

    t0, written, pending = time.time(), 0, []

    def write(items):
        for (slot, _), value in zip(items, block20_of([f for _, f in items])):  # noqa: B905 -- Python 3.9
            cache[slot] = value
            done[slot] = True
        return len(items)

    with torch.no_grad():
        for item in loader:
            pending.append(item)
            if len(pending) < batch:
                continue
            before, written = written, written + write(pending)
            pending = []
            if written // 4000 > before // 4000:
                cache.flush()
                np.save(done_path, done)
                rate = written / (time.time() - t0)
                print(
                    f"{int(done.sum())}/{n} rows, {rate:.1f}/s, eta {(n - done.sum()) / rate / 60:.0f} min",
                    flush=True,
                )
        if pending:
            written += write(pending)
    cache.flush()
    np.save(done_path, done)
    if not done.all():
        raise SystemExit(f"{int((~done).sum())} rows not written; rerun to resume")
    manifest = {
        "pass": "block20",
        "branch_from": vit_lora.BRANCH_FROM,
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "labels": str(labels_dir),
        "reference_vit_out": str(vit_dir),
        "rows": n,
        "train_rows": B3_TRAIN_ROWS,
        "train_seed": B3_SEED,
        "array": [[512, 1024], "float16"],
        "batch": batch,
        "parity_frozen_tail_with_vit_out": parity,
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)


class _Samples(torch.utils.data.IterableDataset):
    """(row, features) for the planned rows, built by the vendor's code; workers shard by log."""

    def __init__(self, plan: list, logs_dir: Path, blobs_dir: Path):
        self.plan, self.logs_dir, self.blobs_dir = plan, logs_dir, blobs_dir

    def __iter__(self):
        from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
        from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
        from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder
        from navsim.common.dataclasses import AgentInput

        from gallops.drivejepa_perception import navsim_data

        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        torch.set_num_threads(1)  # 24 workers x torch's default 128 threads only slowed each one down
        use_fast_geometry()
        builder = DriveJEPAFeatureBuilder(config=DriveJEPAConfig())
        sensors = DriveJEPAAgent.get_sensor_config(None)  # uses no instance state; imported, not restated
        for i, (log, rows) in enumerate(self.plan):
            if i % workers != worker:
                continue
            windows = navsim_data.log_windows(log, self.logs_dir, {token for _, token in rows})
            for row, token in rows:
                agent_input = AgentInput.from_scene_dict_list(
                    windows[token],
                    self.blobs_dir,
                    num_history_frames=navsim_data.HISTORY,
                    sensor_config=sensors,
                )
                yield row, builder.compute_features(agent_input)


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.tokens")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="the first N rows only (smoke runs)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--vit-out-all", action="store_true", help="only the ViT's own output, every row (A1)"
    )
    parser.add_argument("--tokens", default="", help="--vit-out-all: the token pass whose taps it is held to")
    parser.add_argument("--block20", action="store_true", help="B3: blocks 1-20's output, B3's rows only")
    parser.add_argument("--vit-dir", default="", help="--block20: the ViT-out pass it is held to")
    args = parser.parse_args(argv)
    if args.block20:
        run_block20(
            SiteConfig.load(), Path(args.labels), Path(args.vit_dir), Path(args.out), args.workers, args.batch
        )
        return
    if args.vit_out_all:
        run_vit_out(
            SiteConfig.load(), Path(args.labels), Path(args.tokens), Path(args.out), args.workers, args.batch
        )
    else:
        run(SiteConfig.load(), Path(args.labels), Path(args.out), args.workers, args.limit, args.batch)


if __name__ == "__main__":
    main()
