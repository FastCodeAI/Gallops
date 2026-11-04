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

**Batch 1, eval mode, a fresh feature dict per forward.** The vendor mixes samples within a batch
(drivejepa_b2d/batch_fix.py; a median 1.49 m off at batch 64, measured in GC04) and slices
`lidar2img` inside the dict it is given (drive_jepa_model.py). GridMask is a no-op in eval mode.

The inputs are built by the vendor's own code -- its windows (`filter_scenes`), `AgentInput` and
feature builder -- in loader workers sharded by log, so a log's pickle is read once. The workers
never build the model. The pass resumes: rows already written are skipped (`done.npy`), and the
staging directory becomes `--out` only when every row is done.
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cfg, labels_dir: Path, out: Path, workers: int, limit: int) -> None:
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

    t0, written = time.time(), 0
    with torch.no_grad():
        for row, features in loader:
            batch = {k: v.unsqueeze(0).cuda() for k, v in features.items() if torch.is_tensor(v)}
            output = model(batch)
            tok = captured["tokens"]
            if tok.shape != (1, 512, 1, 256):
                raise SystemExit(f"unexpected token shape {tuple(tok.shape)}")
            mm["tokens"][row] = tok[0, :, 0].half().cpu().numpy()
            mm["pooled"][row] = captured["pooled"][0].half().cpu().numpy()
            mm["proposals"][row] = output["proposals"][0].float().cpu().numpy()
            mm["logits"][row] = output["pred_logit"][0].float().cpu().numpy()
            mm["pick"][row] = int(output["pdm_score"][0].argmax())
            for key in ("lidar2img", "img_shape", "ego_status"):
                mm[key][row] = features[key].float().numpy()
            if row in tap_slot:
                for key in TAPS:
                    value = captured[key]
                    value = value[0] if value.dim() == 3 else value
                    tap_mm[key][tap_slot[row]] = value.reshape(512, 1024).half().cpu().numpy()
            done[row] = True
            written += 1
            if written % 2000 == 0:
                for array in [*mm.values(), *tap_mm.values()]:
                    array.flush()
                np.save(done_path, done)
                rate = written / (time.time() - t0)
                print(
                    f"{int(done.sum())}/{n} rows, {rate:.1f}/s, eta {(n - done.sum()) / rate / 60:.0f} min",
                    flush=True,
                )
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
        "batch": 1,
        "mode": "eval",
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "entered"}, indent=1))


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
    args = parser.parse_args(argv)
    run(SiteConfig.load(), Path(args.labels), Path(args.out), args.workers, args.limit)


if __name__ == "__main__":
    main()
