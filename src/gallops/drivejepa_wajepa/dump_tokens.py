"""Stage dump-tokens: the frozen WA-JEPA token cache both runs train from, one pass over navtrain.

Per scene, <out>/<log>/<token>/{drivejepa4v_feature,pad_target,wajepa_tokens}.gz with navsim's
dump_feature_target_to_pickle, each written as .tmp then renamed; scenes with all three files are skipped
(resumable), nothing is overwritten.

Scenes: NAVSIM's navtrain scene filter (navtrain.yaml, instantiated as run_dataset_caching.py does), applied
per log. Images: the NAVSIM sensor blobs, with --current1 laid over them (navtrain_current_1, the split
Drive-JEPA's download script skips; cameras b0 f0 l0 r0 only): a file present there is read from there.
Features: DriveJEPA4VFeatureBuilder (lidar2img, img_shape, ego_status; checked equal to v2's builder), the
unchanged DriveJEPATargetBuilder (pad_target), and encode_wajepa (frozen WA-JEPA, bf16 autocast; checked
identical to WA-JEPA's own encoder under bf16) batched on the GPU, 16 scenes (64 clips) per call; each
scene's tokens are cloned to their own CPU storage before pickling (a slice of the batch tensor pickles the
whole batch, 62 MB a scene).

Runs in Drive-JEPA's interpreter with cwd = the run's tree (code-run2) navsim_v2/, WAJEPA_CODE and
WAJEPA_WEIGHTS set. The 16 Oct dump: 103,288 of 103,288 navtrain scenes, 0 errors, 5.43 h, 3.9 MB a scene.

Ported from paddock-ops wajepa-runs/dump_wajepa.py. Changed: the arguments (the overlay root was a
machine-path default), and the exit code, which is now non-zero when any scene failed.

    python -m gallops.drivejepa_wajepa.dump_tokens --navtrain <navtrain.yaml> --data <navsim root>
        --current1 <current_split_1> --out <cache> --workers 8 [--max-logs N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

NAMES = ("drivejepa4v_feature.gz", "pad_target.gz", "wajepa_tokens.gz")
BATCH = 16
SENSOR_CAMERAS = dict(cam_f0=[2, 3], cam_l0=[2, 3], cam_l1=[], cam_l2=[], cam_r0=[2, 3], cam_r1=[], cam_r2=[],
                      cam_b0=[2, 3], lidar_pc=[])  # fmt: skip


class OverlayRoot:
    """sensor_blobs_path: `root / data_path` is the --current1 file if present, else the NAVSIM one (navsim
    only joins with `/`: Cameras.from_camera_dict, Lidar.from_paths)."""

    def __init__(self, nas, current1):
        self.nas, self.current1 = Path(nas), Path(current1)

    def __truediv__(self, rel):
        local = self.current1 / rel
        return local if local.is_file() else self.nas / rel


class Scenes(IterableDataset):
    def __init__(self, filter_cfg, logs, data, current1, out):
        self.filter_cfg, self.logs, self.data, self.current1, self.out = filter_cfg, logs, data, current1, out

    def __iter__(self):
        from navsim.agents.drive_jepa_perception_based.bevformer.wajepa import clips_from_agent_input
        from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
        from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
            DriveJEPA4VFeatureBuilder,
            DriveJEPATargetBuilder,
        )
        from navsim.common.dataclasses import SensorConfig
        from navsim.common.dataloader import SceneLoader

        wi = get_worker_info()
        logs = self.logs[wi.id :: wi.num_workers] if wi else self.logs
        cfg = DriveJEPAConfig()
        light, target = DriveJEPA4VFeatureBuilder(cfg), DriveJEPATargetBuilder(cfg)
        sensor = SensorConfig(**SENSOR_CAMERAS)
        for log in logs:
            out = Path(self.out) / log
            done = {
                t
                for t in (os.listdir(out) if out.is_dir() else [])
                if all((out / t / n).is_file() for n in NAMES)
            }
            sf = instantiate(self.filter_cfg)  # NAVSIM's navtrain scene filter, restricted to this log
            sf.log_names = [log]
            try:
                loader = SceneLoader(
                    data_path=Path(self.data) / "navsim_logs/trainval",
                    sensor_blobs_path=OverlayRoot(Path(self.data) / "sensor_blobs/trainval", self.current1),
                    synthetic_scenes_path=None,
                    scene_filter=sf,
                    sensor_config=sensor,
                )
            except Exception:  # noqa: BLE001
                yield {"error": f"{log} loader\n{traceback.format_exc()}"}
                continue
            yield {"log_tokens": (log, len(loader.tokens), len(done))}
            for tok in loader.tokens:
                if tok in done:
                    continue
                try:
                    scene = loader.get_scene_from_token(tok)
                    ai = scene.get_agent_input()
                    yield {
                        "log": log,
                        "token": tok,
                        "light": light.compute_features(ai),
                        "target": target.compute_targets(scene),
                        "clips": clips_from_agent_input(ai),
                    }
                except Exception:  # noqa: BLE001
                    yield {"error": f"{log} {tok}\n{traceback.format_exc()}"}


def write(path, data):
    from navsim.planning.training.dataset import dump_feature_target_to_pickle

    tmp = Path(str(path) + ".tmp")
    dump_feature_target_to_pickle(tmp, data)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--navtrain", required=True, help="navsim's navtrain scene-filter yaml")
    ap.add_argument("--data", required=True, help="the NAVSIM root (navsim_logs/, sensor_blobs/)")
    ap.add_argument("--current1", required=True, help="navtrain_current_1's sensor blobs, laid over --data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-logs", type=int, default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())

    from navsim.agents.drive_jepa_perception_based.bevformer.wajepa import (
        build_wajepa_encoder,
        encode_clips,
        split_scenes,
    )

    filter_cfg = OmegaConf.load(a.navtrain)
    logs = sorted(filter_cfg.log_names)[: a.max_logs]
    print(
        f"navtrain filter: {len(filter_cfg.log_names)} logs, {len(filter_cfg.tokens)} tokens; "
        f"dumping {len(logs)} logs to {a.out}",
        flush=True,
    )
    enc, msg = build_wajepa_encoder("cuda")
    assert not msg.missing_keys and not msg.unexpected_keys
    dl = DataLoader(
        Scenes(filter_cfg, logs, a.data, a.current1, a.out),
        batch_size=None,
        num_workers=a.workers,
        prefetch_factor=4,
        collate_fn=lambda x: x,
        persistent_workers=False,
    )
    writer = ThreadPoolExecutor(4)
    pending, buf = [], []
    n_done = n_err = n_total = n_skip = 0
    t0, last = time.time(), time.time()

    def flush():
        nonlocal n_done
        clips = torch.cat([b["clips"] for b in buf])
        toks = split_scenes(encode_clips(enc, clips).cpu(), len(buf))
        for b, t in zip(buf, toks):  # noqa: B905 -- Python 3.9
            d = Path(a.out) / b["log"] / b["token"]
            d.mkdir(parents=True, exist_ok=True)
            pending.append(writer.submit(write, d / NAMES[0], b["light"]))
            pending.append(writer.submit(write, d / NAMES[1], b["target"]))
            pending.append(writer.submit(write, d / NAMES[2], {"wajepa_tokens": t}))
        n_done += len(buf)
        buf.clear()
        while len(pending) > 96:  # keep the write queue bounded
            pending.pop(0).result()

    for item in dl:
        if "error" in item:
            n_err += 1
            print("ERROR", item["error"], flush=True)
            continue
        if "log_tokens" in item:
            log, n, k = item["log_tokens"]
            n_total += n
            n_skip += k
            continue
        buf.append(item)
        if len(buf) == BATCH:
            flush()
        if time.time() - last > 120:
            last = time.time()
            rate = n_done / (last - t0)
            print(
                f"{time.strftime('%H:%M:%S')} written {n_done} (skipped {n_skip}, errors {n_err}) "
                f"of {n_total} seen so far, {rate:.2f} scenes/s, "
                f"{rate * 3600 * 3.95 / 1000:.1f} GB/h of tokens",
                flush=True,
            )
    if buf:
        flush()
    for p in pending:
        p.result()
    writer.shutdown()
    print(
        f"DUMP_DONE written {n_done} skipped {n_skip} errors {n_err} navtrain scenes seen {n_total} "
        f"in {(time.time() - t0) / 3600:.2f} h",
        flush=True,
    )
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
