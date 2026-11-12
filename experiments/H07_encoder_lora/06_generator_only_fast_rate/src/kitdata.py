"""Training / picking data for phase 1, built once.  CPU + NAS reads only.

    ./run.sh src/kitdata.py build     # scene_meta.pt + moments.pt into DATA, then the feature gate

Per scene (482 train + 50 picking), with the evaluation wrapper's own code:
  - KITScenesSceneAdapter(path) estimates the ground height from one lidar sweep (ADP:208-230).
  - One full build_sample at the scene's first moment, through the upstream feature builder
    (DriveJEPAFeatureBuilder.compute_features, unchanged), gives lidar2img and img_shape, the only
    calibration features the model reads (drive_jepa_model.py:50; bevformer/encoder.py:123,141-142).
Per moment (05_generator_survey out/d_moments.json: t % 5 == 0, t >= 15, t + 40 <= N - 1):
  - build_sample(t, num_future_frames=8, camera_frames=(), command_mode="gt"): the same poses,
    ego statuses, command and future_poses as evaluation, without loading any image.
  - ego_status is assembled exactly as compute_features does (drive_jepa_features.py:72-84).
Images come from the s2 frame cache (bit-identical to the baseline image path, s2 gate A).

Gate (after the build): on the 103 baseline picking moments, the assembled features
(frame cache images + per-scene calibration + ego status) and the human path are bit-identical
to the saved selection features built through the full wrapper path (s2 c_reference.py).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

import cache as K           # s2 frame cache
import common as C          # s1
import evalsets as E        # s2
import splits as S          # s1

ADP = C.kit_adapter_module()
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "out"
DATA = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s1")
S0OUT = HERE.parent / "05_generator_survey" / "out"
N_FUT = 8
_to_tensor = transforms.ToTensor()


def builder():
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder
    return DriveJEPAFeatureBuilder(DriveJEPAConfig())


def ego_feature(agent_input) -> torch.Tensor:
    """drive_jepa_features.py:72-84, verbatim logic."""
    rows = []
    for es in agent_input.ego_statuses:
        if es is None:
            continue
        rows.append(torch.cat([torch.tensor(es.ego_pose, dtype=torch.float32),
                               torch.tensor(es.ego_velocity, dtype=torch.float32),
                               torch.tensor(es.ego_acceleration, dtype=torch.float32),
                               torch.tensor(es.driving_command, dtype=torch.float32)], dim=-1))
    return torch.stack(rows)


def scene_job(args):
    sid, path, ts, check_last = args
    ad = ADP.KITScenesSceneAdapter(Path(path))
    b = builder()
    full = b.compute_features(ad.build_sample(ts[0], num_future_frames=N_FUT, command_mode="gt").agent_input)
    meta = {"lidar2img": full["lidar2img"], "img_shape": full["img_shape"], "ground_offset": ad.ground_offset,
            "frames": ad.num_frames}
    if check_last and len(ts) > 1:
        last = b.compute_features(ad.build_sample(ts[-1], num_future_frames=N_FUT, command_mode="gt").agent_input)
        meta["same_at_last_moment"] = bool(torch.equal(last["lidar2img"], full["lidar2img"])
                                           and torch.equal(last["img_shape"], full["img_shape"]))
    moms = []
    for t in ts:
        smp = ad.build_sample(t, num_future_frames=N_FUT, camera_frames=(), command_mode="gt")
        moms.append({"scene_id": sid, "t": int(t), "ego_status": ego_feature(smp.agent_input),
                     "future_poses": smp.future_poses, "future_interval_s": smp.future_interval_s,
                     "command": smp.agent_input.ego_statuses[-1].driving_command.copy()})
    return sid, meta, moms


def assemble(m: dict, meta: dict, fc) -> dict:
    """Model features for one moment: camera_feature_1 = newest (t), camera_feature_2 = t-5
    (drive_jepa_features.py:46-57: cameras[-1] and cameras[-2], history stride 5)."""
    return {"camera_feature_1": _to_tensor(fc.frame(m["scene_id"], m["t"])),
            "camera_feature_2": _to_tensor(fc.frame(m["scene_id"], m["t"] - 5)),
            "ego_status": m["ego_status"], "lidar2img": meta["lidar2img"], "img_shape": meta["img_shape"]}


def build() -> None:
    t0 = time.time()
    data = S.load_splits()
    dm = json.loads((S0OUT / "d_moments.json").read_text())
    jobs = [(s, data["scene_paths"][s], dm[split][s], split == "selection")
            for split in ("train", "selection") for s in sorted(dm[split])]
    assert len(jobs) == 532
    meta, moments = {}, {"train": [], "selection": []}
    split_of = {s: sp for sp in ("train", "selection") for s in dm[sp]}
    with ThreadPoolExecutor(16) as pool:
        for i, (sid, mt, ms) in enumerate(pool.map(scene_job, jobs)):
            meta[sid] = mt
            moments[split_of[sid]].extend(ms)
            if i % 50 == 0:
                print(f"{i + 1}/532 scenes {time.time() - t0:.0f}s", flush=True)
    torch.save(meta, DATA / "scene_meta.pt")
    torch.save(moments, DATA / "moments.pt")
    out = {"scenes": len(meta), "train_moments": len(moments["train"]), "picking_moments": len(moments["selection"]),
           "picking_scenes_calibration_same_at_first_and_last_moment":
               [sum(m.get("same_at_last_moment", True) for s, m in meta.items() if split_of[s] == "selection"),
                sum(split_of[s] == "selection" for s in meta)],
           "img_shape_values": sorted({str(m["img_shape"].tolist()) for m in meta.values()}),
           "build_s": round(time.time() - t0, 1)}
    out["gate"] = gate(meta, moments)
    (OUT / "data_build.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str))


def gate(meta, moments) -> dict:
    """The 103 baseline moments: assembled == saved full-path features, bit for bit."""
    sel = E.load_selection_features()
    by = {(m["scene_id"], m["t"]): m for m in moments["selection"]}
    fc = K.FrameCache()
    feats_eq, human_eq, missing = 0, 0, 0
    lidar_const = {}
    for e in sel:
        k = (e["scene_id"], e["keyframe"])
        if k not in by:
            missing += 1
            continue
        a = assemble(by[k], meta[k[0]], fc)
        feats_eq += int(all(torch.equal(a[x], e["features"][x]) for x in a))
        human_eq += int(np.array_equal(by[k]["future_poses"], e["future_poses"]))
        lidar_const.setdefault(k[0], []).append(e["features"]["lidar2img"])
    multi = {s: v for s, v in lidar_const.items() if len(v) > 1}
    return {"moments": len(sel), "missing_from_main": missing, "features_bit_equal": feats_eq,
            "human_path_bit_equal": human_eq,
            "scenes_with_2plus_moments_lidar2img_constant": [sum(all(torch.equal(v[0], x) for x in v) for v in multi.values()), len(multi)],
            "unused_key_not_assembled": sorted(set(sel[0]["features"]) - {"camera_feature_1", "camera_feature_2", "ego_status", "lidar2img", "img_shape"}),
            "pass": feats_eq == len(sel) and human_eq == len(sel) and missing == 0}


if __name__ == "__main__":
    {"build": build}[sys.argv[1]]()
