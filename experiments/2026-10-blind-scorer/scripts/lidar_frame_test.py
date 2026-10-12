"""Does any of Drive-JEPA's image sampling land inside the front camera, with nuScenes' real lidar2ego and without its rotation?"""
import os, sys
import numpy as np
sys.path.insert(0, "{gallops}/src")
from gallops.config import SiteConfig
cfg = SiteConfig.load()
for name, template in (("NAVSIM_DEVKIT_ROOT", "{data.drivejepa_repo}"), ("NAVSIM_EXP_ROOT", "{data.navsim.exp}"),
                       ("OPENSCENE_DATA_ROOT", "{data.navsim}"), ("NUPLAN_MAPS_ROOT", "{data.navsim}/maps")):
    os.environ.setdefault(name, cfg.resolve(template))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"); os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")
import torch
_orig = torch.load; torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": "cpu"})
from pathlib import Path
from gallops.train import swerve_data
from gallops.train.swerve import build_model, load_ticks
from navsim.agents.drive_jepa_perception_based.bevformer import encoder as enc

COVER = []
cls = [c for c in vars(enc).values() if isinstance(c, type) and hasattr(c, "point_sampling")]
for c in cls:
    orig = c.point_sampling
    def wrapped(self, reference_points, img_metas, _orig=orig):
        rp, mask = _orig(self, reference_points, img_metas)
        COVER.append(float(mask.float().mean())); return rp, mask
    c.point_sampling = wrapped
print("hooked", [c.__name__ for c in cls])

real_calib = swerve_data.scene_calibration
def calib_variant(kind):
    def f(root, scene):
        c = dict(real_calib(root, scene)); pose, k = c["LIDAR_TOP"]; pose = np.array(pose, float)
        if kind == "no_rotation": pose[:3, :3] = np.eye(3)
        if kind == "identity": pose = np.eye(4)  # NAVSIM: lidar2ego is the identity
        c["LIDAR_TOP"] = (pose, k); return c
    return f

L = Path("{runs}/labelled_gt")
ticks = [t for t in load_ticks(L, blind=True) if t["episode"] == "frontal-0070/run_0"]
tick = sorted(ticks, key=lambda t: t["timestamp"])[-1]
ckpt = cfg.resolve("{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt")
agent, model, _ = build_model(ckpt, "cpu", cfg.resolve("{data.drivejepa_repo}"), train_parts=("__nothing__",)); model.eval()
pose = np.array(real_calib(swerve_data.TickDataset([tick], "cpu").nuscenes_root, "scene-0070")["LIDAR_TOP"][0])
print("nuScenes LIDAR_TOP -> ego rotation (yaw deg):", round(float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0]))), 1), " translation", pose[:3, 3].round(2).tolist())

for kind in ("identity",):
    swerve_data.scene_calibration = calib_variant(kind)
    ds = swerve_data.TickDataset([tick], "cpu")
    outs = []
    for img in ("real", "blank"):
        x = ds.features(tick, fast=False)
        if img == "blank":
            x["camera_feature_1"] = torch.zeros_like(x["camera_feature_1"]); x["camera_feature_2"] = torch.zeros_like(x["camera_feature_2"])
        COVER.clear()
        with torch.no_grad(): o = model(x)
        outs.append(o["trajectory"].float()); cov = np.mean(COVER) if COVER else float("nan")
        print(f"  lidar2ego={kind:12s} image={img:5s}: share of sample points inside the front image {cov:.3%}; plan end {o['trajectory'][0, -1, :2].numpy().round(2).tolist()}", flush=True)
    print(f"  lidar2ego={kind:12s}: real vs blank image, trajectory diff {float((outs[0] - outs[1]).abs().max()):.3f} m", flush=True)
