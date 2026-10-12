"""paddock's own Drive-JEPA adapter (whichever src is on PYTHONPATH), real vs blank front image, on four real
0070 GT frames 0.5 s apart, nuScenes calibration including the real LIDAR_TOP. CPU."""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.append("{gallops}/src")
from gallops.config import SiteConfig
cfg = SiteConfig.load()
for name, template in (("NAVSIM_DEVKIT_ROOT", "{data.drivejepa_repo}"), ("NAVSIM_EXP_ROOT", "{data.navsim.exp}"),
                       ("OPENSCENE_DATA_ROOT", "{data.navsim}"), ("NUPLAN_MAPS_ROOT", "{data.navsim}/maps")):
    os.environ.setdefault(name, cfg.resolve(template))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"); os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")
import torch
_orig = torch.load; torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": "cpu"})
import paddock
from paddock.adapters.models.drivejepa.policy import DriveJepaPolicy
from paddock.policy.protocol import Command, Observation
from gallops.train.swerve_data import scene_calibration
from navsim.agents.drive_jepa_perception_based.bevformer import encoder as enc
COVER = []
orig_ps = enc.BEVFormerEncoder.point_sampling
def ps(self, rp, metas):
    a, m = orig_ps(self, rp, metas); COVER.append(float(m.float().mean())); return a, m
enc.BEVFormerEncoder.point_sampling = ps
print("paddock from", paddock.__file__, flush=True)

ep = Path("{runs}/collect_gt/frontal-0070/run_0")
poses = json.loads((ep / "ego_poses.json").read_text())
stamps = sorted(int(p.stem) for p in (ep / "CAM_FRONT").glob("*.jpg"))
now = stamps[-1]; chosen = [min(stamps, key=lambda s: abs(s - (now - k * 500_000))) for k in (3, 2, 1, 0)]
cal = scene_calibration(cfg.resolve("{data.nuscenes}"), "scene-0070")
cams = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK")

def obs(stamp, blank):
    front = np.asarray(Image.open(ep / "CAM_FRONT" / f"{stamp}.jpg").convert("RGB"))
    images = {c: np.zeros_like(front) for c in cams}
    if not blank: images["CAM_FRONT"] = front
    return Observation(images=images, ego2world=np.asarray(poses[str(stamp)]), canbus=np.zeros(18), timestamp_us=stamp,
                       command=Command.STRAIGHT, camera2image={c: np.asarray(cal[c][1], float) for c in cams},
                       camera2ego={c: np.asarray(cal[c][0], float) for c in cams}, lidar2ego=np.asarray(cal["LIDAR_TOP"][0], float))

policy = DriveJepaPolicy(cfg.resolve("{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"), device="cpu")
policy.load()
out = {}
for blank in (False, True):
    policy.reset(); COVER.clear()
    for s in chosen[:-1]: policy.observe(obs(s, blank))
    with torch.no_grad(): o = policy.forward(obs(chosen[-1], blank))
    out[blank] = o["trajectory"][0].float().numpy()
    print(f"  front image {'BLANK' if blank else 'real '}: sample points inside the front image {np.mean(COVER):.1%}; "
          f"plan end (x, y) {out[blank][-1, :2].round(2).tolist()}", flush=True)
print(f"  real vs blank: plan differs by up to {np.abs(out[False] - out[True])[..., :2].max():.2f} m", flush=True)
