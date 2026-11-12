"""The held-out moments for the ADE report: every frame of the 47 held-out clips with a Phase 1 target row (its 1.5 s history exists) and 3 s of expert future. Inputs, the same for
the LoRA and the released encoder:
  pixels   the two cached `native` frames (t - 0.5 s, t)
  camera   front: the dataset's true camera, rear-axle frame (Phase 1 cam2ego), native intrinsics f 571.26, c (400, 225), through the feature builder's own lidar2img
  speed    [anno speed, 0]; accel [(v_t - v_(t-0.1 s)) / 0.1, 0], the runner's dv/dt at the anno's 10 Hz
  command  the runner's: anno command_near through agent.py CMD and policy COMMAND_ONE_HOT, so LANEFOLLOW -> unknown [0,0,0,1], lane changes -> straight
  expert   Phase 1 ego_future: rear-axle poses at 0.5 .. 3.0 s in the rear-axle frame at t
-> $RUNS/ade_set.pt   (cwd anywhere)"""
import gzip
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wiring as W  # noqa: E402

R3 = "{ops}/run3a-adapted"
sys.path.insert(0, R3)
cfg = yaml.safe_load(open(f"{R3}/phase1/config.yaml"))
W.setup()
from phase1.data.dataset import Phase1Dataset  # noqa: E402
from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder  # noqa: E402
from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus  # noqa: E402

RO = {1: [1, 0, 0, 0], 2: [0, 0, 1, 0], 3: [0, 1, 0, 0], 4: [0, 0, 0, 1], 5: [0, 1, 0, 0], 6: [0, 1, 0, 0]}   # LEFT RIGHT STRAIGHT LANEFOLLOW CHANGE-L CHANGE-R
K = np.array([[571.2592, 0, 400], [0, 571.2592, 225], [0, 0, 1]], np.float32)                                  # anno CAM_FRONT intrinsic / 2
held = set(json.load(open(W.ROOT / "splits/clips.json"))["heldout"])
frames = json.load(open(W.CACHE / "index.json"))["frames"]
ds = Phase1Dataset("b2d", cfg, split="all", load_images=False)
builder = DriveJEPAFeatureBuilder(None)
zero = np.zeros((450, 800, 3), np.uint8)


def anno(clip, n):
    for s in ("b2d_base", "b2d_pilot"):
        p = W.BASE / s / clip / "anno" / f"{n:05d}.json.gz"
        if p.exists():
            return json.load(gzip.open(p))
    raise FileNotFoundError((clip, n))


out, skipped = [], dict(no_future=0, first_frame=0)
for k in range(len(ds)):
    r = ds.row(k)
    if r["log"] not in held:
        continue
    if not bool(np.all(r["ego_future_valid"][:6])):
        skipped["no_future"] += 1
        continue
    name = os.path.basename(r["images"][1][-1])
    i = frames[r["log"]].index(name)
    if i < 1:
        skipped["first_frame"] += 1
        continue
    n = int(name.split(".")[0])
    a0, a1 = anno(r["log"], n - 1), anno(r["log"], n)
    cams = {}
    for slot, ci in (("cam_l0", 0), ("cam_f0", 1), ("cam_r0", 2), ("cam_b0", 3)):
        T = np.asarray(r["cam2ego"][ci], np.float64)
        cams[slot] = Camera(image=zero, sensor2lidar_rotation=T[:3, :3].astype(np.float32), sensor2lidar_translation=T[:3, 3].astype(np.float32), intrinsics=K)
    e = Camera()
    C = Cameras(cam_f0=cams["cam_f0"], cam_l0=cams["cam_l0"], cam_l1=e, cam_l2=e, cam_r0=cams["cam_r0"], cam_r1=e, cam_r2=e, cam_b0=cams["cam_b0"])
    st = EgoStatus(ego_pose=np.zeros(3, np.float32), ego_velocity=np.array([a1["speed"], 0.0], np.float32),
                   ego_acceleration=np.array([(a1["speed"] - a0["speed"]) / 0.1, 0.0], np.float32),
                   driving_command=np.array(RO[int(a1["command_near"])], np.int64))
    f = builder.compute_features(AgentInput(ego_statuses=[st], cameras=[C, C], lidars=[]))
    out.append(dict(clip=r["log"], i=i, frame=name, lidar2img=f["lidar2img"], img_shape=f["img_shape"], ego_status=f["ego_status"],
                    expert=np.asarray(r["ego_future"][:6, :2], np.float32), command=RO[int(a1["command_near"])], speed=float(a1["speed"])))
assert len({o["clip"] for o in out}) <= len(held)
torch.save(dict(moments=out, K=K, skipped=skipped), W.RUNS / "ade_set.pt")
print("ADE_SET", len(out), "moments,", len({o["clip"] for o in out}), "clips, skipped", skipped, "commands", {tuple(c): sum(o["command"] == c for o in out) for c in RO.values()})
