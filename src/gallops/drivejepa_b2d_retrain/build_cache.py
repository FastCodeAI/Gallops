"""Frame cache: every front-camera frame of the 944 clips as the 256 x 512 x 3 uint8 array behind ToTensor.
Crop `native`: the 1600 x 900 original resized to 800 x 450, then Drive-JEPA's own
DriveJEPAFeatureBuilder._get_camera_feature (imported). One <clip>.bin per clip (frames back to back,
file-name order), written to .part then renamed; index.json at the end. Resumable.
    python -m gallops.drivejepa_b2d_retrain.build_cache <workers>"""

import json
import multiprocessing as mp
import os
import sys
import time

import cv2

from gallops.drivejepa_b2d_retrain import wiring as W

cv2.setNumThreads(1)
H, Wd = 256, 512
BYTES = H * Wd * 3


def clip_dir(c):
    for s in ("b2d_base", "b2d_pilot"):
        d = W.BASE / s / c / "camera" / "rgb_front"
        if d.is_dir():
            return d
    raise FileNotFoundError(c)


class _Cam:
    def __init__(self, image):
        self.cam_f0 = type("C", (), {"image": image})()


def native(path, builder, to_tensor):
    """BGR file -> RGB 1600 x 900 -> 800 x 450 -> builder -> uint8 HWC, exact."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr.shape == (900, 1600, 3), (path, bgr.shape)
    img = cv2.resize(bgr[:, :, ::-1].copy(), (800, 450), interpolation=cv2.INTER_AREA)
    t = builder._get_camera_feature(type("A", (), {"cameras": [_Cam(img), _Cam(img)]})())["camera_feature_1"]
    u = (t * 255.0).round().to(__import__("torch").uint8).permute(1, 2, 0).contiguous().numpy()
    return u


def work(c):
    out = W.CACHE / f"{c}.bin"
    d = clip_dir(c)
    names = sorted(p.name for p in d.iterdir() if p.suffix == ".jpg")
    if out.exists() and out.stat().st_size == len(names) * BYTES:
        return c, names
    import torch
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder
    from torchvision import transforms

    torch.set_num_threads(1)
    to_tensor = transforms.ToTensor()
    b = DriveJEPAFeatureBuilder(None)
    tmp = out.with_suffix(".part")
    with open(tmp, "wb") as fh:
        for i, n in enumerate(names):
            u = native(d / n, b, to_tensor)
            if i == 0:
                bgr = cv2.imread(str(d / n), cv2.IMREAD_COLOR)
                img = cv2.resize(bgr[:, :, ::-1].copy(), (800, 450), interpolation=cv2.INTER_AREA)
                ref = b._get_camera_feature(type("A", (), {"cameras": [_Cam(img), _Cam(img)]})())[
                    "camera_feature_1"
                ]
                assert torch.equal(to_tensor(u), ref), f"uint8 round trip {c}"
            fh.write(u.tobytes())
    os.replace(tmp, out)
    return c, names


def init():
    W.setup()


if __name__ == "__main__":
    W.CACHE.mkdir(parents=True, exist_ok=True)
    assert not (W.CACHE / "index.json").exists(), "never overwrite a cache"
    clips = json.load(open(W.SPLITS))["clips"]
    t0, frames = time.time(), {}
    with mp.get_context("spawn").Pool(int(sys.argv[1]), initializer=init) as pool:
        for i, (c, names) in enumerate(pool.imap_unordered(work, clips, chunksize=1)):
            frames[c] = names
            if (i + 1) % 20 == 0:
                print(
                    f"{time.strftime('%T')} {i + 1}/{len(clips)} clips, {sum(map(len, frames.values()))} frames, {time.time() - t0:.0f} s",  # noqa: E501
                    flush=True,
                )
    json.dump(dict(frames=frames, h=H, w=Wd), open(W.CACHE / "index.json", "w"))
    print(
        "CACHE_DONE",
        len(frames),
        "clips",
        sum(map(len, frames.values())),
        "frames",
        round(time.time() - t0),
        "s",
        flush=True,
    )
