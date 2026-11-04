"""Pictures of one NAVSIM sample, for review: its camera frames, its boxes on the image the ViT reads,
and the same scene from above.

The review notebooks (experiments/H15_perception_sidecar/notebooks/) draw with these; the geometry
is here so it is tested, and so the pictures use the trainer's own projection (`targets`), not a
second copy of it.

- **A sample's window** is read straight from its log's pickle: the frame whose `token` is the
  sample's, with the 3 frames before it and the 10 after, as NAVSIM's `filter_scenes` slices it
  (`navsim_data.HISTORY`, `FUTURE`; frames are 0.5 s apart). No navsim import, so a notebook never
  has to enter Drive-JEPA's environment to look at a sample.
- **What the ViT reads** of a frame is CAM_F0, rows 28 to -28, resized to 512x256 (the vendor's
  `_get_camera_feature`). Its token grid is 16 rows by 32 columns of 16x16 pixels.
- **Boxes** are (x, y, z, length, width, height, yaw) in the ego frame, x forward, y left; z is
  taken as the box's centre height. Corners project through `targets.front_projection`, which is
  the projection the side-car's rays come from.

numpy only at import; PIL (or cv2) is imported where an image is read or resized.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from gallops.drivejepa_perception import navsim_data
from gallops.drivejepa_perception import targets as tg

#: The vendor's crop and the ViT's input size (drive_jepa_features._get_camera_feature).
CROP_ROWS, MODEL_W, MODEL_H = tg.CROP_ROWS, 512, 256
PATCH = 16
#: The 12 edges of a box, as pairs of `box_corners` indices: bottom face, top face, verticals.
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def window_frames(navsim_root, log: str, token: str, split: str = "trainval") -> list:
    """The sample's 14 raw frames, current at index navsim_data.CURRENT, from its log's pickle."""
    with open(Path(navsim_root) / "navsim_logs" / split / f"{log}.pkl", "rb") as fh:
        frames = pickle.load(fh)
    index = [f["token"] for f in frames].index(token)
    first = index - navsim_data.CURRENT
    window = frames[first : index + navsim_data.FUTURE + 1]
    if first < 0 or len(window) != navsim_data.HISTORY + navsim_data.FUTURE:
        raise ValueError(f"{token}: the log has no full window around it")
    return window


def camera_path(navsim_root, frame: dict, camera: str = "CAM_F0", split: str = "trainval") -> Path:
    return Path(navsim_root) / "sensor_blobs" / split / frame["cams"][camera]["data_path"]


def read_image(path) -> np.ndarray:
    """(H, W, 3) uint8 RGB."""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def model_view(image: np.ndarray) -> np.ndarray:
    """What the ViT reads of one full-size frame: the 28-row crop top and bottom, then 512x256.

    cv2's bilinear resize, as the vendor's; PIL's bilinear where cv2 is missing (a picture only).
    """
    cropped = image[CROP_ROWS:-CROP_ROWS]
    try:
        import cv2

        return cv2.resize(cropped, (MODEL_W, MODEL_H))
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(cropped).resize((MODEL_W, MODEL_H), Image.BILINEAR))


def box_corners(box: np.ndarray) -> np.ndarray:
    """(N, 7) x, y, z, length, width, height, yaw -> (N, 8, 3) corners in the ego frame.

    Corners 0-3 are the bottom face and 4-7 the top, each counter-clockwise from front-left as
    seen from above; 0 and 1 are the front edge (+length / 2 along the heading).
    """
    box = np.asarray(box, dtype=np.float64).reshape(-1, 7)
    half = box[:, 3:6, None] / 2  # N, 3 (l, w, h), 1
    signs = np.array(
        [
            [1, 1, -1, -1, 1, 1, -1, -1],  # along the heading
            [1, -1, -1, 1, 1, -1, -1, 1],  # across it
            [-1, -1, -1, -1, 1, 1, 1, 1],  # up
        ],
        dtype=np.float64,
    )
    local = half * signs[None]  # N, 3, 8
    c, s = np.cos(box[:, 6]), np.sin(box[:, 6])
    x = c[:, None] * local[:, 0] - s[:, None] * local[:, 1]
    y = s[:, None] * local[:, 0] + c[:, None] * local[:, 1]
    corners = np.stack([x, y, local[:, 2]], axis=-1)
    return corners + box[:, None, :3]


def project(projection: np.ndarray, points: np.ndarray) -> tuple:
    """(4, 4) ego -> image projection, (..., 3) points -> pixels (..., 2) and depth (...,).

    Depth is the camera's z: a point with depth <= 0 is behind the camera and its pixel is
    meaningless (it is returned, not masked; callers decide).
    """
    points = np.asarray(points, dtype=np.float64)
    h = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1) @ np.asarray(projection).T
    depth = h[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = h[..., :2] / depth[..., None]
    return uv, depth


def box_edges_2d(projection: np.ndarray, box: np.ndarray, min_depth: float = 0.5) -> list:
    """Each box's edges on the image: a list (one per box) of ((u0, v0), (u1, v1)) segments.

    An edge with an end closer than `min_depth` to the camera plane is dropped rather than clipped:
    it is a picture, and a clipped edge of a box beside the camera would mislead more than help.
    """
    uv, depth = project(projection, box_corners(box))
    out = []
    for b in range(len(uv)):
        out.append([(uv[b, i], uv[b, j]) for i, j in EDGES if min(depth[b, i], depth[b, j]) >= min_depth])
    return out


def footprint(box: np.ndarray) -> np.ndarray:
    """(N, 7) boxes -> (N, 4, 2) their outline from above (x, y), front-left first."""
    return box_corners(box)[:, :4, :2]


def camera_ray(projection: np.ndarray, u: float, v: float) -> tuple:
    """The camera centre (3,) and the unit ray (3,) through pixel (u, v), in the ego frame.

    The inverse of `project`, as `sidecar.token_rays` computes it for every token's centre.
    """
    p = np.asarray(projection, dtype=np.float64)
    m_inv = np.linalg.inv(p[:3, :3])
    centre = -m_inv @ p[:3, 3]
    ray = m_inv @ np.array([u, v, 1.0])
    return centre, ray / np.linalg.norm(ray)


def view_edges(projection: np.ndarray, reach: float = 60.0) -> np.ndarray:
    """(3, 2) the camera's view from above: its centre, and the points `reach` metres out along the
    rays through the image's left and right edges at its middle row (x, y). NAVSIM's principal
    point is at row 133 of 256, so the middle row's rays are the horizontal field of view."""
    centre, left = camera_ray(projection, 0.0, MODEL_H / 2)
    _, right = camera_ray(projection, float(MODEL_W), MODEL_H / 2)
    flat = [d[:2] / np.linalg.norm(d[:2]) for d in (left, right)]
    return np.array([centre[:2], centre[:2] + reach * flat[0], centre[:2] + reach * flat[1]])


def token_of(uv: np.ndarray) -> np.ndarray:
    """(..., 2) pixels on the 512x256 image -> (...,) the row-major index of the token they fall in,
    -1 outside the image (the side-car's 512 tokens are row-major over 16 x 32)."""
    uv = np.asarray(uv, dtype=np.float64)
    col, row = np.floor(uv[..., 0] / PATCH), np.floor(uv[..., 1] / PATCH)
    inside = (col >= 0) & (col < tg.W_TOK) & (row >= 0) & (row < tg.H_TOK)
    return np.where(inside, row * tg.W_TOK + col, -1).astype(np.int64)
