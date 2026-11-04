"""The BEV grid the auxiliary map head predicts, and the rasterisers that fill it.

The grid is upstream's, not ours: `MapHead` was written for TransFuser's BEV semantic map and
its output size is fixed by `DriveJEPAConfig` (`bev_pixel_height` 128, `bev_pixel_width` 256,
`bev_pixel_size` 0.25 m). Upstream draws each layer with OpenCV on a (256, 128) image whose
pixel is (col = x / 0.25, row = y / 0.25 + 128), then takes `np.rot90(mask)[::-1]` --
which is a transpose. So in the target the model sees:

    bev[i, j]  is the cell whose centre is  x = (i + 0.5) * 0.25   (forward, 0 to 32 m)
                                            y = (j + 0.5) * 0.25 - 32   (left positive)

Row 0 is at the ego, row 127 is 32 m ahead; column 0 is 32 m to the RIGHT. Plot with
`bev[::-1, ::-1]` to see forward up and left on the left. `step.check_grid` holds these
numbers to the config the model was built with, so a change upstream fails loudly.

Rasterisation tests cell centres (point in polygon, distance to a polyline) rather than
calling OpenCV, so this module needs numpy alone and `make check` can test it. Upstream's
OpenCV fill truncates vertices to whole pixels, a difference of at most one cell at a border;
these heads start from random weights, so nothing depends on matching that to the pixel.
"""

from __future__ import annotations

import numpy as np

ROWS = 128  # DriveJEPAConfig.bev_pixel_height = lidar_resolution_height // 2
COLS = 256  # DriveJEPAConfig.bev_pixel_width = lidar_resolution_width
CELL_M = 0.25  # DriveJEPAConfig.bev_pixel_size
X_MAX = ROWS * CELL_M  # 32 m ahead
Y_HALF = COLS * CELL_M / 2  # 32 m either side

#: A cell the loss must not see: outside the camera's view, or nothing labels it.
IGNORE = 255

#: Upstream's seven TransFuser classes, in its order, plus lane dividers. Upstream paints its
#: classes 1..6 in order, later over earlier; dividers go after the centreline and before the
#: objects, so a car standing on a painted line is still a car.
CLASSES = (
    "background",
    "road",
    "walkway",
    "centreline",
    "static_object",
    "vehicle",
    "pedestrian",
    "lane_divider",
)
PAINT_ORDER = ("road", "walkway", "centreline", "lane_divider", "static_object", "vehicle", "pedestrian")
#: Upstream draws linestrings 2 px (0.5 m) thick.
LINE_HALF_WIDTH_M = 0.25


def cell_centres() -> tuple[np.ndarray, np.ndarray]:
    """(x of each row, y of each column), metres in the ego frame."""
    xs = (np.arange(ROWS) + 0.5) * CELL_M
    ys = (np.arange(COLS) + 0.5) * CELL_M - Y_HALF
    return xs, ys


def cell_grid() -> tuple[np.ndarray, np.ndarray]:
    """x and y of every cell centre, each (ROWS, COLS)."""
    xs, ys = cell_centres()
    return np.meshgrid(xs, ys, indexing="ij")


def xy_to_cell(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Row and column holding the point (x, y); can fall outside the grid."""
    return (
        np.floor(np.asarray(x) / CELL_M).astype(int),
        np.floor((np.asarray(y) + Y_HALF) / CELL_M).astype(int),
    )


def in_grid(rows, cols) -> np.ndarray:
    rows, cols = np.asarray(rows), np.asarray(cols)
    return (rows >= 0) & (rows < ROWS) & (cols >= 0) & (cols < COLS)


def _inside_ring(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd test of points against one closed ring (N, 2)."""
    inside = np.zeros(px.shape, dtype=bool)
    x0, y0 = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):  # noqa: B905 -- Python 3.9
        crosses = (ay > py) != (by > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = ax + (py - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (px < at)
    return inside


def polygon_mask(polygons) -> np.ndarray:
    """Cells whose centre lies inside any polygon.

    Each polygon is an (N, 2) exterior ring in the ego frame, or a tuple (exterior, [holes]).
    Only the cells inside each polygon's bounding box are tested.
    """
    mask = np.zeros((ROWS, COLS), dtype=bool)
    gx, gy = cell_grid()
    for polygon in polygons:
        exterior, holes = polygon if isinstance(polygon, tuple) else (polygon, [])
        exterior = np.asarray(exterior, dtype=float)
        r0, c0 = xy_to_cell(exterior[:, 0].min(), exterior[:, 1].min())
        r1, c1 = xy_to_cell(exterior[:, 0].max(), exterior[:, 1].max())
        r0, r1 = max(int(r0), 0), min(int(r1) + 1, ROWS)
        c0, c1 = max(int(c0), 0), min(int(c1) + 1, COLS)
        if r0 >= r1 or c0 >= c1:
            continue
        px, py = gx[r0:r1, c0:c1], gy[r0:r1, c0:c1]
        inside = _inside_ring(px, py, exterior)
        for hole in holes:
            inside &= ~_inside_ring(px, py, np.asarray(hole, dtype=float))
        mask[r0:r1, c0:c1] |= inside
    return mask


def polyline_mask(lines, half_width: float = LINE_HALF_WIDTH_M) -> np.ndarray:
    """Cells whose centre lies within `half_width` of any segment of any (N, 2) polyline."""
    mask = np.zeros((ROWS, COLS), dtype=bool)
    gx, gy = cell_grid()
    for line in lines:
        line = np.asarray(line, dtype=float)
        if len(line) < 2:
            continue
        for (ax, ay), (bx, by) in zip(line[:-1], line[1:]):  # noqa: B905 -- Python 3.9
            lo_r, lo_c = xy_to_cell(min(ax, bx) - half_width, min(ay, by) - half_width)
            hi_r, hi_c = xy_to_cell(max(ax, bx) + half_width, max(ay, by) + half_width)
            r0, r1 = max(int(lo_r), 0), min(int(hi_r) + 1, ROWS)
            c0, c1 = max(int(lo_c), 0), min(int(hi_c) + 1, COLS)
            if r0 >= r1 or c0 >= c1:
                continue
            px, py = gx[r0:r1, c0:c1], gy[r0:r1, c0:c1]
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = (
                np.zeros(px.shape)
                if length2 == 0
                else np.clip(((px - ax) * dx + (py - ay) * dy) / length2, 0, 1)
            )
            near = (px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2 <= half_width**2
            mask[r0:r1, c0:c1] |= near
    return mask


def box_corners(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
    """The four corners (4, 2) of a box on the ground, in the frame its centre is given in."""
    c, s = np.cos(heading), np.sin(heading)
    half = np.array([[length, width], [length, -width], [-length, -width], [-length, width]]) / 2
    return np.stack([x + half[:, 0] * c - half[:, 1] * s, y + half[:, 0] * s + half[:, 1] * c], axis=1)


def box_mask(boxes) -> np.ndarray:
    """Cells under any box; each box is (x, y, heading, length, width)."""
    return polygon_mask([box_corners(*box[:5]) for box in boxes])


def project(
    points_ego: np.ndarray, intrinsics: np.ndarray, cam2ego: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel (u, v) of ego-frame points (N, 3), and their depth along the camera's axis."""
    rotation, translation = cam2ego[:3, :3], cam2ego[:3, 3]
    in_camera = (np.asarray(points_ego, dtype=float) - translation) @ rotation  # R^T (p - t), row-wise
    depth = in_camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (in_camera @ np.asarray(intrinsics, dtype=float).T)[:, :2] / depth[:, None]
    return uv, depth


def visible(
    points_ego: np.ndarray,
    intrinsics: np.ndarray,
    cam2ego: np.ndarray,
    width: int,
    rows: tuple[int, int],
    min_depth: float = 0.5,
) -> np.ndarray:
    """Which ego-frame points (N, 3) land inside the part of the image the model reads.

    `rows` is the band of image rows kept: Drive-JEPA crops 28 rows off the top and the bottom
    of the frame before resizing it (`swerve_data.TickDataset.frame`), so a point that projects
    there is not seen.
    """
    uv, depth = project(points_ego, intrinsics, cam2ego)
    u, v = uv[:, 0], uv[:, 1]
    with np.errstate(invalid="ignore"):
        return (depth > min_depth) & (u >= 0) & (u < width) & (v >= rows[0]) & (v < rows[1])


def view_mask(intrinsics: np.ndarray, cam2ego: np.ndarray, width: int, rows: tuple[int, int]) -> np.ndarray:
    """Cells whose centre, on the ground (z = 0), the camera sees: the cells a label may cover."""
    gx, gy = cell_grid()
    points = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    return visible(points, intrinsics, cam2ego, width, rows).reshape(ROWS, COLS)


def semantic_map(layers: dict, view: np.ndarray) -> np.ndarray:
    """The target: one class per cell, IGNORE outside the camera's view.

    `layers` maps a class name to its boolean mask. A visible cell that no layer covers is
    background (class 0): the camera sees it and it is none of the classes.
    """
    unknown = set(layers) - set(CLASSES)
    if unknown:
        raise ValueError(f"no class for layer(s) {sorted(unknown)}; classes are {CLASSES}")
    target = np.zeros((ROWS, COLS), dtype=np.uint8)
    for name in PAINT_ORDER:
        if name in layers:
            target[layers[name]] = CLASSES.index(name)
    target[~view] = IGNORE
    return target
