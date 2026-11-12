"""Check 2 (b2d_drivejepa/vocab/CLAUDE.md): a known left turn curves toward +y, in the expert path and in its nearest vocabulary path.

Examples: the turn scenarios (SignalizedJunctionLeftTurn etc.) with command_near LEFT / RIGHT and speed above 2 m/s, the frame with the
largest end heading, one per direction per split. A training path can be its own vocabulary path at large sizes, so held-out examples
are included. Where the clip's raw anno is on disk, the direction is also read from CARLA's own numbers, which do not pass through
Phase 0a's flip: in CARLA (left-handed, y right) a left turn has a negative yaw change and a negative offset along the right vector.

Also: every frame with that command and an end heading beyond 30 degrees (the turn cut of Phase 0a), and the rear-axle test: on turning
frames a rear axle moves along its heading, so |path tangent - heading| is compared for the shift applied zero, one and two times and
in the wrong direction.

Writes (refuses to overwrite): check_handedness.json."""
import gzip
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "{ops}/run3a-adapted")
from phase1.data import geometry as G                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "{data.b2d_clips}/b2d_base"
SIZES = (256, 512, 1024, 2048, 4096, 8192)
NAVSIM = "{workspace_root}/wa-jepa/Drive-JEPA-upstream/navsim_v2/data/8192.npy"
VOCABS = [(f"b2d_{k}", f"{HERE}/b2d_{k}.npy") for k in SIZES] + [("navsim_8192", NAVSIM)]
TURN_SCENARIOS = {"LEFT": ("SignalizedJunctionLeftTurn", "NonSignalizedJunctionLeftTurn"),
                  "RIGHT": ("SignalizedJunctionRightTurn", "NonSignalizedJunctionRightTurn")}
OUT = f"{HERE}/check_handedness.json"


def carla_turn(clip, n):
    """Raw anno at frames n and n + 40 -> CARLA yaw change (degrees) and offset along CARLA's right vector at n (metres)."""
    def ego(i):
        fr = json.load(gzip.open(f"{RAW}/{clip}/anno/{i:05d}.json.gz", "rt"))
        e = next(b for b in fr["bounding_boxes"] if b["class"] == "ego_vehicle")
        return float(e["location"][0]), float(e["location"][1]), float(e["rotation"][2])
    x0, y0, yaw0 = ego(n)
    x1, y1, yaw1 = ego(n + 40)
    c, s = math.cos(math.radians(yaw0)), math.sin(math.radians(yaw0))
    return dict(carla_yaw_change_deg=(yaw1 - yaw0 + 180) % 360 - 180, carla_offset_right_m=-s * (x1 - x0) + c * (y1 - y0))


def shift(p, o):
    q = p.copy()
    q[..., 0] = p[..., 0] - o * np.cos(p[..., 2]) + o
    q[..., 1] = p[..., 1] - o * np.sin(p[..., 2])
    return q


def tangent_error(p):
    d = np.diff(p[..., :2], axis=1)
    h = np.angle(np.exp(1j * p[:, 1:, 2]) + np.exp(1j * p[:, :-1, 2]))
    e = np.angle(np.exp(1j * (np.arctan2(d[..., 1], d[..., 0]) - h)))
    return float(np.abs(e[np.hypot(d[..., 0], d[..., 1]) > 0.1]).mean())


def main():
    if os.path.exists(OUT):
        sys.exit(f"STOP: {OUT} exists, not overwriting")
    z = np.load(f"{HERE}/paths_cache.npz")
    rear, raw, split, clip_id, frame, cmd, speed = (z[k] for k in "rear raw split clip_id frame cmd speed".split())
    names, commands = list(z["clip_names"]), list(z["commands"])
    scen = np.asarray([c.split("_Town")[0] for c in names])[clip_id]
    yaw_end, y_end = rear[:, -1, 2], rear[:, -1, 1]
    vocab = {n: np.load(p) for n, p in VOCABS}
    near = {n: np.load(f"{HERE}/nearest_{n}.npz") for n, _ in VOCABS}

    examples = []
    for side, sign in (("LEFT", 1.0), ("RIGHT", -1.0)):
        for sid, sn in ((0, "train"), (1, "heldout")):
            m = (cmd == commands.index(side)) & (split == sid) & np.isin(scen, TURN_SCENARIOS[side]) & (speed > 2)
            i = int(np.flatnonzero(m)[np.argmax(sign * yaw_end[m])])
            clip, n = names[clip_id[i]], int(frame[i])
            ex = dict(command=side, split=sn, clip=clip, frame=n, speed_mps=float(speed[i]),
                      expert=dict(x_end=float(rear[i, -1, 0]), y_end=float(y_end[i]), yaw_end_deg=math.degrees(yaw_end[i]),
                                  y_at_8_points=[round(float(v), 2) for v in rear[i, 4::5, 1]]))
            if os.path.isdir(f"{RAW}/{clip}/anno"):
                ex["raw_carla"] = carla_turn(clip, n)
                ex["front_camera_dir"] = f"{RAW}/{clip}/camera/rgb_front"
            ex["nearest"] = {}
            for name, _ in VOCABS:
                v = vocab[name][int(near[name]["idx"][i])]
                ex["nearest"][name] = dict(index=int(near[name]["idx"][i]), ade=float(near[name]["ade"][i]), x_end=float(v[-1, 0]),
                                           y_end=float(v[-1, 1]), yaw_end_deg=math.degrees(float(v[-1, 2])),
                                           y_at_8_points=[round(float(a), 2) for a in v[4::5, 1]])
            examples.append(ex)

    agg = []
    for side, sign in (("LEFT", 1.0), ("RIGHT", -1.0)):
        for sid, sn in ((0, "train"), (1, "heldout")):
            m = (cmd == commands.index(side)) & (split == sid)
            t = m & (np.abs(yaw_end) > math.radians(30))
            row = dict(command=side, split=sn, frames_with_command=int(m.sum()), end_heading_beyond_30deg=int(t.sum()),
                       expert_share_y_toward_side=float((sign * y_end[t] > 0).mean()),
                       expert_share_yaw_toward_side=float((sign * yaw_end[t] > 0).mean()), nearest_share_y_toward_side={})
            for name, _ in VOCABS:
                v = vocab[name][near[name]["idx"][t]]
                row["nearest_share_y_toward_side"][name] = float((sign * v[:, -1, 1] > 0).mean())
            agg.append(row)

    corr = {n: float(np.corrcoef(v[:, -1, 1], v[:, -1, 2])[0, 1]) for n, v in vocab.items()}
    corr["expert_train"] = float(np.corrcoef(y_end[split == 0], yaw_end[split == 0])[0, 1])

    turning = (np.abs(yaw_end) > 0.8) & (speed > 2)
    o = G.B2D_REAR_AXLE_OFFSET
    actor = raw[turning].astype(np.float64)
    once = rear[turning]
    rear_test = dict(frames=int(turning.sum()), actor_origin=tangent_error(actor), rear_axle_once=tangent_error(once),
                     rear_axle_twice=tangent_error(shift(once, o)), wrong_direction=tangent_error(shift(actor, -o)),
                     max_gap_rear_vs_own_shift_m=float(np.abs(shift(actor, o) - once).max()))

    timing = (speed > 2)
    d1 = np.hypot(raw[timing, 0, 0], raw[timing, 0, 1])
    d5 = np.hypot(raw[timing, 4, 0], raw[timing, 4, 1])
    v = speed[timing]
    time_check = dict(frames=int(timing.sum()), seconds_at_index_0=float((d1 * v).sum() / (v ** 2).sum()),
                      seconds_at_index_4=float((d5 * v).sum() / (v ** 2).sum()))

    out = dict(examples=examples, all_turn_frames=agg, correlation_y_end_yaw_end=corr, rear_axle_test_rad=rear_test, timing=time_check)
    with open(OUT, "x") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
