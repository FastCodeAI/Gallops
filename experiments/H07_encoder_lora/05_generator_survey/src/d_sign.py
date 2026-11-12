"""Check D, axis sign: on turning training moments, does the human path's sideways
offset (y) at 4 s have the same sign as its heading change?  Left turn -> y > 0 and
heading > 0 if the frame is x forward, y left, heading counter-clockwise, which is
the generator's (NAVSIM) convention.  Read-only, CPU only.

    ./run.sh src/d_sign.py
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import common as C
import splits as S

ADP = C.kit_adapter_module()
OUT = Path(__file__).resolve().parent.parent / "out"


def scene(args):
    path, ts = args
    _, poses = ADP.load_poses(Path(path))
    res = []
    for t in ts:
        rel = ADP.to_relative_poses(poses[t], poses[[t + 5 * (k + 1) for k in range(8)]])
        res.append((float(rel[-1, 1]), float(rel[-1, 2])))
    return res


def main() -> None:
    data = S.load_splits()
    moments = json.loads((OUT / "d_moments.json").read_text())["train"]
    with ThreadPoolExecutor(16) as pool:
        rows = [r for rs in pool.map(scene, [(data["scene_paths"][s], ts) for s, ts in moments.items()]) for r in rs]
    a = np.array(rows)
    turn = np.abs(a[:, 1]) > 0.3  # > ~17 degrees of heading change over 4 s
    out = {"moments": len(a), "turning_moments": int(turn.sum()),
           "sign_y_equals_sign_heading": int((np.sign(a[turn, 0]) == np.sign(a[turn, 1])).sum()),
           "left_turns (heading > 0.3)": int((a[:, 1] > 0.3).sum()), "right_turns (heading < -0.3)": int((a[:, 1] < -0.3).sum())}
    (OUT / "d_sign.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
