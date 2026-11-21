"""Stage extract-agents (E1 step A, 21 Oct): agents at time t for every navtrain sample in run 2's token
cache, from the navtrain logs on the NAS. Keeps vehicles, pedestrians and cyclists ('bicycle') of the token's
own frame: boxes (x, y, z, l, w, h, yaw) in the ego frame (= NAVSIM's lidar frame, the frame lidar2img maps
from; checked in wajepa-runs/audit), names, velocity. 2 readers (NAS rule). Ported from paddock-ops
bev-e1/extract_agents.py; only this docstring changed.

    python -m gallops.drivejepa_wajepa.extract_agents <token cache root> <navsim_logs/trainval> <out.pkl>"""

import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

CACHE, LOGS, OUT = sys.argv[1:4]
KEEP = ("vehicle", "pedestrian", "bicycle")


def one(log):
    toks = set(os.listdir(os.path.join(CACHE, log)))
    p = os.path.join(LOGS, f"{log}.pkl")
    if not os.path.exists(p):
        return log, None
    out = {}
    for f in pickle.load(open(p, "rb")):
        if f["token"] in toks:
            a = f["anns"]
            m = np.isin(a["gt_names"], KEEP)
            out[f["token"]] = dict(
                boxes=a["gt_boxes"][m].astype(np.float32),
                names=a["gt_names"][m].tolist(),
                vel=a["gt_velocity_3d"][m].astype(np.float32),
            )
    return log, out


if __name__ == "__main__":
    t0 = time.time()
    res = {}
    miss = []
    logs = sorted(os.listdir(CACHE))
    with Pool(2) as pool:
        for k, (log, d) in enumerate(pool.imap_unordered(one, logs)):
            if d is None:
                miss.append(log)
            else:
                res[log] = d
            if k % 100 == 0:
                print(k, "/", len(logs), int(time.time() - t0), "s", flush=True)
    pickle.dump(res, open(OUT + ".tmp", "wb"))
    os.rename(OUT + ".tmp", OUT)
    n = sum(len(v) for v in res.values())
    print(
        "logs",
        len(res),
        "missing",
        len(miss),
        "samples",
        n,
        int(time.time() - t0),
        "s",
        "EXTRACT_DONE",
        flush=True,
    )
