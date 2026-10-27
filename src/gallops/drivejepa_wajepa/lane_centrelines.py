"""Stage lane-centrelines (Run 3A map labels, once): per nuPlan city map, every lane / lane connector
centreline (baseline path, resampled at <= 1 m, float64 [x, y, heading]) and the lane graph (successors /
predecessors through lane connectors). Writes <out>/<map_name>.pkl = {"lines": {id: (n, 3)}, "succ": {id:
[id]}, "pred": {id: [id]}}; ids as the drivable-area map's tokens (str of the fid). Never overwrites. Run
3A's labels read them from RUN3A_CENTRELINES. Ported from paddock-ops run3a/labels/lane_centrelines.py; only
this docstring changed.

    NUPLAN_MAPS_ROOT=<maps> python -m gallops.drivejepa_wajepa.lane_centrelines <out_dir>"""

import os
import pickle
import sys
import time

import numpy as np
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

OUT = sys.argv[1]
root = os.environ["NUPLAN_MAPS_ROOT"]
for name in ("us-nv-las-vegas-strip", "us-ma-boston", "us-pa-pittsburgh-hazelwood", "sg-one-north"):
    path = f"{OUT}/{name}.pkl"
    if os.path.exists(path):
        print("exists, skipped", path)
        continue
    t = time.time()
    m = get_maps_api(root, "nuplan-maps-v1.0", name)
    bp = m._get_vector_map_layer(L.BASELINE_PATHS)
    lc = m._get_vector_map_layer(L.LANE_CONNECTOR)
    lines = {}
    for _, row in bp.iterrows():
        fid = row["lane_fid"] if not np.isnan(row["lane_fid"]) else row["lane_connector_fid"]
        xy = np.asarray(row["geometry"].coords)[:, :2]
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        if d[-1] < 1e-3:
            continue
        s = np.linspace(0, d[-1], max(int(np.ceil(d[-1])) + 1, 2))
        x, y = np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])
        h = np.arctan2(np.gradient(y), np.gradient(x))
        lines[str(int(fid))] = np.stack([x, y, h], -1)
    succ, pred = {}, {}
    cid = lc["fid"] if "fid" in lc.columns else lc.index
    for c, a, b in zip(cid, lc["entry_lane_fid"], lc["exit_lane_fid"]):  # noqa: B905 -- Python 3.9
        c, a, b = str(int(c)), str(int(a)), str(int(b))
        succ.setdefault(a, []).append(c)
        pred.setdefault(c, []).append(a)
        succ.setdefault(c, []).append(b)
        pred.setdefault(b, []).append(c)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"lines": lines, "succ": succ, "pred": pred}, f)
    os.rename(tmp, path)
    print(name, len(lines), "lines", len(lc), "connectors", f"{time.time() - t:.0f} s", flush=True)
print("CENTRELINES_DONE")
