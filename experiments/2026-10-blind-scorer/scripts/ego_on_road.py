"""Is the logged ego footprint on the drivable raster? It drove there, so it should be, nearly always."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, "{gallops}/src")
from gallops.expert.drivable import DrivableArea
R = Path("{runs}")
CORNERS = np.array([[2.509, 0.925], [2.509, -0.925], [-1.509, -0.925], [-1.509, 0.925]])
by_loc = defaultdict(lambda: [0, 0, 0]); worst = []
for ep in sorted((R / "collect_nusc").glob("real-*/run_0")):
    scene = ep.parent.name.split("-")[1]
    ar = DrivableArea.load(R / "drivable_nusc" / f"scene-{scene}.npz")
    poses = json.loads((ep / "ego_poses.json").read_text())
    off = centre_off = 0
    for m in poses.values():
        m = np.asarray(m); pts = CORNERS @ m[:2, :2].T + m[:2, 3]
        off += not ar.contains(pts).all(); centre_off += not ar.contains(m[:2, 3][None])[0]
    s = by_loc[ar.location]; s[0] += len(poses); s[1] += off; s[2] += centre_off
    worst.append((off / len(poses), centre_off / len(poses), ar.location, scene, len(poses)))
for loc, (n, off, c) in sorted(by_loc.items()):
    print(f"{loc:22s} poses {n:6d}  a corner off-road {off / n:6.1%}  centre off-road {c / n:6.1%}")
worst.sort(reverse=True)
print("worst scenes (corner-off share, centre-off share, location, scene, poses):")
for w in worst[:12]: print(f"  {w[0]:6.1%} {w[1]:6.1%} {w[2]:22s} {w[3]} {w[4]}")
print("scenes with any corner off:", sum(w[0] > 0 for w in worst), "of", len(worst))
