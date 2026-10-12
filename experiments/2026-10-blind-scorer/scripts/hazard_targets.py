"""Per tick of cache v2 (probe_input order): the target car's position in the ego frame at +1..+4 s.
(T, 4, 3) = [present, x, y]; present = the car exists then and is within 100 m."""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, "{gallops}/src")
from gallops.expert.grader import actor_at
C = Path("{dj_cache}/v2")
P = np.load(C / "probe_input.npz"); out = []; names = []
for src in sorted(p for p in C.iterdir() if p.is_dir() and not p.name.startswith("__")):
    for f in sorted(src.glob("*.npz")):
        if f.name.endswith(".labels.npz") or ".tmp" in f.name or not f.with_name(f.name.replace(".npz", ".labels.npz")).exists(): continue
        z = np.load(f); ep = Path(str(z["frame_dir"])); poses = json.loads((ep / "ego_poses.json").read_text())
        tg = [a for a in json.loads((ep / "actors.json").read_text()) if a.get("is_target_actor")]
        for ts in z["timestamp"]:
            e2w = np.asarray(poses[str(int(ts))]); rot, org = e2w[:2, :2], e2w[:2, 3]; row = np.zeros((4, 3), np.float32)
            for k in range(4):
                st = actor_at(tg[0], float(ts) + (k + 1) * 1e6) if tg else None
                if st is not None:
                    xy = (st[0] - org) @ rot
                    if np.hypot(*xy) < 100: row[k] = [1, xy[0], xy[1]]
            out.append(row); names.append(f"{src.name}/{f.stem}")
assert names == list(P["episode"]), "order differs from probe_input"
np.save(C / "hazard_targets.npy", np.stack(out)); print("saved", len(out), "present share", np.stack(out)[..., 0].mean().round(3))
