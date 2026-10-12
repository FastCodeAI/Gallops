"""plans.npz: the 64 plans and ego speed per tick, in exactly the order scorer_folds.load builds folds_input."""
import numpy as np
from pathlib import Path
cache = Path("{dj_cache}/v1")
plans, speed, episodes = [], [], []
for src_dir in sorted(p for p in cache.iterdir() if p.is_dir()):
    for f in sorted(src_dir.glob("*.npz")):
        if f.name.endswith(".labels.npz") or ".tmp" in f.name:
            continue
        if not f.with_name(f.name.replace(".npz", ".labels.npz")).exists():
            continue
        z = np.load(f)
        plans.append(np.concatenate([z["natives"], z["vocab"]], 1).astype(np.float32))
        speed.append(z["speed"].astype(np.float32))
        episodes += [f"{src_dir.name}/{f.stem}"] * len(z["speed"])
d = np.load(cache / "folds_input.npz")
assert list(d["episode"]) == episodes, "episode order differs from folds_input"
np.savez(cache / "plans.npz", plans=np.concatenate(plans), speed=np.concatenate(speed))
print("ok", sum(len(s) for s in speed))
