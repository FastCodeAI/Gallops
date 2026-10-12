"""Does the sub-score scorer see the car, or the place and moment?

GT drives: the car follows the human's logged path, so at one timestamp every run and scenario of a scene has the
ego at exactly the same pose and speed. The straight-ahead swerves (candidates 46-59) depend only on pose and
speed, so they are the same plans. Only the inserted car differs. For pairs (same scene, same timestamp, same
plan, different episode) whose no-collision labels differ, a scorer that sees the car gives the clear one the
higher answer; a scorer that has learned the place gives both the same answer (paired accuracy 0.5).
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

V2 = Path("{dj_cache}/v2")
W = Path("{weights}/drivejepa_subscore/realmask-attngeo-v1/step_000000020")
sys.path.insert(0, "{paddock}/src")
torch.set_num_threads(2)

z = np.load(V2 / "probe_input.npz")
episode = z["episode"]; src = np.array([e.split("/")[0] for e in episode])
rows = np.where(src == "gt")[0]
print("gt ticks", len(rows), flush=True)
plans = z["plans"][rows]; hit5 = z["hit5_at"][rows]; in_frame = z["in_frame"][rows]; speed = z["speed"][rows]
scene = z["scene"][rows]; shape = z["shape"][rows]; ep = episode[rows]

# timestamps: each episode's ticks are contiguous and in file order
ts = np.empty(len(rows), np.int64); i = 0
while i < len(rows):
    e = ep[i]; t = np.load(V2 / f"{e}.npz")["timestamp"]; n = len(t)
    assert (ep[i:i + n] == e).all(), e
    ts[i:i + n] = t; i += n

# held-out predictions: the fold where the scene was held out, mean of 5 seeds
pos = {g: k for k, g in enumerate(rows)}
held = np.zeros((len(rows), 64)); cnt = 0
for s in range(5):
    f = np.load(V2 / f"probe_attngeo_realmask_s{s}.logits.npz"); got = np.zeros((len(rows), 64)); seen = np.zeros(len(rows), bool)
    for sc in ("0053", "0058", "0064", "0070", "0075"):
        for j, g in enumerate(f[f"test_idx_{sc}"]):
            if g in pos: got[pos[g]] = 1 / (1 + np.exp(-f[f"logits_{sc}"][j, :, 0])); seen[pos[g]] = True
    assert seen.all(); held += got; cnt += 1
held /= cnt

# trained-on predictions: the final model (all five scenes), exactly as the adapter scores
from paddock.adapters.models.drivejepa_subscore.policy import build_scorer
dev = "cuda"
own = torch.from_numpy(z["own"][rows]); probe = torch.from_numpy(z["probe"][rows])
geo = torch.from_numpy(np.concatenate([plans[..., :2].reshape(len(rows), 64, 16) / 20.0, plans[..., 2],
                                       np.repeat(speed[:, None, None] / 10.0, 64, 1)], -1).astype(np.float16))
meta = json.loads((W / "scorer.json").read_text()); final = np.zeros((len(rows), 64))
for s in meta["seeds"]:
    m = build_scorer(torch); m.load_state_dict(torch.load(W / f"seed_{s}.pt", map_location="cpu")); m = m.to(dev).eval()
    with torch.no_grad():
        for a in range(0, len(rows), 256):
            o = m(own[a:a + 256].to(dev).float(), probe[a:a + 256].to(dev).float(), geo[a:a + 256].to(dev).float())
            final[a:a + 256] += torch.sigmoid(o[..., 0]).cpu().numpy()
final /= len(meta["seeds"])

label = np.isnan(hit5).astype(float)  # 1 = no collision within 5 s
np.savez(Path(sys.argv[1]), ts=ts, scene=scene, shape=shape, ep=ep, plans=plans, hit5=hit5, in_frame=in_frame,
         speed=speed, held=held, final=final, label=label)
print("saved", sys.argv[1], flush=True)
