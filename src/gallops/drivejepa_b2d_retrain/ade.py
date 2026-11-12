"""Sid's Experiment 1 numbers on the held-out moments: the released planner (generator + scorer, unchanged),
batch 1, fp32, no autocast (evalsets.forward_dump).
closest = best of 32 by ADE over 0.5-3 s; pick = the scorer's choice; variety = mean over the 496 pairs of
the mean waypoint distance over 0.5-3 s."""

import numpy as np
import torch

from gallops.drivejepa_b2d_retrain.frames import FrameCache, _to_tensor


@torch.no_grad()
def evaluate(agent, moments, tag, out_dir):
    fc = FrameCache()
    agent.eval()
    rows, dump = [], []
    iu = np.triu_indices(32, 1)
    for m in moments:
        older, newer = fc.frame(m["clip"], m["i"] - 1), fc.frame(m["clip"], m["i"])
        f = {
            "camera_feature_1": _to_tensor(newer),
            "camera_feature_2": _to_tensor(older),
            "lidar2img": m["lidar2img"],
            "img_shape": m["img_shape"],
            "ego_status": m["ego_status"],
        }
        f = {k: v.unsqueeze(0).cuda() for k, v in f.items()}
        f.update(past_ego_simulated_states=None, metric_cache=None, observation_interval=None)
        o = agent.forward(f)
        p = o["proposals"][0, :, :6, :2].double().cpu().numpy()  # (32, 6, 2)
        c = int(torch.argmax(o["pdm_score"], dim=1)[0])
        ade = np.linalg.norm(p - m["expert"][None], axis=-1).mean(-1)
        d = np.linalg.norm(p[:, None] - p[None], axis=-1).mean(-1)[iu]
        rows.append((float(ade.min()), float(ade[c]), float(d.mean())))
        dump.append(c)
    a = np.array(rows)
    res = dict(
        tag=tag,
        moments=len(a),
        closest_ade_3s=float(a[:, 0].mean()),
        pick_ade_3s=float(a[:, 1].mean()),
        variety_m=float(a[:, 2].mean()),
        closest_median=float(np.median(a[:, 0])),
        pick_median=float(np.median(a[:, 1])),
    )
    np.savez(out_dir / f"ade_{tag}.npz", rows=a, chosen=np.array(dump))
    return res
