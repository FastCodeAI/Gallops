"""Drift check between two candidate dumps (Part F.5).

A dump maps (scene_id, keyframe) -> {"proposals": (32, 8, 3), "pdm_score": (32,),
"chosen": int, ...}.  Proposals are paired by index: proposal i is always decoded
from the same learned queries init_feature rows [8i, 8i+8)
(drive_jepa_model.py:33, :62; traj_refiner.py:24 reshapes (B, 32*8, d) -> (B, 32, 8, 3)),
and nothing between the decoder and the output sorts or re-indexes them
(bev_refiner.py only reshapes / permutes layout, :113-142; the pick is an argmax
over the fixed index, drive_jepa_model.py:85-91).
"""

from __future__ import annotations

import numpy as np


def drift(a: dict, b: dict) -> dict:
    assert a.keys() == b.keys(), "dumps cover different samples"
    rows = {}
    for k in a:
        pa, pb = np.asarray(a[k]["proposals"], dtype=np.float64), np.asarray(b[k]["proposals"], dtype=np.float64)
        sa, sb = np.asarray(a[k]["pdm_score"], dtype=np.float64), np.asarray(b[k]["pdm_score"], dtype=np.float64)
        rows[k] = {
            # metres: mean over the 32 x 8 waypoints of the xy distance between proposal i and proposal i
            "proposal_shift_m": float(np.linalg.norm(pa[..., :2] - pb[..., :2], axis=-1).mean()),
            "proposal_shift_max_m": float(np.linalg.norm(pa[..., :2] - pb[..., :2], axis=-1).max()),
            "grade_change_abs": float(np.abs(sa - sb).mean()),
            "chosen_changed": int(a[k]["chosen"]) != int(b[k]["chosen"]),
        }
    v = list(rows.values())
    summary = {
        "samples": len(v),
        "mean_proposal_shift_m": float(np.mean([r["proposal_shift_m"] for r in v])),
        "max_proposal_shift_m": float(np.max([r["proposal_shift_max_m"] for r in v])),
        "mean_grade_change": float(np.mean([r["grade_change_abs"] for r in v])),
        "chosen_changed": int(sum(r["chosen_changed"] for r in v)),
    }
    return {"summary": summary, "per_sample": rows}
