"""Phase A's readouts: detection, each agent's 1 s path, and the lead vehicle (PREDICTIONS.md P1-P5).

H04's conventions, so the numbers read alike (agent_decoder_data.py, branch drivejepa-wajepa/run3a):
a detection is a slot with confidence 1 - p(none) >= THR, labelled with the most likely of the three
classes; it hits a target of its class within HIT_M metres (centre, x-y), greedily by confidence.
"Moving" is faster than 1 m/s.

Added here:
- recall and precision by range bin;
- path ADE and FDE at 1 s for hit agents, against three baselines: stationary at the predicted
  centre, constant velocity from the predicted velocity, and constant velocity from the label's
  velocity at the label's centre (a kinematic reference the side-car must approach);
- the lead vehicle (the nearest vehicle ahead with |y| <= LANE_HALF_M): gap error against a
  no-camera prior, the train-mean gap.
"""

from __future__ import annotations

import numpy as np

from gallops.drivejepa_perception.targets import NONE, NUM_CLASSES

THR, HIT_M = 0.5, 2.0
RANGE_BINS = ((0.0, 20.0), (20.0, 35.0), (35.0, 50.0))
LANE_HALF_M = 1.75
MOVING_MPS = 1.0
DT = (0.5, 1.0)


def decode(out: dict) -> dict:
    """The last layer's outputs (numpy, one sample) -> detections in the label's terms."""
    prob = np.exp(out["logits"] - out["logits"].max(-1, keepdims=True))
    prob /= prob.sum(-1, keepdims=True)
    centre = out["box"][:, :3]
    return {
        "conf": 1.0 - prob[:, NONE],
        "cls": prob[:, :NUM_CLASSES].argmax(-1),
        "xyz": centre,
        "vel": out["vel"],
        "path": centre[:, None, :2] + out["path"],
    }


def match(conf, cls, xy, gxy, gcls):
    """H04's greedy match: detections by confidence, each to the nearest free target of its class."""
    det = np.flatnonzero(conf >= THR)
    det = det[np.argsort(-conf[det], kind="stable")]
    hit_of_gt = np.full(len(gxy), -1)
    tp = np.zeros(len(det), bool)
    for k, q in enumerate(det):
        free = (gcls == cls[q]) & (hit_of_gt < 0)
        if not free.any():
            continue
        d = np.hypot(gxy[:, 0] - xy[q, 0], gxy[:, 1] - xy[q, 1])
        d[~free] = np.inf
        g = int(np.argmin(d))
        if d[g] <= HIT_M:
            hit_of_gt[g], tp[k] = q, True
    return det, tp, hit_of_gt


def lead(xy: np.ndarray, cls: np.ndarray):
    """Gap to the nearest vehicle ahead in the ego lane, or None."""
    ahead = (cls == 0) & (xy[:, 0] > 0) & (np.abs(xy[:, 1]) <= LANE_HALF_M)
    return float(xy[ahead, 0].min()) if ahead.any() else None


def readout(preds: list, gts: list, prior_gap: float) -> dict:
    """Phase A's numbers over samples. `preds` from `decode`; `gts` from `targets.build`."""
    n_bins = len(RANGE_BINS)
    gt_n, gt_hit = np.zeros((n_bins, NUM_CLASSES)), np.zeros((n_bins, NUM_CLASSES))
    det_n, det_tp = np.zeros((n_bins, NUM_CLASSES)), np.zeros((n_bins, NUM_CLASSES))
    centre_err, vel_err = [], []
    moving_n = moving_hit = 0
    paths = {k: [] for k in ("model", "stationary", "cv_pred", "cv_label")}
    path_moving = []
    lead_err, lead_prior_err = [], []
    for pred, gt in zip(preds, gts):  # noqa: B905 -- Python 3.9
        m = gt["mask"]
        gxyz, gcls, gvel = gt["box"][m, :3].astype(np.float64), gt["cls"][m], gt["vel"][m].astype(np.float64)
        gpath = gt["box"][m, None, :2] + gt["path"][m]
        gok = gt["path_ok"][m]
        det, tp, hit = match(pred["conf"], pred["cls"], pred["xyz"][:, :2], gxyz[:, :2], gcls)
        grange = np.hypot(gxyz[:, 0], gxyz[:, 1])
        drange = np.hypot(pred["xyz"][det, 0], pred["xyz"][det, 1])
        for b, (lo, hi) in enumerate(RANGE_BINS):
            g_in = (grange >= lo) & (grange < hi)
            np.add.at(gt_n[b], gcls[g_in], 1)
            np.add.at(gt_hit[b], gcls[g_in & (hit >= 0)], 1)
            d_in = (drange >= lo) & (drange < hi)
            np.add.at(det_n[b], pred["cls"][det][d_in], 1)
            np.add.at(det_tp[b], pred["cls"][det][d_in & tp], 1)
        speed = np.hypot(gvel[:, 0], gvel[:, 1])
        moving = (gcls == 0) & (speed > MOVING_MPS) & (grange <= 20.0)
        moving_n += int(moving.sum())
        moving_hit += int((moving & (hit >= 0)).sum())
        for g in np.flatnonzero(hit >= 0):
            q = hit[g]
            centre_err.append(float(np.hypot(*(pred["xyz"][q, :2] - gxyz[g, :2]))))
            vel_err.append(float(np.hypot(*(pred["vel"][q] - gvel[g]))))
            if gok[g].all():
                truth = gpath[g]
                guesses = {
                    "model": pred["path"][q],
                    "stationary": np.repeat(pred["xyz"][q, None, :2], 2, axis=0),
                    "cv_pred": pred["xyz"][q, None, :2] + np.outer(DT, pred["vel"][q]),
                    "cv_label": gxyz[g, None, :2] + np.outer(DT, gvel[g]),
                }
                for key, guess in guesses.items():
                    error = np.hypot(*(guess - truth).T)
                    paths[key].append((float(error.mean()), float(error[-1])))
                path_moving.append(bool(speed[g] > MOVING_MPS))
        true_gap = lead(gxyz[:, :2], gcls)
        if true_gap is not None:
            shown = pred["conf"] >= THR
            guess = lead(pred["xyz"][shown, :2], pred["cls"][shown])
            lead_err.append(abs((guess if guess is not None else 50.0) - true_gap))
            lead_prior_err.append(abs(prior_gap - true_gap))

    def ratio(a, b):
        return float(a.sum() / b.sum()) if b.sum() else float("nan")

    within_30 = slice(0, 2)
    out = {
        "recall_2m": ratio(gt_hit, gt_n),
        "precision_2m": ratio(det_tp, det_n),
        "recall_2m_within_30m": ratio(gt_hit[within_30], gt_n[within_30]),
        "precision_2m_within_30m": ratio(det_tp[within_30], det_n[within_30]),
        "recall_by_range": [ratio(gt_hit[b], gt_n[b]) for b in range(n_bins)],
        "recall_by_class": [ratio(gt_hit[:, c], gt_n[:, c]) for c in range(NUM_CLASSES)],
        "targets": int(gt_n.sum()),
        "centre_error_m": float(np.mean(centre_err)) if centre_err else float("nan"),
        "velocity_error_mps": float(np.mean(vel_err)) if vel_err else float("nan"),
        "moving_vehicles_20m_recall": moving_hit / moving_n if moving_n else float("nan"),
        "lead_gap_mae_m": float(np.mean(lead_err)) if lead_err else float("nan"),
        "lead_gap_prior_mae_m": float(np.mean(lead_prior_err)) if lead_prior_err else float("nan"),
        "lead_samples": len(lead_err),
    }
    moving = np.array(path_moving, dtype=bool)
    for key, values in paths.items():
        values = np.array(values).reshape(-1, 2)
        for name, sel in (("moving", moving), ("static", ~moving)):
            picked = values[sel] if len(values) else values
            out[f"path_fde_1s_{key}_{name}"] = float(picked[:, 1].mean()) if len(picked) else float("nan")
            out[f"path_ade_1s_{key}_{name}"] = float(picked[:, 0].mean()) if len(picked) else float("nan")
    out["path_hits_moving"] = int(moving.sum())
    out["path_hits_static"] = int((~moving).sum())
    return out
