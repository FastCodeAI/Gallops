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
  no-camera prior, the train-mean gap;
- from the rerun's step 5, every epoch: query collapse (`slot_numbers`) and the recall of the
  targets that matter (`recall_of`, with `importance.py`'s labels);
- from step 7, the plan's adopted metrics (§8.1, choice 11.4), until then computed in notebooks 04
  and 06: AP per class (`average_precision`), calibration of the confidence (`calibration`), the
  position error along the camera's ray and across it (`ray_errors`), and per-sample counts for
  intervals over resampled scenes or logs (`per_sample_counts`).
"""

from __future__ import annotations

import numpy as np

from gallops.drivejepa_perception.targets import NONE, NUM_CLASSES

THR, HIT_M = 0.5, 2.0
#: Range bins, metres. "Within 30 m" is the first two: the plan's limit and H04's (its seen rule kept
#: targets to 30 m). Until 4 Nov the middle bin ran to 35 m, so the number named "within 30 m" was
#: within 35 m; A0's as-run numbers and A1's epoch choice used it (README, "a readout bug").
RANGE_BINS = ((0.0, 20.0), (20.0, 30.0), (30.0, 50.0))
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


def match(conf, cls, xy, gxy, gcls, radius: float = HIT_M):
    """H04's greedy match: detections by confidence, each to the nearest free target of its class
    within `radius` (the readout's HIT_M unless an analysis asks for a looser pairing)."""
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
        if d[g] <= radius:
            hit_of_gt[g], tp[k] = q, True
    return det, tp, hit_of_gt


def lead(xy: np.ndarray, cls: np.ndarray):
    """Gap to the nearest vehicle ahead in the ego lane, or None."""
    ahead = (cls == 0) & (xy[:, 0] > 0) & (np.abs(xy[:, 1]) <= LANE_HALF_M)
    return float(xy[ahead, 0].min()) if ahead.any() else None


def slot_numbers(preds: list) -> dict:
    """Query collapse, read on decoded outputs (the rerun's plan §5.2, notebook 02 §8).

    Detections a sample; duplicate pairs a sample (two detections of one class whose centres lie
    within HIT_M of each other) and the share of detections in such a pair; how many slots ever fire,
    and the share of all detections the busiest eighth of the slots makes.
    """
    fired = np.stack([p["conf"] >= THR for p in preds])  # (samples, slots)
    pairs = in_pair = 0
    for pred, shown in zip(preds, fired):  # noqa: B905
        q = np.flatnonzero(shown)
        xy, cls = pred["xyz"][q, :2], pred["cls"][q]
        close = (np.linalg.norm(xy[:, None] - xy[None], axis=-1) <= HIT_M) & (cls[:, None] == cls[None])
        close = np.triu(close, k=1)
        pairs += int(close.sum())
        in_pair += int((close.any(0) | close.any(1)).sum())
    per_slot = fired.sum(0)
    detections = int(per_slot.sum())
    busiest = np.sort(per_slot)[::-1][: max(len(per_slot) // 8, 1)]
    return {
        "detections_per_sample": float(fired.sum(1).mean()),
        "duplicate_pairs_per_sample": pairs / len(preds),
        "duplicate_share": in_pair / detections if detections else float("nan"),
        "slots": int(fired.shape[1]),
        "slots_firing": int((per_slot > 0).sum()),
        "busiest_eighth_share": float(busiest.sum() / detections) if detections else float("nan"),
    }


#: Confidence floor for AP: every slot at or over it is ranked, not only those at THR (notebook 04 §5).
AP_FLOOR = 0.01
#: Recall points AP averages precision over (101, as COCO's interpolated AP).
AP_POINTS = np.linspace(0.0, 1.0, 101)
#: Confidence bins for calibration (notebook 06 §9): THR's bin is [0.5, 0.6).
CALIBRATION_BINS = np.linspace(0.1, 1.0, 10)


def average_precision(
    preds: list, gts: list, cls: int, radius: float = HIT_M, max_range: float = 30.0
) -> dict:
    """AP of one class within `max_range`: every slot of that class at AP_FLOOR or more, ranked by
    confidence over all samples, each a hit when the nearest free target of its class lies within
    `radius` (centre, x-y). Moved from notebooks 04 and 06 so steps 6 and 7 read it alike.
    Gives `ap`, the curve (`recall`, `precision`) and `targets`."""
    dets, gt_xy = [], []
    for i, (pred, gt) in enumerate(zip(preds, gts)):  # noqa: B905 -- Python 3.9
        m = gt["mask"] & (gt["cls"] == cls)
        g = gt["box"][m, :2].astype(np.float64)
        gt_xy.append(g[np.hypot(g[:, 0], g[:, 1]) < max_range])
        sel = (pred["cls"] == cls) & (pred["conf"] >= AP_FLOOR)
        xy = pred["xyz"][sel, :2].astype(np.float64)
        near = np.hypot(xy[:, 0], xy[:, 1]) < max_range
        dets += [(float(c), i, p) for c, p in zip(pred["conf"][sel][near], xy[near])]  # noqa: B905
    n_gt = sum(len(g) for g in gt_xy)
    dets.sort(key=lambda d: -d[0])
    used = [np.zeros(len(g), bool) for g in gt_xy]
    tp = np.zeros(len(dets))
    for k, (_, i, xy) in enumerate(dets):
        if not len(gt_xy[i]):
            continue
        d = np.hypot(*(gt_xy[i] - xy).T)
        d[used[i]] = np.inf
        g = int(np.argmin(d))
        if d[g] <= radius:
            used[i][g], tp[k] = True, 1.0
    hits = np.cumsum(tp)
    precision = hits / np.arange(1, len(tp) + 1) if len(tp) else np.zeros(0)
    recall = hits / max(n_gt, 1)
    ap = float(np.mean([precision[recall >= r].max() if (recall >= r).any() else 0.0 for r in AP_POINTS]))
    return {"ap": ap, "recall": recall, "precision": precision, "targets": int(n_gt)}


def calibration(preds: list, gts: list, bins: np.ndarray = CALIBRATION_BINS) -> dict:
    """How often a slot of a given confidence is right: every slot at bins[0] or more, matched to the
    nearest free target of its class within HIT_M in order of confidence (the readout's rule, applied
    below THR too). Gives each bin's centre, share right and count."""
    hit_by, n_by = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    for pred, gt in zip(preds, gts):  # noqa: B905
        m = gt["mask"]
        keep = np.flatnonzero(pred["conf"] >= bins[0])
        order = keep[np.argsort(-pred["conf"][keep], kind="stable")]
        free = np.ones(int(m.sum()), bool)
        gxy, gcls = gt["box"][m, :2].astype(np.float64), gt["cls"][m]
        for q in order:
            d = np.hypot(*(gxy - pred["xyz"][q, :2]).T) if len(gxy) else np.zeros(0)
            d[~free | (gcls != pred["cls"][q])] = np.inf
            ok = bool(len(d)) and float(d.min()) <= HIT_M
            if ok:
                free[int(np.argmin(d))] = False
            b = min(int(np.searchsorted(bins, pred["conf"][q], side="right")) - 1, len(bins) - 2)
            n_by[b] += 1
            hit_by[b] += ok
    return {"centre": (bins[:-1] + bins[1:]) / 2, "right": hit_by / np.maximum(n_by, 1), "count": n_by}


def camera_centre(projection: np.ndarray) -> np.ndarray:
    """The ego-frame point an ego -> image projection maps to the camera's own centre."""
    p = np.asarray(projection, dtype=np.float64)
    return -np.linalg.solve(p[:3, :3], p[:3, 3])


def ray_errors(preds: list, gts: list, projections: list, radius: float = 5.0) -> dict:
    """Each target's position error split along the camera's ray and across it (notebook 06 §7).

    Targets are paired with a detection of their class within `radius` by `match` (looser than the
    readout's 2 m, so a misplaced agent still pairs). Per pair: the target's range from the camera,
    its class, the error along the ray (depth: positive means placed too far), across it (lateral,
    positive to the left) and that lateral error as an angle seen from the camera."""
    out = {k: [] for k in ("range", "cls", "depth", "lateral", "direction_deg")}
    for pred, gt, projection in zip(preds, gts, projections):  # noqa: B905
        m = gt["mask"]
        gxy, gcls = gt["box"][m, :2].astype(np.float64), gt["cls"][m]
        _, _, pair = match(pred["conf"], pred["cls"], pred["xyz"][:, :2], gxy, gcls, radius=radius)
        centre = camera_centre(projection)[:2]
        for j in np.flatnonzero(pair >= 0):
            g, d = gxy[j], pred["xyz"][pair[j], :2].astype(np.float64)
            reach = float(np.linalg.norm(g - centre))
            ray = (g - centre) / reach
            err = d - g
            across = float(ray[0] * err[1] - ray[1] * err[0])
            out["range"].append(reach)
            out["cls"].append(int(gcls[j]))
            out["depth"].append(float(err @ ray))
            out["lateral"].append(across)
            out["direction_deg"].append(float(np.degrees(np.arctan2(across, reach))))
    return {k: np.asarray(v) for k, v in out.items()}


def per_sample_counts(preds: list, gts: list, cls: int = -1, max_range: float = 30.0) -> dict:
    """Per sample, the targets within `max_range` and how many were hit, and the detections there
    and how many were right (one class, or all with -1): what interval draws over scenes resample."""
    out = {k: np.zeros(len(preds), np.int64) for k in ("targets", "hits", "detections", "right")}
    for i, (pred, gt) in enumerate(zip(preds, gts)):  # noqa: B905
        m = gt["mask"]
        gxy, gcls = gt["box"][m, :2].astype(np.float64), gt["cls"][m]
        det, tp, hit = match(pred["conf"], pred["cls"], pred["xyz"][:, :2], gxy, gcls)
        g_in = np.hypot(gxy[:, 0], gxy[:, 1]) < max_range
        d_in = np.hypot(pred["xyz"][det, 0], pred["xyz"][det, 1]) < max_range
        if cls >= 0:
            g_in &= gcls == cls
            d_in &= pred["cls"][det] == cls
        out["targets"][i], out["hits"][i] = int(g_in.sum()), int((g_in & (hit >= 0)).sum())
        out["detections"][i], out["right"][i] = int(d_in.sum()), int((d_in & tp).sum())
    return out


def recall_of(preds: list, gts: list, subsets: dict) -> dict:
    """Recall at HIT_M of chosen targets, all ranges (to 50 m): `subsets` maps a name to one boolean
    array per sample over that sample's targets, in their order (the causal ones, the nearest; plan
    §3.8). Gives `recall_<name>` and `targets_<name>`."""
    found = []
    for pred, gt in zip(preds, gts):  # noqa: B905
        m = gt["mask"]
        gxy = gt["box"][m, :2].astype(np.float64)
        found.append(match(pred["conf"], pred["cls"], pred["xyz"][:, :2], gxy, gt["cls"][m])[2] >= 0)
    out = {}
    for name, masks in subsets.items():
        hit = np.concatenate([f[np.asarray(s, bool)] for f, s in zip(found, masks)])  # noqa: B905
        out[f"recall_{name}"] = float(hit.mean()) if len(hit) else float("nan")
        out[f"targets_{name}"] = int(len(hit))
    return out


def readout(preds: list, gts: list, prior_gap: float, detail: bool = False) -> dict:
    """Phase A's numbers over samples. `preds` from `decode`; `gts` from `targets.build`.

    With `detail`, also `out["detail"]`: per hit agent with a full path, its sample's index in
    `preds`, whether it moves and each guess's FDE at 1 s; per sample with a lead vehicle, its
    index and both gap errors. Paired SEs need these; the means above do not.
    """
    n_bins = len(RANGE_BINS)
    gt_n, gt_hit = np.zeros((n_bins, NUM_CLASSES)), np.zeros((n_bins, NUM_CLASSES))
    det_n, det_tp = np.zeros((n_bins, NUM_CLASSES)), np.zeros((n_bins, NUM_CLASSES))
    centre_err, vel_err = [], []
    moving_n = moving_hit = 0
    paths = {k: [] for k in ("model", "stationary", "cv_pred", "cv_label")}
    path_moving = []
    lead_err, lead_prior_err = [], []
    path_rows, lead_rows = [], []
    for index, (pred, gt) in enumerate(zip(preds, gts)):  # noqa: B905 -- Python 3.9
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
                path_rows.append(index)
        true_gap = lead(gxyz[:, :2], gcls)
        if true_gap is not None:
            shown = pred["conf"] >= THR
            guess = lead(pred["xyz"][shown, :2], pred["cls"][shown])
            lead_err.append(abs((guess if guess is not None else 50.0) - true_gap))
            lead_prior_err.append(abs(prior_gap - true_gap))
            lead_rows.append(index)

    def ratio(a, b):
        return float(a.sum() / b.sum()) if b.sum() else float("nan")

    within_30 = slice(0, 2)  # 0-20 and 20-30 m
    out = {
        "recall_2m": ratio(gt_hit, gt_n),
        "precision_2m": ratio(det_tp, det_n),
        "recall_2m_within_30m": ratio(gt_hit[within_30], gt_n[within_30]),
        "precision_2m_within_30m": ratio(det_tp[within_30], det_n[within_30]),
        "recall_by_range": [ratio(gt_hit[b], gt_n[b]) for b in range(n_bins)],
        "recall_by_class": [ratio(gt_hit[:, c], gt_n[:, c]) for c in range(NUM_CLASSES)],
        "recall_2m_within_30m_by_class": [
            ratio(gt_hit[within_30, c], gt_n[within_30, c]) for c in range(NUM_CLASSES)
        ],
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
    if detail:
        out["detail"] = {
            "path_row": np.asarray(path_rows, np.int64),
            "path_moving": moving,
            **{f"fde_{k}": np.array(v).reshape(-1, 2)[:, 1] for k, v in paths.items()},
            "lead_row": np.asarray(lead_rows, np.int64),
            "lead_err": np.asarray(lead_err),
            "lead_prior_err": np.asarray(lead_prior_err),
        }
    return out
