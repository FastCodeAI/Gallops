"""Agent decoder (22 Oct): cache dataset, split and held-out metrics for train_agent_decoder.

Split: the fixed held-out list wa-jepa/step2/preflight/heldout_logs.txt (51 navtrain logs, sha1(log) % 20 ==
0, made 15 Oct;
kept for later runs); training = every other sample of run 2's token cache. navtest is never used.
Metrics, final layer, on the held-out samples, against ALL their agents within 32 m in x and y
(navtrain_agents.pkl, so
the 64-agent cut does not count against the decoder):
  detection  confidence = 1 - p(none) >= 0.5, class = argmax of the 3 agent classes; greedy by confidence, a
             detection
             is a hit when an unmatched true agent of its class is within 2 m (centre, x-y); recall /
             precision by region
             (bearing of the true / predicted centre: front |b| <= 45 deg, behind |b| >= 135 deg, beside
             between) and by
             class; centre error (x-y, m) and velocity error (|v_pred - v_true|, m/s) over hits; recall of
             moving
             vehicles (|v| > 1 m/s) within 20 m.
  E1         per camera, E1's seen rule (centre projects inside the 512x256 image, depth > 0, <= 30 m) on
             true and on
             predicted centres. Presence score = highest confidence among the slots the camera sees (0 if
             none); AUC vs
             "a true agent is seen". On samples with a seen true agent: the camera's nearest detection (>=
             0.5; if none,
             its most confident slot) gives distance error vs the nearest seen true agent and lane accuracy
             (own |y| <=
             1.75, left 1.75..5.25, right -5.25..-1.75, further); b0 closing speed: nearest detected vehicle
             seen by b0
             (if none, the most confident vehicle-class slot b0 sees), vx - v_ego, vs the nearest true
             vehicle b0 sees.
"""

import gzip
import math
import os
import pickle

import numpy as np
import torch

# Site paths from the environment (the recipe sets them); as run, constants naming that machine's paths.
CACHE = os.environ.get("WAJEPA_TOKEN_CACHE", "")
AGENTS = os.environ.get("NAVTRAIN_AGENTS", "")
HELDOUT = os.environ.get("HELDOUT_LOGS", "")
CLASSES = ("vehicle", "pedestrian", "bicycle")
CAMS = ("b0", "f0", "l0", "r0")
TO_512 = np.array([[2 / 3, 0, 0, 0], [0, 0.625, -7, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
THR, HIT_M = 0.5, 2.0


def load(p):
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def split(agents):
    held = {line.strip() for line in open(HELDOUT) if line.strip()}
    keys = [f"{lg}/{t}" for lg in sorted(agents) for t in sorted(agents[lg])]
    return [k for k in keys if k.split("/")[0] not in held], [k for k in keys if k.split("/")[0] in held]


class CacheSet(torch.utils.data.Dataset):
    def __init__(self, keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        d = f"{CACHE}/{self.keys[i]}"
        f = load(f"{d}/drivejepa4v_feature.gz")
        t = load(f"{d}/agent_targets.gz")
        t.pop("agent_count")
        return dict(
            t,
            tokens=load(f"{d}/wajepa_tokens.gz")["wajepa_tokens"],
            lidar2img=f["lidar2img"].float(),
            v_ego=f["ego_status"][-1][3].float(),
            idx=i,
        )


@torch.no_grad()
def predict(model, loader, dev="cuda"):
    """-> dict of numpy arrays over the loader's samples (final layer): conf, cls, box (x, y, z), vel; plus
    lidar2img, v_ego."""
    model.eval()
    acc = {k: [] for k in ("idx", "conf", "cls", "xyz", "vel", "lidar2img", "v_ego")}
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(b["tokens"].to(dev, non_blocking=True), b["lidar2img"].to(dev))[-1]
        p = out["logits"].softmax(-1)
        acc["conf"].append((1 - p[..., -1]).cpu())
        acc["cls"].append(p[..., :-1].argmax(-1).cpu())
        acc["xyz"].append(out["box"][..., :3].cpu())
        acc["vel"].append(out["vel"].cpu())
        acc["lidar2img"].append(b["lidar2img"])
        acc["v_ego"].append(b["v_ego"])
        acc["idx"].append(b["idx"])
    model.train()
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def region(x, y):
    b = np.degrees(np.abs(np.arctan2(y, x)))
    return np.where(b <= 45, 0, np.where(b >= 135, 2, 1))  # front, beside, behind


REGIONS = ("front", "beside", "behind")


def lane_of(y):
    return 0 if abs(y) <= 1.75 else 1 if 1.75 < y <= 5.25 else 2 if -5.25 <= y < -1.75 else 3


def gt_of(agents, key):
    lg, t = key.split("/")
    s = agents[lg][t]
    b, v = s["boxes"], s["vel"]
    names = np.array(s["names"])
    m = np.isin(names, CLASSES) & (np.abs(b[:, 0]) <= 32) & (np.abs(b[:, 1]) <= 32)
    return b[m, :3], np.array([CLASSES.index(n) for n in names[m]], np.int64), v[m, :2]


def match(conf, cls, xy, gxy, gcls):
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


def seen(P, xyz):
    """E1's rule: (N, 3) centres -> (4, N) seen by each camera."""
    h = np.einsum("cij,nj->cni", P, np.c_[xyz, np.ones(len(xyz))])
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = h[..., 0] / h[..., 2], h[..., 1] / h[..., 2]
    return (
        (h[..., 2] > 0)
        & (u >= 0)
        & (u < 512)
        & (v >= 0)
        & (v < 256)
        & (np.hypot(xyz[:, 0], xyz[:, 1]) <= 30)[None]
    )


def auc(score, y):
    from scipy.stats import rankdata

    y = np.asarray(y, bool)
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def metrics(pred, keys, agents):
    """pred from predict(); keys aligned with pred['idx']."""
    R = {
        "gt": np.zeros((3, 3)),
        "hit": np.zeros((3, 3)),
        "det": np.zeros((3, 3)),
        "tp": np.zeros((3, 3)),
    }  # region x class
    ctr_err, vel_err = [], []
    mv_n = mv_hit = mv_b_n = mv_b_hit = 0
    pres_s, pres_y = [[] for _ in CAMS], [[] for _ in CAMS]
    dist_err, lane_ok, n_noslot = [[] for _ in CAMS], [[] for _ in CAMS], np.zeros(4, int)
    close_err = []
    for n, i in enumerate(pred["idx"]):
        gxyz, gcls, gvel = gt_of(agents, keys[i])
        conf, cls, xyz, vel = pred["conf"][n], pred["cls"][n], pred["xyz"][n], pred["vel"][n]
        det, tp, hit = match(conf, cls, xyz[:, :2], gxyz[:, :2], gcls)
        greg = region(gxyz[:, 0], gxyz[:, 1])
        np.add.at(R["gt"], (greg, gcls), 1)
        np.add.at(R["hit"], (greg[hit >= 0], gcls[hit >= 0]), 1)
        dreg = region(xyz[det, 0], xyz[det, 1])
        np.add.at(R["det"], (dreg, cls[det]), 1)
        np.add.at(R["tp"], (dreg[tp], cls[det][tp]), 1)
        for g in np.flatnonzero(hit >= 0):
            q = hit[g]
            ctr_err.append(math.hypot(*(xyz[q, :2] - gxyz[g, :2])))
            vel_err.append(math.hypot(*(vel[q] - gvel[g])))
        mv = (gcls == 0) & (np.hypot(gvel[:, 0], gvel[:, 1]) > 1) & (np.hypot(gxyz[:, 0], gxyz[:, 1]) <= 20)
        mv_n += mv.sum()
        mv_hit += (mv & (hit >= 0)).sum()
        mv_b_n += (mv & (greg == 2)).sum()
        mv_b_hit += (mv & (greg == 2) & (hit >= 0)).sum()
        # E1's questions
        P = TO_512 @ pred["lidar2img"][n].astype(np.float64)
        gs, ps = seen(P, gxyz), seen(P, xyz)
        gr, pr = np.hypot(gxyz[:, 0], gxyz[:, 1]), np.hypot(xyz[:, 0], xyz[:, 1])
        for ci in range(4):
            pres_y[ci].append(gs[ci].any())
            pres_s[ci].append(conf[ps[ci]].max() if ps[ci].any() else 0.0)
            if not gs[ci].any():
                continue
            g = np.flatnonzero(gs[ci])[np.argmin(gr[gs[ci]])]
            cand = np.flatnonzero(ps[ci] & (conf >= THR))
            if len(cand):
                q = cand[np.argmin(pr[cand])]
            elif ps[ci].any():
                q = np.flatnonzero(ps[ci])[np.argmax(conf[ps[ci]])]
            else:
                n_noslot[ci] += 1
                continue
            dist_err[ci].append(abs(pr[q] - gr[g]))
            lane_ok[ci].append(lane_of(xyz[q, 1]) == lane_of(gxyz[g, 1]))
        gv = np.flatnonzero(gs[0] & (gcls == 0))
        if len(gv):
            g = gv[np.argmin(gr[gv])]
            pv = ps[0] & (cls == 0)
            cand = np.flatnonzero(pv & (conf >= THR))
            q = (
                cand[np.argmin(pr[cand])]
                if len(cand)
                else (np.flatnonzero(pv)[np.argmax(conf[pv])] if pv.any() else None)
            )
            if q is not None:
                close_err.append(abs((vel[q, 0] - pred["v_ego"][n]) - (gvel[g, 0] - pred["v_ego"][n])))
    rd = lambda x, k=3: round(float(x), k)  # noqa: E731
    out = {"n_samples": int(len(pred["idx"])), "threshold": THR, "hit_m": HIT_M}
    sl = {"all": (slice(None), slice(None))}
    sl.update({r: (ri, slice(None)) for ri, r in enumerate(REGIONS)})
    sl.update({c: (slice(None), ci) for ci, c in enumerate(CLASSES)})
    out["detection"] = {
        k: {
            "recall": rd(R["hit"][s].sum() / max(R["gt"][s].sum(), 1)),
            "precision": rd(R["tp"][s].sum() / max(R["det"][s].sum(), 1)),
            "n_true": int(R["gt"][s].sum()),
            "n_det": int(R["det"][s].sum()),
        }
        for k, s in sl.items()
    }
    out["detection_region_class"] = {
        f"{r}/{c}": {
            "recall": rd(R["hit"][ri, ci] / max(R["gt"][ri, ci], 1)),
            "precision": rd(R["tp"][ri, ci] / max(R["det"][ri, ci], 1)),
            "n_true": int(R["gt"][ri, ci]),
        }
        for ri, r in enumerate(REGIONS)
        for ci, c in enumerate(CLASSES)
    }
    out["centre_err_m"] = (
        {"mean": rd(np.mean(ctr_err)), "median": rd(np.median(ctr_err))} if ctr_err else None
    )
    out["velocity_err_mps"] = (
        {"mean": rd(np.mean(vel_err)), "median": rd(np.median(vel_err))} if vel_err else None
    )
    out["moving_vehicles_20m"] = {
        "recall": rd(mv_hit / max(mv_n, 1)),
        "n": int(mv_n),
        "behind_recall": rd(mv_b_hit / max(mv_b_n, 1)),
        "behind_n": int(mv_b_n),
    }
    out["e1"] = {
        c: {
            "presence_auc": rd(auc(pres_s[ci], pres_y[ci]), 4),
            "dist_mae_m": rd(np.mean(dist_err[ci]), 2) if dist_err[ci] else None,
            "dist_median_m": rd(np.median(dist_err[ci]), 2) if dist_err[ci] else None,
            "lane_acc": rd(np.mean(lane_ok[ci]), 4) if lane_ok[ci] else None,
            "n_pos": int(np.sum(pres_y[ci])),
            "n_no_slot_in_view": int(n_noslot[ci]),
        }
        for ci, c in enumerate(CAMS)
    }
    out["e1"]["b0"]["closing_mae_mps"] = rd(np.mean(close_err)) if close_err else None
    out["e1"]["b0"]["closing_n"] = len(close_err)
    return out
