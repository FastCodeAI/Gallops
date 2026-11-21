"""Correction (Nov 4) refit: paired CIs against k = 1 (the model's own view) at k = 2, 4, 8.

One-off, as run (Siddesh's choice 13 of 4-5 Nov, numbered in his working notes, not decision D13). Reuses Step 2's pooled.npy, pca.npz and targets.npz untouched and
the same code path as `gallops.memory.probe_control.run`: same rows, ridge / logistic fits with
log-grouped CV, Step 2's bootstrap draws (probe.log_weights, SEED, 2,000). Only V1 (= Step 2's
"images") and V4 (= "ego_only") are fitted, at k = 1, 2, 4, 8.

Before computing any difference it checks that every refitted value and CI equals Step 2's
probe.json (atol 1e-12); if one does not, it writes refit_k1_failed.json and exits non-zero.

    PYTHONPATH=<worktree>/src python refit_vs_k1.py   # writes {probe}/k1/refit_k1.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gallops.memory import probe, ticks
from gallops.memory.probe_control import variant_columns

KS = (1, 2, 4, 8)
VARIANTS = {"V1": "images", "V4": "ego_only"}
METRICS = ("speed_1s", "speed_2s", "speed_3s", "brake_logloss", "brake_ap")


def main() -> None:
    from gallops.config import SiteConfig

    t0 = time.time()
    cfg = SiteConfig.load()
    cache_root = Path(cfg.resolve("{data.navtrain.cache}"))
    probe_dir = cache_root / "probe"
    out = probe_dir / "k1"
    out.mkdir(parents=True, exist_ok=True)
    step2 = json.loads((probe_dir / "probe.json").read_text())
    reps = int(step2["reps"])

    table = ticks.load(cache_root / "ticks")
    split = table["split"]
    candidates = np.flatnonzero(table["graded"] & table["inputs_ok"] & (table["chain_pos"] >= max(KS)))
    with np.load(probe_dir / "targets.npz") as z:
        targets = {k: z[k] for k in z.files}
    order = np.argsort(targets["row"])
    targets = {k: v[order] for k, v in targets.items()}
    assert np.array_equal(targets["row"], candidates)
    keep = targets["has_future"]
    rows = candidates[keep]
    tg = {k: v[keep] for k, v in targets.items()}
    assert len(rows) == step2["rows"]["with_3s_future"], "not Step 2's row set"
    pooled = np.load(probe_dir / "pooled.npy", mmap_mode="r")
    with np.load(probe_dir / "pca.npz") as z:
        fit = {k: z[k] for k in z.files}
    ego = np.asarray(np.load(cache_root / "release" / "ego.npy", mmap_mode="r"))
    per_moment = np.concatenate([probe.pca_apply(pooled, fit), ego.astype(np.float32)], 1)
    print(f"loaded ({time.time() - t0:.0f} s)", flush=True)

    index = {k: probe.window_index(table, rows, k) for k in KS}  # leak check inside
    is_train, is_val = split[rows] == "train", split[rows] == "val"
    groups = table["log"][rows]
    y_speed = np.stack([tg[name] for name in probe.SPEED_NAMES], 1)
    y_brake = tg["onset"].astype(np.float64)
    val_logs, val_group = np.unique(groups[is_val], return_inverse=True)
    weights = probe.log_weights(len(val_logs), reps, probe.SEED)
    yv_s, yv_b, ev = y_speed[is_val], y_brake[is_val], tg["event"][is_val]

    preds = {}
    for k in KS:
        x_all = probe.window_features(per_moment, index[k])
        for v in VARIANTS:
            x = x_all[:, variant_columns(v, k)].astype(np.float64)
            xtr, xva = probe.standardise(x[is_train], x[is_val])
            r = probe.fit_ridge(xtr, y_speed[is_train], groups[is_train])
            lg = probe.fit_logit(xtr, y_brake[is_train], groups[is_train])
            preds[(v, k)] = (probe.predict_ridge(r, xva), lg["model"].predict_proba(xva)[:, 1])
            print(f"  {v} k={k} fitted ({time.time() - t0:.0f} s)", flush=True)

    def boot(mask, sp, pb):
        g = val_group[mask]
        cnt = probe.per_log_sums(g, np.ones(mask.sum()), len(val_logs))
        res = {}
        for t, name in enumerate(probe.SPEED_NAMES):
            sse = probe.per_log_sums(g, (sp[mask, t] - yv_s[mask, t]) ** 2, len(val_logs))
            res[name] = (float(np.sqrt(sse.sum() / cnt.sum())), np.sqrt((weights @ sse) / (weights @ cnt)))
        ll = probe.per_log_sums(g, probe.log_loss_terms(yv_b[mask], pb[mask]), len(val_logs))
        res["brake_logloss"] = (float(ll.sum() / cnt.sum()), (weights @ ll) / (weights @ cnt))
        row_w = weights[:, g]
        ap = np.array([probe.average_precision(yv_b[mask], pb[mask], row_w[r]) for r in range(reps)])
        res["brake_ap"] = (probe.average_precision(yv_b[mask], pb[mask]), ap)
        return res

    slices = {"all": np.ones(len(yv_b), bool), "events": ev}
    draws, reproduced = {}, {}
    for (v, k), (sp, pb) in preds.items():
        for sl, mask in slices.items():
            for metric, (point, per_draw) in boot(mask, sp, pb).items():
                draws[(sl, metric, v, k)] = (point, per_draw)
                theirs = step2["metrics"][sl][metric][f"ridge/{VARIANTS[v]}/k{k}"]
                reproduced[f"{v}/k{k}/{sl}/{metric}"] = bool(
                    np.isclose(point, theirs["value"], rtol=0, atol=1e-12)
                    and np.allclose(probe.ci(per_draw), theirs["ci95"], rtol=0, atol=1e-12)
                )
    if not all(reproduced.values()):
        bad = [k for k, ok in reproduced.items() if not ok]
        (out / "refit_k1_failed.json").write_text(json.dumps({"not_reproduced": bad}, indent=1))
        raise SystemExit(f"refit does not reproduce Step 2 ({len(bad)} of {len(reproduced)}): STOP")

    deltas = {}
    for sl in slices:
        for metric in METRICS:
            for v in VARIANTS:
                p1, d1 = draws[(sl, metric, v, 1)]
                for k in (2, 4, 8):
                    pk, dk = draws[(sl, metric, v, k)]
                    lo, hi = probe.ci(dk - d1)
                    deltas.setdefault(sl, {}).setdefault(metric, {})[f"{v}/k{k}-k1"] = {
                        "delta": pk - p1,
                        "ci95": [lo, hi],
                        "ci_excludes_0": bool(lo > 0 or hi < 0),
                    }
    result = {
        "reproduces_step2": {"all": True, "checked": len(reproduced)},
        "deltas_vs_k1": deltas,
        "reps": reps,
        "git_sha": probe._git_sha(),
        "step2_git_sha": step2.get("git_sha"),
        "total_s": round(time.time() - t0, 1),
    }
    (out / "refit_k1.json").write_text(json.dumps(result, indent=1) + "\n")
    ev_ = deltas["events"]
    for metric in ("speed_2s", "brake_logloss"):
        for v in VARIANTS:
            for k in (2, 4, 8):
                d = ev_[metric][f"{v}/k{k}-k1"]
                print(f"events {metric} {v} k{k}-k1 {d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
