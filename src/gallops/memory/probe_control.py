"""Probe control (c): is the gain of a longer window the camera's past or the car state's past?

    python -m gallops.memory.probe_control run [--probe DIR] [--reps 2000]   # CPU only

Step 2's probe (`probe.py`) found one pass of its GO rule: ridge, speed at +2 s, k = 4 vs k = 2.
Its ego-only control gained more over the same window, so the gain may be the car state's
history rather than the camera's. This control keeps the camera's present in every variant and
varies only what has history:

    V1  camera history + car history over k    (Step 2's "images" features)
    V2  camera now only + car history over k
    V3  camera history over k + car now only
    V4  car history over k, no camera          (Step 2's "ego-only" features)

"Now" is the newest moment's block of the window, "history over k" all k blocks (newest first,
`probe.window_features`). Same rows, val logs, targets, PCA features, ridge / logistic fits with
log-grouped CV and log-clustered bootstrap (same seed) as Step 2: pooled.npy, pca.npz and
targets.npz are read from Step 2's folder, never recomputed. V1 and V4 are refitted, because the
paired differences need their per-row predictions; their metrics must reproduce Step 2's
probe.json exactly, which `run` checks before writing anything else.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gallops.compat import zip_strict
from gallops.memory import probe

KS = (2, 4, 8)
VARIANTS = ("V1", "V2", "V3", "V4")
DESCRIPTION = {
    "V1": "camera history + car history over k",
    "V2": "camera now only + car history over k",
    "V3": "camera history over k + car now only",
    "V4": "car history over k, no camera",
}
#: Step 2's names for the two variants it already ran.
STEP2_NAME = {"V1": "images", "V4": "ego_only"}


def variant_columns(variant: str, k: int) -> np.ndarray:
    """Columns of a k-moment window (blocks of camera PCA_DIM then car EGO_DIM, newest first)."""
    block = probe.PCA_DIM + probe.EGO_DIM
    camera = [np.arange(probe.PCA_DIM) + j * block for j in range(k)]
    car = [np.arange(probe.PCA_DIM, block) + j * block for j in range(k)]
    parts = {
        "V1": [c for pair in zip_strict(camera, car) for c in pair],
        "V2": [camera[0], *car],
        "V3": [*camera, car[0]],
        "V4": car,
    }[variant]
    return np.sort(np.concatenate(parts))


def run(cfg, probe_dir: Path, out: Path, reps: int) -> dict:
    from gallops.memory import ticks

    timing: dict = {}
    t0 = time.time()
    for name in ("pooled.npy", "pca.npz", "targets.npz", "probe.json"):
        if not (probe_dir / name).exists():
            raise FileNotFoundError(
                f"{probe_dir / name}: control (c) reuses Step 2's outputs, run the probe first"
            )
    step2 = json.loads((probe_dir / "probe.json").read_text())
    cache_root = Path(cfg.resolve("{data.navtrain.cache}"))
    table = ticks.load(cache_root / "ticks")
    split = table["split"]
    candidates = np.flatnonzero(
        table["graded"] & table["inputs_ok"] & (table["chain_pos"] >= max(probe.WINDOWS))
    )
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
    timing["load_s"] = round(time.time() - t0, 1)

    t1 = time.time()
    index = {k: probe.window_index(table, rows, k) for k in KS}  # the leak check runs inside
    timing["windows_and_leak_check_s"] = round(time.time() - t1, 1)

    is_train, is_val = split[rows] == "train", split[rows] == "val"
    groups = table["log"][rows]
    y_speed = np.stack([tg[name] for name in probe.SPEED_NAMES], 1)
    y_brake = tg["onset"].astype(np.float64)
    val_logs, val_group = np.unique(groups[is_val], return_inverse=True)
    weights = probe.log_weights(len(val_logs), reps, probe.SEED)  # Step 2's draws
    yv_s, yv_b, ev = y_speed[is_val], y_brake[is_val], tg["event"][is_val]

    preds, fitted, counts = {}, {}, {}
    t1 = time.time()
    for k in KS:
        x_all = probe.window_features(per_moment, index[k])
        for v in VARIANTS:
            cols = variant_columns(v, k)
            counts[f"{v}/k{k}"] = int(len(cols))
            x = x_all[:, cols].astype(np.float64)
            xtr, xva = probe.standardise(x[is_train], x[is_val])
            gtr = groups[is_train]
            r = probe.fit_ridge(xtr, y_speed[is_train], gtr)
            lg = probe.fit_logit(xtr, y_brake[is_train], gtr)
            preds[(v, k)] = (probe.predict_ridge(r, xva), lg["model"].predict_proba(xva)[:, 1])
            fitted[f"{v}/k{k}"] = {"ridge_alpha": r["alpha"], "logit_C": lg["C"], "features": int(len(cols))}
            print(f"  {v} k={k}: {len(cols)} features ({time.time() - t1:.0f} s)", flush=True)
    timing["fit_s"] = round(time.time() - t1, 1)

    t1 = time.time()

    def boot(mask, sp, pb):
        """Point value and per-draw values of each metric on `mask` (probe.evaluate's `boot`)."""
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
    metrics, draws = {}, {}
    for (v, k), (sp, pb) in preds.items():
        for sl, mask in slices.items():
            for metric, (point, per_draw) in boot(mask, sp, pb).items():
                metrics.setdefault(sl, {}).setdefault(metric, {})[f"{v}/k{k}"] = {
                    "value": point,
                    "ci95": probe.ci(per_draw),
                }
                draws[(sl, metric, v, k)] = (point, per_draw)
    timing["bootstrap_s"] = round(time.time() - t1, 1)

    # V1 and V4 must be Step 2's numbers exactly (same code, data, seeds)
    reproduced = {}
    for v, name in STEP2_NAME.items():
        for k in KS:
            for sl in slices:
                for metric in ("speed_2s", "brake_logloss", "brake_ap"):
                    ours = metrics[sl][metric][f"{v}/k{k}"]
                    theirs = step2["metrics"][sl][metric][f"ridge/{name}/k{k}"]
                    reproduced[f"{v}/k{k}/{sl}/{metric}"] = bool(
                        np.isclose(ours["value"], theirs["value"], rtol=0, atol=1e-12)
                        and np.allclose(ours["ci95"], theirs["ci95"], rtol=0, atol=1e-12)
                    )
    if not all(reproduced.values()):
        bad = [k for k, ok in reproduced.items() if not ok]
        (out / "control_c_failed.json").write_text(
            json.dumps({"not_reproduced": bad, "metrics": metrics}, indent=1)
        )
        raise SystemExit(
            f"V1/V4 do not reproduce Step 2's probe.json ({len(bad)} values): "
            "stopping, see control_c_failed.json"
        )

    def diff(sl, metric, a, ka, b, kb):
        pa, da = draws[(sl, metric, a, ka)]
        pb_, db = draws[(sl, metric, b, kb)]
        lo, hi = probe.ci(da - db)
        return {"delta": pa - pb_, "ci95": [lo, hi], "ci_excludes_0": bool(lo > 0 or hi < 0)}

    comparisons = {}
    for sl in slices:
        for metric in ("speed_2s", "brake_logloss", "brake_ap", "speed_1s", "speed_3s"):
            c = comparisons.setdefault(sl, {}).setdefault(metric, {})
            for k in (4, 8):
                c[f"camera_past_on_top_of_car_history/k{k} (V1-V2)"] = diff(sl, metric, "V1", k, "V2", k)
                c[f"camera_past_alone_vs_bar/k{k} (V3-V3@k2)"] = diff(sl, metric, "V3", k, "V3", 2)
                c[f"car_past_alone_vs_bar/k{k} (V2-V2@k2)"] = diff(sl, metric, "V2", k, "V2", 2)
                c[f"step2_bar_check/k{k} (V1-V1@k2)"] = diff(sl, metric, "V1", k, "V1", 2)
            c["camera_history_vs_camera_now_at_k2 (V1-V2 @k2)"] = diff(sl, metric, "V1", 2, "V2", 2)

    verdict = []
    for metric, unit in (("speed_2s", "m/s RMSE"), ("brake_logloss", "log-loss")):
        for k in (4, 8):
            d = comparisons["events"][metric][f"camera_past_on_top_of_car_history/k{k} (V1-V2)"]
            name = "speed at +2 s" if metric == "speed_2s" else "braking"
            verdict.append(
                f"Camera past adds {d['delta']:+.4f} {unit} "
                f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] on top of car "
                f"history at k={k} ({name}, events slice; V1 - V2, negative = camera past helps): CI "
                f"{'excludes' if d['ci_excludes_0'] else 'includes'} 0"
            )
    result = {
        "rows": {
            "total": int(len(rows)),
            "train": int(is_train.sum()),
            "val": int(is_val.sum()),
            "val_events": int(ev.sum()),
        },
        "variants": DESCRIPTION,
        "feature_counts": counts,
        "fitted": fitted,
        "reproduces_step2": {"all": all(reproduced.values()), "checked": len(reproduced)},
        "metrics": metrics,
        "comparisons": comparisons,
        "verdict": verdict,
        "timing": timing,
        "reps": reps,
        "git_sha": probe._git_sha(),
        "step2_git_sha": step2.get("git_sha"),
    }
    (out / "control_c.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("run",))
    ap.add_argument("--probe", type=Path, help="Step 2's outputs; default {data.navtrain.cache}/probe")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    from gallops.config import SiteConfig

    cfg = SiteConfig.load()
    probe_dir = (args.probe or Path(cfg.resolve("{data.navtrain.cache}")) / "probe").resolve()
    out = probe_dir / "c"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = run(cfg, probe_dir, out, args.reps)
    result["timing"]["total_s"] = round(time.time() - started, 1)
    (out / "control_c.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"feature_counts": result["feature_counts"], "verdict": result["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
