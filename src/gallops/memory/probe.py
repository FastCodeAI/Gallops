"""The go / no-go check before any memory is built: does the past predict the human better?

    python -m gallops.memory.probe run [--out DIR] [--reps 2000]     # CPU only, every stage below

**The question.** The released model sees one moment: the front image now and 0.5 s ago, encoded
together. A memory would let the picker see further back. That is only worth building if the
last 2-4 s predict what the human driver does next better than the last 1 s does. So: predict
the human's speed at +1, +2 and +3 s, and whether they start braking within 2 s, from the last
k moments, k in {1, 2, 4, 8}, and compare every k with k = 2 (1 s, the released model's view).

**Per-moment features** (`pool`, `pca`). The cached encoder map is 16 x 32 tokens of 256
(the 512 x 256 front image in 16-pixel patches, row-major). Pooled to its global mean (256) and a
4 x 8 average pool of 4 x 4-token blocks (32 x 256), then PCA to 128, fitted on train-split
moments only and applied to all; plus the 11 ego values the model was given.

**Windows** (`window_index`). The features of window k for a row are those of the row's moment
and the k - 1 moments before it in the same chain, newest first. Nothing after "now" can enter:
every index is checked against the row's chain and timestamp (`check_no_future`).

**Targets** (`targets_for_log`), from the log pickles, never from images or the cache, defined as
Step 1's missing-log check (`ticks.dynamics`): forward speed is the log's own
`ego_dynamic_state[0]` at frames +2, +4, +6 (2 Hz, so +1, +2, +3 s), unsmoothed; a braking
onset is ax >= -1.5 m/s^2 now and ax < -1.5 m/s^2 at one of the next 4 frames (2 s); an event is
a forward-speed change of more than 1 m/s within those 2 s.

**Rows**: graded moments with 8 moments in their window (chain_pos >= 8: the chain's first
frame is not a moment) and 3 s of log ahead. Every k and every model is scored on these rows.

**Models**: ridge (speeds) and L2-penalised logistic regression (braking), penalty chosen by
5-fold cross-validation grouped by log on the train split; a one-hidden-layer MLP (256) with
early stopping on a log-grouped tenth of train. Controls: ego-only features at every k, and a
shuffled-target fit at k = 8. Evaluated on the val split's logs, all rows and the events slice;
paired differences against k = 2 with a log-clustered bootstrap.

Runs in Drive-JEPA's interpreter (numpy, scipy, scikit-learn, torch on CPU).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from gallops.compat import zip_strict

GRID_H, GRID_W, DIM = 16, 32, 256
POOL_H, POOL_W = 4, 8
PCA_DIM = 128
EGO_DIM = 11
WINDOWS = (1, 2, 4, 8)
BAR = 2
#: Frames ahead of each speed target at 2 Hz: +1, +2, +3 s.
SPEED_STEPS = (2, 4, 6)
SPEED_NAMES = ("speed_1s", "speed_2s", "speed_3s")
SEED = 0


# --- per-moment features -----------------------------------------------------------------------


def pool_map(encoder: np.ndarray) -> np.ndarray:
    """(..., 512, 256) token maps -> (..., 33 * 256): global mean, then the 4 x 8 block means."""
    lead = encoder.shape[:-2]
    grid = encoder.reshape(*lead, GRID_H, GRID_W, DIM)
    bh, bw = GRID_H // POOL_H, GRID_W // POOL_W
    blocks = grid.reshape(*lead, POOL_H, bh, POOL_W, bw, DIM).mean(axis=(-4, -2))
    return np.concatenate([grid.mean(axis=(-3, -2)), blocks.reshape(*lead, POOL_H * POOL_W * DIM)], -1)


def pca_fit(x: np.ndarray, dim: int, chunk: int = 4096) -> dict:
    """Mean, top-`dim` components and explained variance of the rows of `x` (float64 covariance)."""
    mean = x.mean(0, dtype=np.float64)
    cov = np.zeros((x.shape[1], x.shape[1]))
    for i in range(0, len(x), chunk):
        block = x[i : i + chunk].astype(np.float64) - mean
        cov += block.T @ block
    cov /= len(x) - 1
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    total = float(values.sum())
    return {
        "mean": mean.astype(np.float32),
        "components": vectors[:, :dim].T.astype(np.float32),
        "explained_ratio": (values[:dim] / total).astype(np.float64),
    }


def pca_apply(x: np.ndarray, fit: dict, chunk: int = 8192) -> np.ndarray:
    out = np.empty((len(x), len(fit["components"])), np.float32)
    for i in range(0, len(x), chunk):
        out[i : i + chunk] = (x[i : i + chunk] - fit["mean"]) @ fit["components"].T
    return out


# --- windows and the leak check -----------------------------------------------------------------


def window_index(table: dict, rows: np.ndarray, k: int) -> np.ndarray:
    """(len(rows), k) table rows: each row's moment and the k - 1 before it in its chain, newest first.

    Raises if a row has fewer than k moments in its chain (chain_pos counts the chain's frames before
    the moment, and the chain's first frame is not a moment, so k moments need chain_pos >= k).
    """
    key = {}
    for i, (log, chain, pos) in enumerate(zip_strict(table["log"], table["chain"], table["chain_pos"])):
        key[(str(log), int(chain), int(pos))] = i
    out = np.empty((len(rows), k), np.int64)
    for r, i in enumerate(rows):
        log, chain, pos = str(table["log"][i]), int(table["chain"][i]), int(table["chain_pos"][i])
        for j in range(k):
            out[r, j] = key[(log, chain, pos - j)]  # KeyError: not enough history
    check_no_future(table, rows, out)
    return out


def check_no_future(table: dict, rows: np.ndarray, index: np.ndarray) -> None:
    """Every window moment is in the row's own log and chain, and no later than the row's moment."""
    rows = np.asarray(rows)[:, None]
    same_log = table["log"][index] == table["log"][rows]
    same_chain = table["chain"][index] == table["chain"][rows]
    not_later = table["timestamp"][index] <= table["timestamp"][rows]
    not_after = table["chain_pos"][index] <= table["chain_pos"][rows]
    if not (same_log.all() and same_chain.all() and not_later.all() and not_after.all()):
        raise AssertionError("a window reaches outside its row's chain or past its row's moment")


def window_features(per_moment: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Concatenate the per-moment features of each window, newest first."""
    return per_moment[index].reshape(len(index), -1)


# --- targets ------------------------------------------------------------------------------------


def targets_for_log(frames: list, frame_idxs) -> dict:
    """Per frame index: speeds at +1/+2/+3 s, braking onset and event within 2 s (ticks.dynamics).

    `has_future` is False where the log does not reach 3 s ahead; the other values are then NaN/False.
    Also returns, for a check of the speed definition, the pose-differenced speed at +2 s.
    """
    from gallops.memory import ticks

    d = ticks.dynamics(frames)
    n = len(frames)
    xy = np.asarray([f["ego2global_translation"][:2] for f in frames], dtype=np.float64)
    t = np.asarray([f["timestamp"] for f in frames], dtype=np.float64) / 1e6
    out = {name: [] for name in (*SPEED_NAMES, "onset", "event", "has_future", "pose_speed_2s", "gap_s")}
    for i in frame_idxs:
        ok = i + max(SPEED_STEPS) < n
        out["has_future"].append(ok)
        for name, step in zip_strict(SPEED_NAMES, SPEED_STEPS):
            out[name].append(d["speed"][i + step] if ok else np.nan)
        out["onset"].append(bool(d["onset"][i]) if ok else False)
        out["event"].append(bool(d["event"][i]) if ok else False)
        if ok:
            a, b = i + 3, i + 5  # centred difference about +2 s
            out["pose_speed_2s"].append(float(np.linalg.norm(xy[b] - xy[a]) / (t[b] - t[a])))
            out["gap_s"].append(float(t[i + max(SPEED_STEPS)] - t[i]))
        else:
            out["pose_speed_2s"].append(np.nan)
            out["gap_s"].append(np.nan)
    return {k: np.asarray(v) for k, v in out.items()}


# --- metrics and the bootstrap ------------------------------------------------------------------


def log_loss_terms(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def average_precision(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Area under the precision-recall curve as a step sum (scikit-learn's average_precision_score)."""
    order = np.argsort(-score, kind="stable")
    y, score = y[order].astype(np.float64), score[order]
    w = np.ones_like(y) if weight is None else weight[order].astype(np.float64)
    tp, seen = np.cumsum(w * y), np.cumsum(w)
    # one point per distinct score: the last index of each run of ties
    last = np.r_[np.flatnonzero(np.diff(score) != 0), len(score) - 1]
    tp, seen = tp[last], seen[last]
    if tp[-1] <= 0:
        return float("nan")
    precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def per_log_sums(groups: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(groups, weights=values, minlength=n_groups)


def log_weights(n_groups: int, reps: int, seed: int) -> np.ndarray:
    """(reps, n_groups): how many times each log is drawn in each log-clustered bootstrap draw."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(reps, n_groups))
    w = np.zeros((reps, n_groups))
    for r in range(reps):
        w[r] = np.bincount(draws[r], minlength=n_groups)
    return w


def ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


# --- models -------------------------------------------------------------------------------------

RIDGE_ALPHAS = np.logspace(-2, 5, 15)
LOGIT_CS = np.logspace(-4, 0, 5)
CV_FOLDS = 5


def standardise(train: np.ndarray, *others: np.ndarray):
    mean, std = train.mean(0), train.std(0)
    std[std == 0] = 1.0
    return [(x - mean) / std for x in (train, *others)]


def log_folds(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Fold of each row, whole logs per fold, logs assigned in a seeded random order."""
    logs = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip_strict(rng.permutation(logs), np.arange(len(logs)) % folds))
    return np.asarray([fold_of[g] for g in groups])


def ridge_path(x: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """(len(alphas), d, t) ridge coefficients of centred y on centred x, one eigendecomposition."""
    values, vectors = np.linalg.eigh(x.T @ x)
    projected = vectors.T @ (x.T @ y)
    return np.stack([vectors @ (projected / (values + a)[:, None]) for a in alphas])


def fit_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Ridge with one penalty per target, chosen by log-grouped CV mean squared error."""
    fold = log_folds(groups, CV_FOLDS, SEED)
    errors = np.zeros((len(RIDGE_ALPHAS), y.shape[1]))
    for f in range(CV_FOLDS):
        tr, te = fold != f, fold == f
        xm, ym = x[tr].mean(0), y[tr].mean(0)
        coef = ridge_path(x[tr] - xm, y[tr] - ym, RIDGE_ALPHAS)
        for a in range(len(RIDGE_ALPHAS)):
            pred = (x[te] - xm) @ coef[a] + ym
            errors[a] += ((pred - y[te]) ** 2).sum(0)
    best = errors.argmin(0)
    xm, ym = x.mean(0), y.mean(0)
    coef = ridge_path(x - xm, y - ym, RIDGE_ALPHAS)
    weights = np.stack([coef[best[t], :, t] for t in range(y.shape[1])], 1)
    return {"xm": xm, "ym": ym, "w": weights, "alpha": RIDGE_ALPHAS[best].tolist()}


def predict_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    return (x - model["xm"]) @ model["w"] + model["ym"]


def fit_logit(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """L2 logistic regression, C chosen by log-grouped CV log-loss (scikit-learn, lbfgs)."""
    from sklearn.linear_model import LogisticRegression

    fold = log_folds(groups, CV_FOLDS, SEED)
    losses = np.zeros(len(LOGIT_CS))
    for f in range(CV_FOLDS):
        tr, te = fold != f, fold == f
        for c, value in enumerate(LOGIT_CS):
            m = LogisticRegression(C=value, max_iter=1000, tol=1e-6).fit(x[tr], y[tr])
            losses[c] += log_loss_terms(y[te], m.predict_proba(x[te])[:, 1]).sum()
    best = LOGIT_CS[int(losses.argmin())]
    model = LogisticRegression(C=best, max_iter=1000, tol=1e-6).fit(x, y)
    return {"model": model, "C": float(best)}


def fit_mlp(x: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str, max_epochs: int = 60) -> dict:
    """One hidden layer of 256 (ReLU), Adam, early stopping on a log-grouped tenth of the train rows."""
    import torch

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    hold = log_folds(groups, 10, SEED + 1) == 0
    ym, ys = (y[~hold].mean(0), y[~hold].std(0)) if kind == "regression" else (0.0, 1.0)
    yt = (y - ym) / ys
    net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, y.shape[1])
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss() if kind == "regression" else torch.nn.BCEWithLogitsLoss()
    xtr, ytr = torch.tensor(x[~hold], dtype=torch.float32), torch.tensor(yt[~hold], dtype=torch.float32)
    xho, yho = torch.tensor(x[hold], dtype=torch.float32), torch.tensor(yt[hold], dtype=torch.float32)
    gen = torch.Generator().manual_seed(SEED)
    best, best_state, stale, epochs = np.inf, None, 0, 0
    for epoch in range(max_epochs):
        net.train()
        for batch in torch.randperm(len(xtr), generator=gen).split(256):
            opt.zero_grad()
            loss_fn(net(xtr[batch]), ytr[batch]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            held = float(loss_fn(net(xho), yho))
        epochs = epoch + 1
        if held < best - 1e-5:
            best, stale = held, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
    net.load_state_dict(best_state)
    return {"net": net, "ym": ym, "ys": ys, "kind": kind, "epochs": epochs, "held_loss": best}


def predict_mlp(model: dict, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        out = model["net"](torch.tensor(x, dtype=torch.float32)).numpy().astype(np.float64)
    if model["kind"] == "regression":
        return out * model["ys"] + model["ym"]
    return 1.0 / (1.0 + np.exp(-out[:, 0]))


# --- the stages ---------------------------------------------------------------------------------


def stage_pool(cache_dir: Path, out: Path, n: int, chunk: int = 1000) -> Path:
    """Pooled maps of every moment (N, 8448) float32, from the cache's encoder array."""
    path = out / "pooled.npy"
    if path.exists():
        return path
    encoder = np.load(cache_dir / "encoder.npy", mmap_mode="r")
    partial = out / "pooled.partial.npy"
    pooled = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(n, (1 + POOL_H * POOL_W) * DIM)
    )
    for i in range(0, n, chunk):
        pooled[i : i + chunk] = pool_map(np.asarray(encoder[i : i + chunk]))
    pooled.flush()
    del pooled
    os.replace(partial, path)
    return path


def stage_targets(cfg, table: dict, rows: np.ndarray, out: Path) -> dict:
    path = out / "targets.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    logs_dir = cfg.datasets["navtrain"].extra["logs"]
    by_log: dict = {}
    for i in rows:
        by_log.setdefault(str(table["log"][i]), []).append(int(i))
    parts = {}
    for log, log_rows in by_log.items():
        with open(logs_dir / f"{log}.pkl", "rb") as f:
            frames = pickle.load(f)
        parts[log] = (log_rows, targets_for_log(frames, [int(table["frame_idx"][i]) for i in log_rows]))
    order = np.concatenate([np.asarray(r) for r, _ in parts.values()])
    result = {"row": order}
    for name in next(iter(parts.values()))[1]:
        result[name] = np.concatenate([t[name] for _, t in parts.values()])
    np.savez(path, **result)
    return result


def evaluate(cfg, out: Path, reps: int) -> dict:
    from gallops.memory import ticks

    timing: dict = {}
    t0 = time.time()
    cache_root = Path(cfg.resolve("{data.navtrain.cache}"))
    table = ticks.load(cache_root / "ticks")
    n = len(table["log"])
    split = table["split"]

    # A4 candidate rows: graded, 8 moments in the window; the 3 s of future comes from the targets
    candidates = np.flatnonzero(table["graded"] & table["inputs_ok"] & (table["chain_pos"] >= max(WINDOWS)))
    targets = stage_targets(cfg, table, candidates, out)
    timing["targets_s"] = round(time.time() - t0, 1)
    order = np.argsort(targets["row"])
    targets = {k: v[order] for k, v in targets.items()}
    assert np.array_equal(targets["row"], candidates)
    keep = targets["has_future"]
    rows = candidates[keep]
    tg = {k: v[keep] for k, v in targets.items()}

    # A1 per-moment features
    t1 = time.time()
    pooled = np.load(stage_pool(cache_root / "release", out, n), mmap_mode="r")
    timing["pool_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    pca_path = out / "pca.npz"
    train_moments = np.flatnonzero((split == "train") & table["inputs_ok"])
    if pca_path.exists():
        with np.load(pca_path) as z:
            fit = {k: z[k] for k in z.files}
    else:
        fit = pca_fit(np.asarray(pooled[train_moments]), PCA_DIM)
        np.savez(pca_path, **fit)
    ego = np.asarray(np.load(cache_root / "release" / "ego.npy", mmap_mode="r"))
    per_moment = np.concatenate([pca_apply(pooled, fit), ego.astype(np.float32)], 1)
    timing["pca_s"] = round(time.time() - t1, 1)

    # A2 windows, with the A5 check inside window_index
    t1 = time.time()
    index = {k: window_index(table, rows, k) for k in WINDOWS}
    timing["windows_s"] = round(time.time() - t1, 1)

    is_train = split[rows] == "train"
    is_val = split[rows] == "val"
    groups = table["log"][rows]
    events = tg["event"]
    y_speed = np.stack([tg[name] for name in SPEED_NAMES], 1)
    y_brake = tg["onset"].astype(np.float64)
    val_logs, val_group = np.unique(groups[is_val], return_inverse=True)
    weights = log_weights(len(val_logs), reps, SEED)

    preds: dict = {}
    fitted: dict = {}
    t1 = time.time()
    for k in WINDOWS:
        x_all = window_features(per_moment, index[k])
        ego_cols = np.concatenate(
            [np.arange(PCA_DIM, PCA_DIM + EGO_DIM) + j * (PCA_DIM + EGO_DIM) for j in range(k)]
        )
        for feats, cols in (("images", slice(None)), ("ego_only", ego_cols)):
            x = x_all[:, cols].astype(np.float64)
            xtr, xva = standardise(x[is_train], x[is_val])
            gtr = groups[is_train]
            r = fit_ridge(xtr, y_speed[is_train], gtr)
            lg = fit_logit(xtr, y_brake[is_train], gtr)
            preds[("ridge", feats, k)] = (predict_ridge(r, xva), lg["model"].predict_proba(xva)[:, 1])
            ms = fit_mlp(xtr, y_speed[is_train], gtr, "regression")
            mb = fit_mlp(xtr, y_brake[is_train][:, None], gtr, "classification")
            preds[("mlp", feats, k)] = (predict_mlp(ms, xva), predict_mlp(mb, xva))
            fitted[f"{feats}_k{k}"] = {
                "ridge_alpha": r["alpha"],
                "logit_C": lg["C"],
                "mlp_speed_epochs": ms["epochs"],
                "mlp_brake_epochs": mb["epochs"],
                "dims": int(x.shape[1]),
            }
            print(f"  fitted {feats} k={k} ({time.time() - t1:.0f} s)", flush=True)
    timing["fit_s"] = round(time.time() - t1, 1)

    # B3 shuffled targets at k = 8 (images): permute the train targets across train rows
    t1 = time.time()
    rng = np.random.default_rng(SEED + 2)
    perm = rng.permutation(int(is_train.sum()))
    x = window_features(per_moment, index[8]).astype(np.float64)
    xtr, xva = standardise(x[is_train], x[is_val])
    gtr = groups[is_train]
    r = fit_ridge(xtr, y_speed[is_train][perm], gtr)
    lg = fit_logit(xtr, y_brake[is_train][perm], gtr)
    preds[("ridge", "shuffled", 8)] = (predict_ridge(r, xva), lg["model"].predict_proba(xva)[:, 1])
    timing["shuffled_s"] = round(time.time() - t1, 1)

    # B4 / C1 / C2 metrics with the log-clustered bootstrap
    t1 = time.time()
    yv_s, yv_b, ev = y_speed[is_val], y_brake[is_val], events[is_val]
    const_speed = np.broadcast_to(y_speed[is_train].mean(0), yv_s.shape)
    const_brake = np.full(len(yv_b), y_brake[is_train].mean())
    preds[("constant", "train_mean", 0)] = (const_speed, const_brake)

    def boot(mask, sp, pb):
        """Point value and per-draw values of every metric on the rows in `mask`."""
        g = val_group[mask]
        cnt = per_log_sums(g, np.ones(mask.sum()), len(val_logs))
        res = {}
        for t, name in enumerate(SPEED_NAMES):
            sse = per_log_sums(g, (sp[mask, t] - yv_s[mask, t]) ** 2, len(val_logs))
            res[name] = (float(np.sqrt(sse.sum() / cnt.sum())), np.sqrt((weights @ sse) / (weights @ cnt)))
        ll = per_log_sums(g, log_loss_terms(yv_b[mask], pb[mask]), len(val_logs))
        res["brake_logloss"] = (float(ll.sum() / cnt.sum()), (weights @ ll) / (weights @ cnt))
        row_w = weights[:, g]
        ap = np.array([average_precision(yv_b[mask], pb[mask], row_w[r]) for r in range(reps)])
        res["brake_ap"] = (average_precision(yv_b[mask], pb[mask]), ap)
        return res

    slices = {"all": np.ones(len(yv_b), bool), "events": ev}
    metrics: dict = {}
    draws: dict = {}
    for (model, feats, k), (sp, pb) in preds.items():
        for sl, mask in slices.items():
            res = boot(mask, sp, pb)
            for metric, (point, per_draw) in res.items():
                metrics.setdefault(sl, {}).setdefault(metric, {})[f"{model}/{feats}/k{k}"] = {
                    "value": point,
                    "ci95": ci(per_draw),
                }
                draws[(sl, metric, model, feats, k)] = per_draw
    deltas: dict = {}
    for (sl, metric, model, feats, k), per_draw in draws.items():
        if model in ("ridge", "mlp") and feats in ("images", "ego_only") and k != BAR:
            bar = draws[(sl, metric, model, feats, BAR)]
            point = (
                metrics[sl][metric][f"{model}/{feats}/k{k}"]["value"]
                - metrics[sl][metric][f"{model}/{feats}/k{BAR}"]["value"]
            )
            deltas.setdefault(sl, {}).setdefault(metric, {})[f"{model}/{feats}/k{k}"] = {
                "delta": point,
                "ci95": ci(per_draw - bar),
            }
    timing["bootstrap_s"] = round(time.time() - t1, 1)

    verdict = go_rule(deltas)
    pose = tg["pose_speed_2s"]
    result = {
        "rows": {
            "candidates_graded_chain_pos_ge_8": int(len(candidates)),
            "with_3s_future": int(len(rows)),
            "train": int(is_train.sum()),
            "val": int(is_val.sum()),
            "train_logs": int(len(np.unique(groups[is_train]))),
            "val_logs": int(len(val_logs)),
            "val_events": int(ev.sum()),
            "val_onsets": int(yv_b.sum()),
            "val_onsets_in_events": int(yv_b[ev].sum()),
            "train_onset_rate": float(y_brake[is_train].mean()),
            "graded_with_8_frames_chain_pos_ge_7": int(
                (table["graded"] & table["inputs_ok"] & (table["chain_pos"] >= 7)).sum()
            ),
        },
        "pca": {
            "fit_moments": int(len(train_moments)),
            "input_dim": int(pooled.shape[1]),
            "explained_ratio_total": float(fit["explained_ratio"].sum()),
            "explained_ratio_first10": [float(v) for v in fit["explained_ratio"][:10]],
        },
        "speed_definition_check": {
            "median_abs_diff_log_vx_vs_pose_speed_2s": float(np.nanmedian(np.abs(pose - tg["speed_2s"]))),
            "corr": float(np.corrcoef(pose, tg["speed_2s"])[0, 1]),
            "future_span_s_min_max": [float(np.nanmin(tg["gap_s"])), float(np.nanmax(tg["gap_s"]))],
        },
        "fitted": fitted,
        "metrics": metrics,
        "deltas_vs_k2": deltas,
        "verdict": verdict,
        "timing": timing,
        "reps": reps,
        "git_sha": _git_sha(),
    }
    (out / "probe.json").write_text(json.dumps(result, indent=1) + "\n")
    figure(result, out / "probe_error_vs_k.png", "speed_2s", "RMSE of speed at +2 s (m/s)")
    figure(
        result, out / "probe_error_vs_k_braking.png", "brake_logloss", "log-loss, braking onset within 2 s"
    )
    return result


def go_rule(deltas: dict) -> dict:
    """The GO rule, verbatim: with image features, k=4 or k=8 beats k=2 on the EVENTS slice, CI excluding
    0, on braking or on speed at +2 s. Braking is judged on log-loss (lower is better); its
    precision-recall area is reported beside it. Judged per model; the overall verdict is GO if any
    target is GO for any model, and says so when only one model gets there.
    """
    out: dict = {"per_target": {}, "detail": {}}
    targets = {"speed_2s": "speed_2s", "braking": "brake_logloss"}
    any_go = {"ridge": False, "mlp": False}
    for target, metric in targets.items():
        target_go = {}
        for model in ("ridge", "mlp"):
            hits = []
            for k in (4, 8):
                d = deltas["events"][metric][f"{model}/images/k{k}"]
                beats = d["ci95"][1] < 0
                out["detail"][f"{target}/{model}/k{k}"] = {**d, "beats_k2": beats}
                hits.append(beats)
            target_go[model] = any(hits)
            any_go[model] |= any(hits)
        out["per_target"][target] = "GO" if any(target_go.values()) else "NO-GO"
        out["per_target"][f"{target}_by_model"] = {m: ("GO" if v else "NO-GO") for m, v in target_go.items()}
        ap = {
            f"{m}/k{k}": deltas["events"]["brake_ap"][f"{m}/images/k{k}"]
            for m in ("ridge", "mlp")
            for k in (4, 8)
        }
        if target == "braking":
            out["braking_ap_deltas_events"] = ap
    overall = any(any_go.values())
    out["overall"] = "GO" if overall else "NO-GO"
    if overall and not all(any_go.values()):
        out["overall_note"] = "only " + ", ".join(m for m, v in any_go.items() if v) + " meets the rule"
    return out


def figure(result: dict, path: Path, metric: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    colours = {"ridge": "#2a6fb0", "mlp": "#d0742c"}
    for ax, sl in zip_strict(axes, ("all", "events")):
        m = result["metrics"][sl][metric]
        for model in ("ridge", "mlp"):
            for feats, style in (("images", "-"), ("ego_only", ":")):
                vals = [m[f"{model}/{feats}/k{k}"]["value"] for k in WINDOWS]
                lo = [m[f"{model}/{feats}/k{k}"]["ci95"][0] for k in WINDOWS]
                hi = [m[f"{model}/{feats}/k{k}"]["ci95"][1] for k in WINDOWS]
                label = f"{model}, {'images + ego' if feats == 'images' else 'ego only'}"
                ax.plot(WINDOWS, vals, style, marker="o", color=colours[model], label=label)
                if feats == "images":
                    ax.fill_between(WINDOWS, lo, hi, color=colours[model], alpha=0.15)
        ax.axvline(BAR, color="grey", lw=0.8, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(WINDOWS)
        ax.set_xticklabels([f"{k}\n({k * 0.5:g} s)" for k in WINDOWS])
        ax.set_xlabel("moments in the window (time covered)")
        ax.set_title(f"{'all rows' if sl == 'all' else 'events slice'} (val logs)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Error vs how far back the features reach; bands: 95% log-clustered bootstrap; dashed: k=2")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _git_sha() -> str | None:
    try:
        from gallops.drivejepa_b2d.provenance import git_sha

        return git_sha()
    except Exception:  # noqa: BLE001 (provenance is best effort here)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("run",))
    ap.add_argument("--out", type=Path, help="default: {data.navtrain.cache}/probe")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    from gallops.config import SiteConfig

    cfg = SiteConfig.load()
    out = (args.out or Path(cfg.resolve("{data.navtrain.cache}")) / "probe").resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = evaluate(cfg, out, args.reps)
    result["timing"]["total_s"] = round(time.time() - started, 1)
    (out / "probe.json").write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {"rows": result["rows"], "verdict": result["verdict"], "timing": result["timing"]}, indent=1
        )
    )


if __name__ == "__main__":
    main()
