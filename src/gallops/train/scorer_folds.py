"""Leave-one-scene-out training of Drive-JEPA's scorer on the cache: does it generalise at all?

s1-s3 retrained the scorer on four rendered scenes. Its held-out ranking loss ended
at the value a random ranking gets (3.45-3.58 against log 32 = 3.47), so it
memorised the scenes it saw. Those runs also saw only ~2,000 steps at batch 8,
because each step reran the encoder. This trains on cached features
(`scorer_cache`, `scorer_labels`), so 40 epochs take minutes, and holds out each
scene in turn to answer the one question: can these five scenes teach a scorer
that works on a scene it never saw?

Two scorers, same MLP (the vendor's 256 -> 1024 -> 1024 -> 6, started from the
released weights), trained on native and vocabulary plans alike:

* **control** -- the target s1-s3 used (`losses.score_target`, blended, peaked at
  the drawn expert's distance) plus the listwise ranking loss, on the last logit.
* **subscores** -- the six logits each learn one answer, with BCE:
  [no collision (5 s, any vehicle, ego stopped or not), on road, no TTC violation,
  within 1.75 m of the lane centre, progress = min(1, travel / human travel),
  final = NC * DAC * (5 EP + 5 TTC + 2 LK) / 12]. The pick is a fixed formula of
  the sigmoids, sigma(NC)^4 * sigma(DAC) * (5 EP + 5 TTC + 2 LK) / 12, whose
  constants were set before any fold ran and are not tuned.

The epoch count is fixed in advance too; held-out numbers are reported, never
used to choose anything.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gallops.train.losses import RANK_TEMPERATURE, score_target, travelled

SCENES = ("0053", "0058", "0064", "0070", "0075")
LANE_LINE_M = 1.75
CHANNELS = ("nc", "dac", "ttc", "lk", "ep", "final")


def load(cache: Path, labelled: dict[str, Path]) -> dict:
    """Every cached tick as arrays: features (T, P, 256) fp16 and per-plan labels."""
    feats, rows = (
        [],
        {
            k: []
            for k in (
                "hit",
                "hit_t",
                "off_road",
                "ttc",
                "dev",
                "travel",
                "human",
                "pdm",
                "logit",
                "usable",
                "in_frame",
                "expert_travel",
                "scene",
                "shape",
                "source",
                "episode",
            )
        },
    )
    for src_dir in sorted(p for p in cache.iterdir() if p.is_dir()):
        expert_rows = {}
        for f in sorted(src_dir.glob("*.npz")):
            if f.name.endswith(".labels.npz") or ".tmp" in f.name:
                continue
            lab_file = f.with_name(f.name.replace(".npz", ".labels.npz"))
            if not lab_file.exists():
                continue
            d, lab = np.load(f), np.load(lab_file)
            tokens = np.concatenate([d["native_tokens"], d["vocab_tokens"]], axis=1)  # (T, P, 8, 256)
            feats.append(tokens.max(axis=2))
            shape, scene = f.stem.split("__")[0].split("-")
            jl = labelled[src_dir.name] / (f.stem + ".jsonl")
            if jl not in expert_rows:
                expert_rows[jl] = {
                    json.loads(line)["timestamp"]: json.loads(line) for line in jl.read_text().splitlines()
                }
            ex = expert_rows[jl]
            t = len(d["timestamp"])
            rows["hit"].append(lab["hit5_at"])
            rows["hit_t"].append(lab["hit_target_at"])
            rows["off_road"].append(lab["off_road_at"])
            rows["ttc"].append(lab["ttc_at"])
            rows["dev"].append(lab["max_deviation"])
            rows["travel"].append(lab["travel"])
            rows["human"].append(lab["human_travel"])
            rows["pdm"].append(d["pdm_score"])
            rows["logit"].append(d["pred_logit"])
            rows["usable"].append(d["usable"])
            rows["in_frame"].append(d["in_frame"])
            rows["expert_travel"].append(
                np.array([travelled(np.asarray(ex[int(ts)]["expert"])) for ts in d["timestamp"]])
            )
            rows["scene"].append(np.full(t, scene))
            rows["shape"].append(np.full(t, shape))
            rows["source"].append(np.full(t, src_dir.name))
            rows["episode"].append(np.full(t, f"{src_dir.name}/{f.stem}"))
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["features"] = np.concatenate(feats)
    return out


def subscore_targets(d: dict) -> np.ndarray:
    nc = np.isnan(d["hit"]).astype(np.float32)
    dac = np.isnan(d["off_road"]).astype(np.float32)
    ttc = np.isnan(d["ttc"]).astype(np.float32)
    lk = (d["dev"] <= LANE_LINE_M).astype(np.float32)
    ep = np.clip(d["travel"] / np.maximum(d["human"][:, None], 1e-3), 0, 1).astype(np.float32)
    final = nc * dac * (5 * ep + 5 * ttc + 2 * lk) / 12
    return np.stack([nc, dac, ttc, lk, ep, final], axis=-1)


def control_targets(d: dict) -> np.ndarray:
    t, p = d["hit_t"].shape
    out = np.zeros((t, p), dtype=np.float32)
    for i in range(t):
        for j in range(p):
            hit = d["hit_t"][i, j]
            road = d["off_road"][i, j]
            out[i, j] = score_target(
                None if np.isnan(hit) else float(hit),
                None if np.isnan(road) else float(road),
                float(d["travel"][i, j]),
                float(d["expert_travel"][i]),
                float(d["dev"][i, j]),
            )
    return out


def formula(logits, torch):
    s = torch.sigmoid(logits)
    return s[..., 0] ** 4 * s[..., 1] * (5 * s[..., 4] + 5 * s[..., 2] + 2 * s[..., 3]) / 12


def make_mlp(state: dict, torch):
    nn = torch.nn
    mlp = nn.Sequential(
        nn.Linear(256, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(),
        nn.Linear(1024, 6),
    )
    mlp.load_state_dict(state)
    return mlp


def released_state(checkpoint: str, torch) -> dict:
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "agent._pad_model.scorer.pred_score.mlp."
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


def evaluate(pick_scores: np.ndarray, d: dict, idx: np.ndarray) -> dict:
    """Outcome of driving each tick's argmax, on ticks `idx`."""
    pick = pick_scores.argmax(1)
    rows = np.arange(len(idx))
    safe = np.isnan(d["hit"][idx]) & np.isnan(d["off_road"][idx])  # (n, P)
    rel = d["pdm"][idx].argmax(1)
    rel_safe = safe[rows, rel]
    rel_hits = ~np.isnan(d["hit"][idx][rows, rel])
    rescue = rel_hits & d["in_frame"][idx]
    calm = rel_safe
    native_safe = safe[:, :32].any(1)
    picked_vocab = pick >= 32
    travel_ratio = d["travel"][idx][rows, pick] / np.maximum(d["human"][idx], 1e-3)

    def share(mask, value):
        return float(value[mask].mean()) if mask.any() else float("nan")

    return {
        "ticks": int(len(idx)),
        "rescue_ticks": int(rescue.sum()),
        "rescued": share(rescue, safe[rows, pick]),
        "rescue_possible": share(rescue, safe.any(1)),
        "calm_ticks": int(calm.sum()),
        "calm_kept_safe": share(calm, safe[rows, pick]),
        "false_swerve": share(calm & native_safe, picked_vocab),
        "picked_vocab": float(picked_vocab.mean()),
        "travel_ratio_median": float(np.median(travel_ratio)),
    }


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label > 0.5], score[label <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train(kind: str, state: dict, x, y, train_idx, args, torch):
    mlp = make_mlp(state, torch).to(args.device)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    rng = np.random.default_rng(0)
    for _ in range(args.epochs):
        order = rng.permutation(train_idx)
        for s in range(0, len(order), args.batch):
            b = order[s : s + args.batch]
            feats = x[b].float()
            target = y[b]
            logits = mlp(feats)
            if kind == "control":
                loss = bce(logits[..., -1], target)
                want = torch.softmax(target / RANK_TEMPERATURE, dim=-1)
                loss = loss + (-(want * torch.log_softmax(logits[..., -1], dim=-1)).sum(-1).mean())
            else:
                loss = bce(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    mlp.eval()
    return mlp


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--labelled", nargs=2, action="append", required=True, metavar=("NAME", "DIR"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--smoke-split", action="store_true", help="hold out the last episode instead of a scene"
    )
    args = parser.parse_args()

    started = time.time()
    d = load(args.cache, {name: Path(p) for name, p in args.labelled})
    print(
        f"  loaded {len(d['scene'])} ticks x {d['features'].shape[1]} plans in {time.time() - started:.0f} s",
        flush=True,
    )
    state = released_state(args.checkpoint, torch)
    released = make_mlp(state, torch).to(args.device).eval()
    x = torch.from_numpy(d["features"]).to(args.device)  # fp16
    with torch.no_grad():
        rel_logits = (
            torch.cat([released(x[s : s + 256].float()) for s in range(0, len(x), 256)]).cpu().numpy()
        )
    drift = np.abs(rel_logits[:, :32] - d["logit"]).max()
    agree = (rel_logits[:, :32, -1].argmax(1) == d["logit"][..., -1].argmax(1)).mean()
    print(
        f"  released MLP on cached fp16 features: max logit drift {drift:.4f}, pick agreement {agree:.2%}",
        flush=True,
    )
    if agree < 0.97:
        raise SystemExit("cached features do not reproduce the released scorer's picks; stop")

    sub = subscore_targets(d)
    ctl = control_targets(d)
    y_sub = torch.from_numpy(sub).to(args.device)
    y_ctl = torch.from_numpy(ctl).to(args.device)
    usable = d["usable"]
    folds = [("smoke", None)] if args.smoke_split else [(s, s) for s in SCENES if (d["scene"] == s).any()]
    results = {"released_drift": float(drift), "released_pick_agreement": float(agree), "folds": {}}
    for name, scene in folds:
        if scene is None:
            last = d["episode"] == sorted(set(d["episode"]))[-1]
            test_mask, train_mask = last, ~last
        else:
            test_mask, train_mask = d["scene"] == scene, d["scene"] != scene
        train_idx = np.where(train_mask & usable)[0]
        test_idx = np.where(test_mask)[0]
        fold = {"train_ticks": int(len(train_idx)), "test_ticks": int(len(test_idx))}
        natives_only = np.where(np.arange(rel_logits.shape[1]) < 32, rel_logits[..., -1], -np.inf)
        picks = {
            "released_natives": natives_only[test_idx],
            "released_with_vocab": rel_logits[test_idx, :, -1],
        }
        for kind, y in (("control", y_ctl), ("subscores", y_sub)):
            t0 = time.time()
            mlp = train(kind, state, x, y, train_idx, args, torch)
            with torch.no_grad():
                logits = torch.cat([mlp(x[s : s + 256].float()) for s in range(0, len(x), 256)])[test_idx]
                final = logits[..., -1].cpu().numpy()
                picked = formula(logits, torch).cpu().numpy()
            if kind == "control":
                picks["control"] = final
            else:
                picks["subscores_formula"] = picked
                picks["subscores_final"] = final
                lg = logits.cpu().numpy()
                fold["auc"] = {
                    c: auc(lg[..., k].ravel(), sub[test_idx][..., k].ravel())
                    for k, c in enumerate(CHANNELS[:4])
                }
                fold["auc"]["released_final_vs_nc"] = auc(
                    rel_logits[test_idx][..., -1].ravel(), sub[test_idx][..., 0].ravel()
                )
            print(f"  fold {name}: {kind} trained in {time.time() - t0:.0f} s", flush=True)
        for method, scores in picks.items():
            fold[method] = evaluate(scores, d, test_idx)
            for shape in ("frontal", "stationary", "side"):
                m = d["shape"][test_idx] == shape
                if m.any():
                    fold.setdefault("by_shape", {}).setdefault(shape, {})[method] = evaluate(
                        scores[m], d, test_idx[m]
                    )
        results["folds"][name] = fold
        print(json.dumps({name: {k: v for k, v in fold.items() if k != "by_shape"}}, indent=1), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
