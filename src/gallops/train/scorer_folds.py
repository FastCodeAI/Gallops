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
from collections import Counter
from pathlib import Path

import numpy as np

from gallops.train.losses import RANK_TEMPERATURE, score_target, travelled

SCENES = ("0053", "0058", "0064", "0070", "0075")
LANE_LINE_M = 1.75
CHANNELS = ("nc", "dac", "ttc", "lk", "ep", "final")
#: Gated pick: override the released pick only when sigma(NC) of it is below this. Fixed, not tuned.
VETO = 0.5


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
            feats = x[b].to(args.device).float()
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


def probe_geo_metrics(s: np.ndarray, d: dict, idx: np.ndarray) -> dict:
    """Held-out outcomes of the sub-score scorer's picks, from its seed-mean sigmoids `s`.

    rescued: on ticks where the released pick hits with the hazard in view, the new pick is safe
    (no hit within 5 s, on the road). kept: on ticks where the released pick was safe, it stays
    safe. frontal_gt2: frontal rescue ticks more than 2 s before the released pick's contact.
    """
    from gallops.train import subscorer

    rows = np.arange(len(idx))
    safe = np.isnan(d["hit5_at"][idx]) & np.isnan(d["off_road_at"][idx])
    rel = d["pdm"][idx].argmax(1)
    t_hit = d["hit5_at"][idx][rows, rel]
    rescue = ~np.isnan(t_hit) & d["in_frame"][idx]
    calm = safe[rows, rel]
    frontal2 = rescue & (d["shape"][idx] == "frontal") & (t_hit > 2)

    def share(mask, value):
        return round(float(value[mask].mean()), 4) if mask.any() else None

    out = {
        "rescue_ticks": int(rescue.sum()),
        "calm_ticks": int(calm.sum()),
        "rescue_possible": share(rescue, safe.any(1)),
    }
    for rule in ("formula", "absolute", "relative"):
        pick = subscorer.formula(s).argmax(1) if rule == "formula" else subscorer.choose(s, rel, rule)
        ok = safe[rows, pick]
        out[rule] = {
            "rescued": share(rescue, ok),
            "kept": share(calm, ok),
            "net": int(ok[rescue].sum() - (~ok[calm]).sum()),
            "frontal_gt2_rescued": share(frontal2, ok),
            "override": share(np.ones(len(idx), bool), pick != rel),
        }
    out["frontal_gt2_possible"] = share(frontal2, safe.any(1))
    return out


def run_probe_geo(args) -> None:
    """Leave-one-scene-out for the sub-score scorer over probes and geometry (`subscorer`)."""
    import torch

    from gallops.train import subscorer

    started = time.time()
    parts = [subscorer.reduce(c) for c in [args.cache, *args.extra_cache]]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    source = subscorer.source_of(d)
    print(
        f"  loaded {len(source)} ticks in {time.time() - started:.0f} s: "
        f"{dict(Counter(source.tolist()))}",
        flush=True,
    )
    sight, problems = subscorer.sight_from_cache(d)
    print(f"  sight: {json.dumps(sight)} {problems}", flush=True)
    arrays = (
        torch.from_numpy(d["own"]),
        torch.from_numpy(d["probe"]),
        torch.from_numpy(subscorer.geometry(d["plans"], d["speed"])),
        torch.from_numpy(subscorer.targets(d)),
        torch.from_numpy(subscorer.channel_weights(source, args.real)),
    )
    y = subscorer.targets(d)
    state = released_state(args.checkpoint, torch)
    holdout = np.where(source == "nusc_holdout")[0]
    rendered = ~np.isin(source, subscorer.REAL_SOURCES)
    results = {"sight": sight, "real": args.real, "seeds": args.seeds, "folds": {}}
    saved = {}
    hold_sum = None
    for scene in SCENES:
        test = np.where(rendered & (d["scene"] == scene))[0]
        if not len(test):
            continue
        train_idx = np.where((d["scene"] != scene) & d["usable"] & (source != "nusc_holdout"))[0]
        mean_s, t0 = None, time.time()
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = subscorer.build(torch, state).to(args.device)
            subscorer.fit(
                model,
                arrays,
                train_idx,
                epochs=args.epochs,
                batch=args.batch,
                lr=args.lr,
                seed=seed,
                device=args.device,
                torch=torch,
            )
            s = 1 / (1 + np.exp(-subscorer.predict(model, arrays, test, device=args.device, torch=torch)))
            mean_s = s if mean_s is None else mean_s + s
            if len(holdout):
                h = 1 / (
                    1 + np.exp(-subscorer.predict(model, arrays, holdout, device=args.device, torch=torch))
                )
                hold_sum = h if hold_sum is None else hold_sum + h
        mean_s /= len(args.seeds)
        fold = {
            "train_ticks": int(len(train_idx)),
            "test_ticks": int(len(test)),
            "seconds": round(time.time() - t0),
        }
        fold.update(probe_geo_metrics(mean_s, d, test))
        fold["auc"] = {
            c: auc(mean_s[..., k].ravel(), y[test][..., k].ravel())
            for k, c in enumerate(subscorer.CHANNELS[:4])
        }
        fold["by_shape"] = {
            shape: probe_geo_metrics(mean_s[d["shape"][test] == shape], d, test[d["shape"][test] == shape])
            for shape in ("frontal", "stationary", "side")
            if (d["shape"][test] == shape).any()
        }
        results["folds"][scene] = fold
        saved[f"sigmoid_{scene}"] = mean_s.astype(np.float32)
        saved[f"test_idx_{scene}"] = test
        print(json.dumps({scene: {k: v for k, v in fold.items() if k != "by_shape"}}), flush=True)
    if hold_sum is not None:
        h = hold_sum / (len(args.seeds) * len(results["folds"]))
        rows = np.arange(len(holdout))
        rel = d["pdm"][holdout].argmax(1)
        safe = np.isnan(d["off_road_at"][holdout]) & (d["max_deviation"][holdout] <= subscorer.LANE_LINE_M)
        results["real_holdout"] = {
            rule: {
                "override": round(float((subscorer.choose(h, rel, rule) != rel).mean()), 4),
                "kept_on_road_in_lane": round(
                    float(safe[rows, subscorer.choose(h, rel, rule)][safe[rows, rel]].mean()), 4
                ),
            }
            for rule in ("absolute", "relative")
        }
        saved["sigmoid_nusc_holdout"] = h.astype(np.float32)
        saved["test_idx_nusc_holdout"] = holdout
        print(json.dumps({"real_holdout": results["real_holdout"]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    np.savez(args.out.with_suffix(".sigmoids.npz"), **saved)
    print(f"  wrote {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--labelled", nargs=2, action="append", default=[], metavar=("NAME", "DIR"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--smoke-split", action="store_true", help="hold out the last episode instead of a scene"
    )
    parser.add_argument(
        "--variant",
        choices=("mlp", "probe-geo"),
        default="mlp",
        help="mlp: the vendor MLP on own tokens (control vs subscores); probe-geo: the sub-score "
        "scorer with probe attention and plan geometry (`subscorer`), the one scorer_train writes",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="probe-geo only")
    parser.add_argument(
        "--extra-cache", type=Path, action="append", default=[], help="probe-geo only, e.g. real nuScenes"
    )
    parser.add_argument("--real", choices=("all", "mask", "weight"), default="mask", help="probe-geo only")
    args = parser.parse_args()
    if args.variant == "probe-geo":
        run_probe_geo(args)
        return
    if not args.labelled:
        parser.error("--labelled is required for --variant mlp")

    import torch

    started = time.time()
    # Reading 450 cache files from the NAS takes ~7 min; keep the reduced arrays beside them.
    reduced = args.cache / "folds_input.npz"
    if reduced.exists():
        d = dict(np.load(reduced))
    else:
        d = load(args.cache, {name: Path(p) for name, p in args.labelled})
        np.savez(reduced, **d)
    print(
        f"  loaded {len(d['scene'])} ticks x {d['features'].shape[1]} plans in {time.time() - started:.0f} s",
        flush=True,
    )
    state = released_state(args.checkpoint, torch)
    released = make_mlp(state, torch).to(args.device).eval()
    # fp16 features stay in host memory and go to the GPU a batch at a time: the GPU is
    # shared and often within a few GB of full, and the first run died holding them all.
    x = torch.from_numpy(d["features"])
    with torch.no_grad():
        rel_logits = (
            torch.cat([released(x[s : s + 256].to(args.device).float()) for s in range(0, len(x), 256)])
            .cpu()
            .numpy()
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
    saved = {}
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
                logits = torch.cat(
                    [
                        mlp(x[test_idx[s : s + 256]].to(args.device).float())
                        for s in range(0, len(test_idx), 256)
                    ]
                )
                final = logits[..., -1].cpu().numpy()
                picked = formula(logits, torch).cpu().numpy()
            if kind == "control":
                picks["control"] = final
            else:
                picks["subscores_formula"] = picked
                picks["subscores_final"] = final
                # Gated: keep the released pick unless the collision head says it hits
                # (sigma(NC) < VETO, fixed before any fold ran), then take the formula's pick.
                rel_pick = picks["released_natives"].argmax(1)
                nc_rel = (
                    torch.sigmoid(logits[torch.arange(len(rel_pick)), torch.as_tensor(rel_pick), 0])
                    .cpu()
                    .numpy()
                )
                veto = nc_rel < VETO
                gated = np.full_like(picked, -np.inf)
                gated[np.arange(len(rel_pick)), rel_pick] = 0.0
                gated[veto] = picked[veto]
                picks["subscores_gated"] = gated
                fold["veto_rate"] = float(veto.mean())
                saved[name] = {"test_idx": test_idx, "logits": logits.cpu().numpy()}
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
    np.savez(
        args.out.with_suffix(".logits.npz"),
        **{f"{k}_{f}": v[k] for f, v in saved.items() for k in ("test_idx", "logits")},
    )
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
