"""B-2-0: what B-2's arms read, built and checked before anything trains.

    python -m gallops.drivejepa_perception.b2_inputs folds \
        --write recipes/drivejepa_perception/part_b2_folds.json
    python -m gallops.drivejepa_perception.b2_inputs hidden --source navtrain --tokens {out}/tokens-v1 \
        --labels {out}/labels-v1 --vit-tokens {out}/tokens-vit-v1 --runs R0,R1,R2,R3,R4 \
        --crossfit {out}/agents-train-crossfit-v3 --out {out}/b2-hidden-navtrain-v1
    python -m gallops.drivejepa_perception.b2_inputs hidden --source nusc --nusc {out}/nusc-v1 \
        --runs R0,R1,R2,R3,R4 --out {out}/b2-hidden-nusc-v1
    python -m gallops.drivejepa_perception.b2_inputs hidden --source navtest ... --fold-shift 1 \
        --out {out}/b2-hidden-navtest-shift1-v1
    python -m gallops.drivejepa_perception.b2_inputs align --runs R0,...,R4 \
        --out {out}/part-b/b2/alignment.json ...

docs/drivejepa-perception-sidecar/part-b/B-2-hidden-refiner.md, Q8 and step 1. B-2 reads the five cross-fit
side-cars of B-0-1 (`H15.r8`) everywhere, at test too: hidden states are each model's own coordinates, so a
reader trained on the fold side-cars' slots cannot read another model's.

**The folds.** A train row reads the side-car that did not train on its log (`train_sidecar.crossfit_folds`).
Select, R-clean and navtest logs belong to no fold; each is assigned one by FOLD_RULE, written once to
`part_b2_folds.json` and re-derived (and refused if edited) by every reader.

**The hidden pass.** Each fold's side-car decodes its rows exactly as `train_sidecar.agent_arrays` does (the
same chunks, fp16 autocast), and writes, beside the decoded list (layer 6, the agents arm D reads), the slots
of layers 3 and 6: each slot's hidden vector (fp16), its centre (the reference point, `box[..., :3]`) and its
confidence (1 - p(none)). Its gates: each fold reproduces the select F1 it recorded, and on navtrain's train
rows the decoded list equals `agents-train-crossfit-v3` exactly. No report-log row is decoded.

**nuScenes** (the out-of-domain read, `ood`'s keyframes): no fold record holds its scenes, so the pass deals
them by the same rule (NUSC_RULE) and writes the table and its sha into its manifest. Its gate: each fold's
decoded list equals `train_sidecar.agent_arrays` run by the same model on the same rows.

**A shifted pass** (`--fold-shift k`, navtest only) decodes each row by fold (f + k) mod 5 in place of its own
fold f: a test-time swap, "another fold's side-car" (B-2's Q7), read through that fold's input map.

**The alignment check** decides whether the five side-cars' layer-6 spaces are one space (ALIGN_BAR,
pre-registered in PREDICTIONS.md, Part B · B-2): fitted on select rows, read on R-clean rows, every fold
against fold 0, slots matched by index (the five share their seed-0 start) and by decoded centre.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FOLDS_FILE = "recipes/drivejepa_perception/part_b2_folds.json"
#: A non-train log's fold: the sorted logs in a seeded permutation, dealt round robin (as crossfit_folds deals
#: the train logs), so the folds' sizes differ by at most one.
FOLD_SEED = 0
FOLD_RULE = (
    "train rows: the cross-fit fold of their log (train_sidecar.crossfit_folds over splits.json's train "
    "logs); select, navtest and R-clean logs: each split's sorted logs in "
    f"default_rng({FOLD_SEED}).permutation, dealt round robin into 5 folds"
)
NUSC_RULE = (
    f"nuScenes keyframes: their sorted scenes in default_rng({FOLD_SEED}).permutation, dealt round robin "
    "into 5 folds (FOLD_RULE's rule for a non-train split)"
)
LAYERS = (3, 6)
#: The alignment rule (PREDICTIONS.md, Part B · B-2): R-squared on R-clean's slots at confidence >= 0.1.
ALIGN_BAR = 0.9
ALIGN_CONF = 0.1


def fold_of_logs(logs: list, k: int = 5, seed: int = FOLD_SEED) -> dict:
    """log -> fold for a split's logs: a seeded permutation of the sorted logs, dealt round robin."""
    ordered = sorted(logs)
    order = np.random.default_rng(seed).permutation(len(ordered))
    return {ordered[i]: int(position % k) for position, i in enumerate(order)}


def _sha(lines: list) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def derive_folds() -> dict:
    """The fold record, derived from the committed lists alone."""
    from gallops.drivejepa_perception.train_sidecar import crossfit_folds

    splits = json.loads((REPO / "recipes/drivejepa_perception/splits.json").read_text())
    navtest = json.loads((REPO / "recipes/drivejepa_perception/navtest.json").read_text())
    fresh = json.loads((REPO / "recipes/drivejepa_perception/fresh.json").read_text())
    train = {log: f for f, logs in enumerate(crossfit_folds(splits["train"])) for log in logs}
    record = {"rule": FOLD_RULE, "seed": FOLD_SEED, "folds": 5, "train": train}
    for name, logs in (("select", splits["select"]), ("navtest", navtest["logs"]), ("fresh", fresh["logs"])):
        if set(logs) & set(splits["train"]):
            raise SystemExit(f"{name} shares logs with the train split")
        record[name] = fold_of_logs(logs)
    record["sha256"] = {
        name: _sha([f"{log} {fold}" for log, fold in sorted(record[name].items())])
        for name in ("train", "select", "navtest", "fresh")
    }
    return record


def load_folds(path=None) -> dict:
    """The committed fold record, re-derived and refused if it differs."""
    stored = json.loads((REPO / (path or FOLDS_FILE)).read_text())
    if stored != derive_folds():
        raise SystemExit(f"{path or FOLDS_FILE} differs from what its rule derives: it was edited")
    return stored


def row_folds(record: dict, split: str, token_logs: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Each row's fold (the side-car it reads), for `rows` of a split: train, select, navtest or fresh."""
    table = record[split]
    return np.array([table[str(token_logs[r])] for r in rows], dtype=np.int8)


# ------------------------------------------------------------------------------------------- the pass

KEYS = {
    "cls": ((), np.int64),
    "conf": ((), np.float32),
    "xyz": ((3,), np.float32),
    "lwh": ((3,), np.float32),
    "yaw": ((), np.float32),
    "vel": ((2,), np.float32),
    "path": ((2, 2), np.float32),
    "mask": ((), bool),
}


def decode_rows(
    model, tokens, slot, rows, projections, device, sink: dict, chunk: int = 256, at=None
) -> None:
    """`train_sidecar.agent_arrays` on `rows`, with layers 3 and 6's slots written into `sink` beside the
    decoded list. The chunking and autocast are agent_arrays' own, so the decoded list is bit-equal to it.
    `at` gives each row's position in `sink` (default: the row itself)."""
    import torch

    from gallops.drivejepa_perception import evaluate
    from gallops.drivejepa_perception import targets as tg

    model = model.to(device).eval()
    rows = np.asarray(rows)
    at = rows if at is None else np.asarray(at)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(rows), chunk):
            part, where = rows[i : i + chunk], at[i : i + chunk]
            tok = (
                torch.from_numpy(np.stack([np.asarray(tokens[slot(int(r))], dtype=np.float16) for r in part]))
                .to(device)
                .float()
            )
            outs = model(tok, torch.from_numpy(projections[part]).to(device))
            last = {k: v.float().cpu().numpy() for k, v in outs[-1].items() if k != "hidden"}
            prob = np.exp(last["logits"] - last["logits"].max(-1, keepdims=True))
            prob /= prob.sum(-1, keepdims=True)
            conf = 1.0 - prob[..., tg.NONE]
            sink["conf"][where] = conf
            sink["cls"][where] = prob[..., : tg.NUM_CLASSES].argmax(-1)
            sink["xyz"][where] = last["box"][..., :3]
            sink["lwh"][where] = np.exp(last["box"][..., 3:6])
            sink["yaw"][where] = np.arctan2(last["box"][..., 6], last["box"][..., 7])
            sink["vel"][where] = last["vel"]
            sink["path"][where] = last["path"]
            sink["mask"][where] = conf >= evaluate.THR
            for layer in LAYERS:
                out = outs[layer - 1]
                logits = out["logits"].float().cpu().numpy()
                p = np.exp(logits - logits.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                sink[f"hidden{layer}"][where] = out["hidden"].to(torch.float16).cpu().numpy()
                sink[f"centre{layer}"][where] = out["box"][..., :3].float().cpu().numpy()
                sink[f"conf{layer}"][where] = 1.0 - p[..., tg.NONE]


def hidden(cfg, args) -> dict:
    """The hidden pass over one source (module docstring)."""
    import torch

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import navsim_data, train_sidecar

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    record = load_folds()
    runs = [Path(r) for r in args.runs.split(",")]
    if len(runs) != 5:
        raise SystemExit("five cross-fit side-cars, fold 0 to 4")
    run_args = json.loads((runs[0] / "run.json").read_text())["args"]
    tables = dict(record)
    if args.source == "nusc":
        from gallops.drivejepa_perception import ood
        from gallops.drivejepa_perception import targets as tg

        nusc = Path(args.nusc)
        outputs = np.load(nusc / "outputs.npz")
        tokens, slot = np.load(nusc / "vit_out.npy", mmap_mode="r"), int
        front = outputs["lidar2img"][:, tg.FRONT_SLOT].astype(np.float64)
        projections = (tg.to_512(*ood.NUSC_SIZE) @ front).astype(np.float32)
        rows = {"all": np.flatnonzero(outputs["built"])}
        token_logs, source = outputs["scene"], "nusc"
        tables["nusc"] = fold_of_logs(sorted(set(token_logs.tolist())))
    else:
        tokens, slot, rows, projections, _ = train_sidecar.load(
            args.tokens, args.labels, run_args["tap"], args.vit_tokens
        )
        labels = np.load(Path(args.labels) / "labels.npz")
        manifest = json.loads((Path(args.labels) / "manifest.json").read_text())
        source = navsim_data.source_of(manifest)
        if source != args.source:
            raise SystemExit(f"{args.labels} is {source}'s labels pass, not {args.source}'s")
        token_logs = np.array(manifest["logs"])[labels["log_id"]]
    n = len(projections)
    if source == "navtrain":
        parts = {"train": rows["train"], "select": rows["select"]}
        if np.isin(np.concatenate(list(parts.values())), rows["report"]).any():
            raise SystemExit("a report-log row is in the row set")
    else:
        parts = {source: rows["all"]}
    fold = np.full(n, -1, np.int8)
    for name, picked in parts.items():
        fold[picked] = row_folds(tables, name, token_logs, picked)
    if args.fold_shift:
        if source != "navtest":
            raise SystemExit("a shifted pass is a test-time swap: navtest only")
        fold[fold >= 0] = (fold[fold >= 0] + args.fold_shift) % 5
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    d = run_args.get("d_model") or 256
    q = run_args.get("queries") or 64
    sink = {k: np.zeros((n, q, *shape), dtype) for k, (shape, dtype) in KEYS.items()}
    sink["lwh"][:] = 1.0
    for layer in LAYERS:
        sink[f"hidden{layer}"] = np.lib.format.open_memmap(
            staging / f"hidden{layer}.npy", mode="w+", dtype=np.float16, shape=(n, q, d)
        )
        sink[f"centre{layer}"] = np.zeros((n, q, 3), np.float32)
        sink[f"conf{layer}"] = np.zeros((n, q), np.float32)
    device = torch.device("cuda")
    checks, nusc_gate = [], {}
    t0 = time.time()
    for k, run in enumerate(runs):
        meta = json.loads((run / "run.json").read_text())["args"]
        if meta.get("crossfit_fold") != k:
            raise SystemExit(f"{run} is not fold {k}'s run")
        model, epoch = train_sidecar.load_run(run)
        check = train_sidecar.select_f1_check(model, run, epoch, device)
        check["fold"] = k
        checks.append(check)
        if check["gap"] > 1e-6:
            raise SystemExit(f"fold {k}, decoded here, does not give the select F1 it recorded: {check}")
        # Each split's rows in their own chunks: the train rows then batch exactly as crossfit_agents batched
        # them, so the gate below can ask for equality, not closeness.
        for name, picked in parts.items():
            mine = picked[fold[picked] == k]
            decode_rows(model, tokens, slot, mine, projections, device, sink)
            print(f"fold {k}, {name}: {len(mine)} rows ({time.time() - t0:.0f} s)", flush=True)
            if source == "nusc":
                plain = train_sidecar.agent_arrays(model, tokens, slot, mine, projections, device)
                nusc_gate[str(k)] = {key: int((sink[key][mine] != plain[key][mine]).sum()) for key in KEYS}
        del model
        torch.cuda.empty_cache()
    gate = {}
    if source == "nusc":
        gate["folds_equal_agent_arrays"] = nusc_gate
        if any(v for per in nusc_gate.values() for v in per.values()):
            raise SystemExit(f"the decoded list differs from agent_arrays on the same rows: {nusc_gate}")
    if source == "navtrain":
        crossfit = np.load(Path(args.crossfit) / "agents.npz")
        train_rows = parts["train"]
        unequal = {k: int((sink[k][train_rows] != crossfit[k][train_rows]).sum()) for k in KEYS}
        gate["train_rows_equal_crossfit"] = {"unequal_values": unequal, "rows": int(len(train_rows))}
        if any(unequal.values()):
            raise SystemExit(f"the train rows' decoded list differs from {args.crossfit}: {unequal}")
    np.savez(staging / "agents.npz", **{k: sink[k] for k in KEYS})
    for layer in LAYERS:
        sink[f"hidden{layer}"].flush()
        np.save(staging / f"centre{layer}.npy", sink[f"centre{layer}"])
        np.save(staging / f"conf{layer}.npy", sink[f"conf{layer}"])
    np.save(staging / "fold.npy", fold)
    written = {
        "pass": "b2-hidden",
        "source": source,
        "gallops_sha": git_sha(),
        "runs": [str(r) for r in runs],
        "folds_file": FOLDS_FILE,
        "folds_sha256": record["sha256"],
        **(
            {
                "nusc_rule": NUSC_RULE,
                "nusc_folds": tables["nusc"],
                "nusc_folds_sha256": _sha([f"{s} {f}" for s, f in sorted(tables["nusc"].items())]),
            }
            if source == "nusc"
            else {}
        ),
        "rows": n,
        "rows_decoded": {name: int(len(p)) for name, p in parts.items()},
        "rows_per_fold": {str(k): int((fold == k).sum()) for k in range(5)},
        "fold_shift": int(args.fold_shift),
        "layers": list(LAYERS),
        "hidden_dtype": "float16",
        "d_model": d,
        "slots": q,
        "select_f1_checks": checks,
        "gate": gate,
        "inputs": {"nusc": str(args.nusc)}
        if source == "nusc"
        else {
            "tokens": str(args.tokens),
            "labels": str(args.labels),
            "vit_tokens": str(args.vit_tokens),
        },
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(written, indent=1))
    staging.rename(out)
    print(json.dumps({k: v for k, v in written.items() if k != "select_f1_checks"}, indent=1))
    return written


# -------------------------------------------------------------------------------------- the alignment


def r_squared(pred: np.ndarray, target: np.ndarray) -> float:
    """1 - SS_res / SS_tot over every dimension, the target's per-dimension mean as the baseline."""
    res = ((target - pred) ** 2).sum()
    tot = ((target - target.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - res / tot)


def ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> tuple:
    """Least squares y ~ x W + b with an L2 penalty `lam` on W."""
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    w = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]), xc.T @ yc)
    return w, ym - xm @ w


def match_by_centre(centre_a: np.ndarray, centre_b: np.ndarray) -> np.ndarray:
    """For one row, slot of B matched to each slot of A by Hungarian on centre distance."""
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(centre_a[:, None, :2] - centre_b[None, :, :2], axis=-1)
    i, j = linear_sum_assignment(cost)
    out = np.empty(len(centre_a), np.int64)
    out[i] = j
    return out


def align(cfg, args) -> dict:
    """The alignment check (module docstring): every fold's layer-6 slots against fold 0's."""
    import torch

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import train_sidecar

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    runs = [Path(r) for r in args.runs.split(",")]
    run_args = json.loads((runs[0] / "run.json").read_text())["args"]
    tok_s, slot_s, rows_s, proj_s, _ = train_sidecar.load(
        args.tokens, args.labels, run_args["tap"], args.vit_tokens
    )
    tok_f, slot_f, rows_f, proj_f, _ = train_sidecar.load(
        args.fresh_tokens, args.fresh_labels, run_args["tap"], args.fresh_vit_tokens
    )
    fit_rows = rows_s["select"]
    eval_rows = np.sort(
        np.random.default_rng(0).choice(rows_f["all"], min(args.eval_rows, len(rows_f["all"])), False)
    )
    device = torch.device("cuda")
    per = {}
    for name, (tok, slot, rows, proj) in {
        "fit": (tok_s, slot_s, fit_rows, proj_s),
        "eval": (tok_f, slot_f, eval_rows, proj_f),
    }.items():
        per[name] = []
        for run in runs:
            model, _ = train_sidecar.load_run(run)
            n, q = len(rows), model.query.shape[0]
            sink = {k: np.zeros((n, q, *shape), dtype) for k, (shape, dtype) in KEYS.items()}
            for layer in LAYERS:
                sink[f"hidden{layer}"] = np.zeros((n, q, model.query.shape[1]), np.float16)
                sink[f"centre{layer}"] = np.zeros((n, q, 3), np.float32)
                sink[f"conf{layer}"] = np.zeros((n, q), np.float32)
            decode_rows(model, tok, slot, rows, proj, device, sink, at=np.arange(n))
            per[name].append({k: sink[k] for k in ("hidden6", "centre6", "conf6")})
            del model
            torch.cuda.empty_cache()
    result = {
        "bar": ALIGN_BAR,
        "conf": ALIGN_CONF,
        "fit_rows": int(len(fit_rows)),
        "eval_rows": int(len(eval_rows)),
    }
    folds = {}
    for k in range(1, len(runs)):
        entry = {}
        for how in ("index", "centre"):
            pairs = {}
            for name in ("fit", "eval"):
                a, b = per[name][k], per[name][0]
                xs, ys = [], []
                for r in range(len(a["hidden6"])):
                    j = (
                        np.arange(a["hidden6"].shape[1])
                        if how == "index"
                        else match_by_centre(a["centre6"][r], b["centre6"][r])
                    )
                    keep = b["conf6"][r][j] >= ALIGN_CONF
                    xs.append(a["hidden6"][r][keep].astype(np.float64))
                    ys.append(b["hidden6"][r][j][keep].astype(np.float64))
                pairs[name] = (np.concatenate(xs), np.concatenate(ys))
            (xf, yf), (xe, ye) = pairs["fit"], pairs["eval"]
            w, bias = ridge(xf, yf)
            entry[how] = {
                "identity_r2": r_squared(xe, ye),
                "ridge_r2": r_squared(xe @ w + bias, ye),
                "eval_slots": int(len(xe)),
            }
        folds[str(k)] = entry
    result["folds"] = folds
    identity_ok = all(v["index"]["identity_r2"] >= ALIGN_BAR for v in folds.values())
    mapped_ok = all(max(v["index"]["ridge_r2"], v["centre"]["ridge_r2"]) >= ALIGN_BAR for v in folds.values())
    result["decision"] = "shared" if identity_ok else ("mapped" if mapped_ok else "per-fold")
    result["gallops_sha"] = git_sha()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.b2_inputs")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("folds")
    p.add_argument("--write", required=True, help="the committed fold record; if it exists, derived again")
    p = sub.add_parser("hidden")
    p.add_argument("--source", required=True, choices=("navtrain", "navtest", "fresh", "nusc"))
    p.add_argument("--tokens", default="")
    p.add_argument("--labels", default="")
    p.add_argument("--vit-tokens", default="")
    p.add_argument("--nusc", default="", help="nusc: `ood`'s keyframe cache (outputs.npz, vit_out.npy)")
    p.add_argument("--fold-shift", type=int, default=0, help="navtest: every row read by fold (f + k) mod 5")
    p.add_argument("--runs", required=True)
    p.add_argument(
        "--crossfit", default="", help="navtrain: the cross-fit agents pass the train rows must equal"
    )
    p.add_argument("--out", required=True)
    p = sub.add_parser("align")
    p.add_argument("--runs", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--vit-tokens", required=True)
    p.add_argument("--fresh-tokens", required=True)
    p.add_argument("--fresh-labels", required=True)
    p.add_argument("--fresh-vit-tokens", required=True)
    p.add_argument("--eval-rows", type=int, default=2000)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "folds":
        path = REPO / args.write
        record = derive_folds()
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise SystemExit(f"{args.write} differs from what its rule derives")
            print(f"{args.write} holds: derived again, equal")
            return
        path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.write}")
        return
    if args.cmd == "hidden":
        if args.source == "navtrain" and not args.crossfit:
            parser.error("navtrain's pass is held to the cross-fit agents: pass --crossfit")
        if args.source == "nusc" and not args.nusc:
            parser.error("nusc reads `ood`'s keyframe cache: pass --nusc")
        if args.source != "nusc" and not (args.tokens and args.labels and args.vit_tokens):
            parser.error("a NAVSIM source needs --tokens, --labels and --vit-tokens")
        hidden(SiteConfig.load(), args)
        return
    align(SiteConfig.load(), args)


if __name__ == "__main__":
    main()
