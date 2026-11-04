"""Phase A: train the side-car alone, on the frozen ViT's cached tokens (A0's probes, A1's runs).

    python -m gallops.drivejepa_perception.train_sidecar overfit --tokens T --labels L --out O
    python -m gallops.drivejepa_perception.train_sidecar train --tokens T --labels L --out O --seed 0
        [--tap post|vit_out|block12] [--layers 3] [--queries 64] [--epochs 12] [--rows train|tap] [--blind]

Only the side-car trains; the ViT, the projector and the whole planner are not even loaded. The
recipe is H04's (train_agent_decoder.py): AdamW 2e-4, weight decay 0.01, batch 64, fp16 autocast,
cosine over the run per step, gradient clip 35.

**Which epoch, and which numbers.** Every epoch is saved and read on the select logs. The kept
epoch is the one with the best F1 at 2 m within 30 m on select. The report logs are read once,
after training, at the kept epoch and at the fixed last epoch -- H14 found that the kept epoch
flatters every arm, so the fixed one is quoted beside it.

**Checks.** `preflight.run` before the first step, with no way to skip it. `overfit` is H04's check
1: 64 train samples, the same batch every step, constant LR, until the centre error of matched slots
is under 0.5 m and their class accuracy over 95%, or 4,000 steps. The sight check is read after
training: with the tokens zeroed, the detections should all but vanish.

`--blind` zeroes the tokens throughout: the no-camera baseline, trained identically.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from gallops.drivejepa_perception import evaluate, navsim_data, sidecar
from gallops.drivejepa_perception import targets as tg

REPO = Path(__file__).resolve().parents[3]
LR, WEIGHT_DECAY, BATCH, CLIP = 2e-4, 0.01, 64, 35.0
TAP_DIMS = {"post": 256, "vit_out": 1024, "block12": 1024}


class Rows(torch.utils.data.Dataset):
    """(tokens, projection, targets) for a set of cache rows."""

    def __init__(self, tokens, slot_of_row, rows, projections, targets, blind=False):
        self.tokens, self.slot, self.rows = tokens, slot_of_row, np.asarray(rows)
        self.projections, self.targets, self.blind = projections, targets, blind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = int(self.rows[i])
        tok = np.array(self.tokens[self.slot(row)], dtype=np.float16)  # a copy: the memmap is read-only
        if self.blind:
            tok = np.zeros_like(tok)
        item = {"tokens": torch.from_numpy(tok), "projection": torch.from_numpy(self.projections[row])}
        item.update({k: torch.from_numpy(v) for k, v in self.targets[row].items()})
        return item


def load(tokens_dir: Path, labels_dir: Path, tap: str, vit_tokens=None):
    """The cache, the splits' rows, and the side-car's targets for every row.

    `vit_tokens`, when given, is a ViT-out pass over every row (tokens.py --vit-out-all): the tap A0
    chose, for A1 and everything after it. Without it, a 1024-d tap exists only for A0's rows.
    """
    tokens_dir, labels_dir = Path(tokens_dir), Path(labels_dir)
    labels = dict(np.load(labels_dir / "labels.npz"))
    logs = json.loads((labels_dir / "manifest.json").read_text())["logs"]
    token_logs = np.array([logs[i] for i in labels["log_id"]])
    splits = json.loads((REPO / "recipes/drivejepa_perception/splits.json").read_text())
    rows = {name: np.flatnonzero(np.isin(token_logs, splits[name])) for name in ("train", "select", "report")}
    tap_rows = np.load(tokens_dir / "tap_rows.npy")
    rows["tap"] = np.intersect1d(tap_rows, rows["train"])
    if tap == "post" or (tap == "vit_out" and vit_tokens):
        source = Path(vit_tokens) / "vit_out.npy" if tap == "vit_out" else tokens_dir / "tokens.npy"
        tokens = np.load(source, mmap_mode="r")

        def slot(row):
            return row
    else:
        tokens = np.load(tokens_dir / f"tap_{tap}.npy", mmap_mode="r")
        where = {int(r): i for i, r in enumerate(tap_rows)}

        def slot(row):
            return where[row]

    projections = tg.front_projection(np.load(tokens_dir / "lidar2img.npy")).astype(np.float32)
    targets = [tg.build(navsim_data.agents_of(labels, r), projections[r]) for r in range(len(token_logs))]
    return tokens, slot, rows, projections, targets


def prior_gap(targets, rows) -> float:
    """The no-camera lead prior: the mean gap to the lead vehicle over the train rows that have one."""
    gaps = []
    for r in rows:
        t = targets[r]
        gap = evaluate.lead(t["box"][t["mask"], :2].astype(np.float64), t["cls"][t["mask"]])
        if gap is not None:
            gaps.append(gap)
    return float(np.mean(gaps)) if gaps else 25.0


def predict(model, dataset, device, batch=128, blind=False):
    """Decoded last-layer outputs for every row of `dataset`."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=8)
    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for item in loader:
            tok = item["tokens"].to(device)
            if blind:
                tok = torch.zeros_like(tok)
            out = model(tok, item["projection"].to(device))[-1]
            out = {k: v.float().cpu().numpy() for k, v in out.items()}
            preds += [evaluate.decode({k: v[b] for k, v in out.items()}) for b in range(len(tok))]
    return preds


def read(model, dataset, targets, rows, device, gap, blind=False) -> dict:
    preds = predict(model, dataset, device, blind=blind)
    result = evaluate.readout(preds, [targets[r] for r in rows], gap)
    result["detections_per_sample"] = float(np.mean([(p["conf"] >= evaluate.THR).sum() for p in preds]))
    return result


def f1(m: dict) -> float:
    r, p = m["recall_2m_within_30m"], m["precision_2m_within_30m"]
    return 0.0 if not (r + p) or math.isnan(r + p) else 2 * r * p / (r + p)


def train(cfg, args) -> None:
    from gallops.drivejepa_perception import preflight

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a run never overwrites one")
    tokens, slot, rows, projections, targets = load(args.tokens, args.labels, args.tap, args.vit_tokens)
    train_rows = rows["tap"] if args.rows == "tap" else rows["train"]
    select_rows = rows["select"]
    report = preflight.run(
        cfg,
        Path(args.tokens),
        Path(args.labels),
        {"train": train_rows, "select": select_rows, "report": rows["report"]},
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    model = sidecar.SideCar(d_in=TAP_DIMS[args.tap], layers=args.layers, queries=args.queries).to(device)
    criterion = sidecar.SideCarLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_rows) // BATCH
    total = steps_per_epoch * args.epochs
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )
    scaler = torch.cuda.amp.GradScaler()
    gap = prior_gap(targets, train_rows)
    generator = torch.Generator().manual_seed(args.seed)
    train_set = Rows(tokens, slot, train_rows, projections, targets, blind=args.blind)
    select_set = Rows(tokens, slot, select_rows, projections, targets, blind=args.blind)
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        generator=generator,
        pin_memory=True,
    )
    out.mkdir(parents=True)
    params = sum(p.numel() for p in model.parameters())
    meta = {
        "args": vars(args),
        "params": params,
        "preflight": report,
        "prior_gap_m": gap,
        "train_git_sha": report["code"]["sha"],
        "rows": {k: int(len(v)) for k, v in rows.items()},
    }
    (out / "run.json").write_text(json.dumps(meta, indent=1, default=str))
    print(
        f"side-car {params / 1e6:.2f} M params; {len(train_rows)} train rows; {steps_per_epoch} steps/epoch",
        flush=True,
    )
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, count = {}, 0
        for item in loader:
            item = {k: v.to(device, non_blocking=True) for k, v in item.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                outs = model(item["tokens"], item["projection"])
            loss, parts = criterion(outs, item)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch}")
            sums["loss"] = sums.get("loss", 0.0) + float(loss)
            count += 1
        metrics = read(model, select_set, targets, select_rows, device, gap)
        record = {
            "epoch": epoch,
            "train_loss": sums["loss"] / max(count, 1),
            "select": metrics,
            "f1_30m": f1(metrics),
        }
        record["seconds"] = round(time.time() - t0)
        with open(out / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
        torch.save(
            {"state_dict": model.state_dict(), "epoch": epoch, "meta": meta}, out / f"epoch_{epoch:02d}.pt"
        )
        recall, precision = metrics["recall_2m_within_30m"], metrics["precision_2m_within_30m"]
        print(
            f"epoch {epoch}: loss {record['train_loss']:.3f} select recall {recall:.3f} "
            f"precision {precision:.3f} f1 {record['f1_30m']:.3f} ({record['seconds']} s)",
            flush=True,
        )
    finish(model, out, args, tokens, slot, rows, projections, targets, gap, device)


def finish(model, out, args, tokens, slot, rows, projections, targets, gap, device) -> None:
    """The kept epoch (best select F1 within 30 m), then the report logs at it and at the last epoch."""
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    kept = max(records, key=lambda r: r["f1_30m"])["epoch"]
    report_set = Rows(tokens, slot, rows["report"], projections, targets, blind=args.blind)
    select_set = Rows(tokens, slot, rows["select"], projections, targets, blind=args.blind)
    summary = {
        "kept_epoch": kept,
        "last_epoch": args.epochs,
        "selection_rule": "best select F1 at 2 m within 30 m",
    }
    # A 1024-d tap exists only for A0's rows (the select logs and 10,000 train samples; tokens.py), so
    # an A0 probe on it is read on the select logs alone, as PREDICTIONS.md P1 says.
    has_report = args.tap == "post" or bool(getattr(args, "vit_tokens", ""))
    for name, epoch in (("kept", kept), ("last", args.epochs)) if has_report else ():
        state = torch.load(out / f"epoch_{epoch:02d}.pt", map_location="cpu")["state_dict"]
        model.load_state_dict(state)
        summary[f"report_{name}"] = read(model, report_set, targets, rows["report"], device, gap)
    if not has_report:
        summary["report"] = "none: this tap has no report rows (A0 reads the select logs)"
    model.load_state_dict(torch.load(out / f"epoch_{kept:02d}.pt", map_location="cpu")["state_dict"])
    summary["sight_select"] = {
        "with_tokens": read(model, select_set, targets, rows["select"], device, gap)["detections_per_sample"],
        "tokens_zeroed": read(model, select_set, targets, rows["select"], device, gap, blind=True)[
            "detections_per_sample"
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("kept_epoch", "sight_select")}), flush=True)


def overfit(cfg, args) -> None:
    """H04's check 1 on 64 train samples; exits 1 if it does not pass."""
    from gallops.drivejepa_perception import preflight

    tokens, slot, rows, projections, targets = load(args.tokens, args.labels, args.tap, args.vit_tokens)
    picked = np.random.default_rng(0).choice(rows["train"], 64, replace=False)
    preflight.run(cfg, Path(args.tokens), Path(args.labels), {"train": picked, "select": rows["select"][:64]})
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = sidecar.SideCar(d_in=TAP_DIMS[args.tap], layers=args.layers).to(device)
    criterion, optimiser = (
        sidecar.SideCarLoss(),
        torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
    )
    batch = torch.utils.data.default_collate(
        [Rows(tokens, slot, picked, projections, targets)[i] for i in range(64)]
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    for step in range(1, 4001):
        model.train()
        outs = model(batch["tokens"].float(), batch["projection"])
        loss, parts = criterion(outs, batch)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimiser.step()
        if step % 100 == 0:
            bi, pi, ti = parts["_match"]
            centre = (outs[-1]["box"][bi, pi, :2] - batch["box"][bi, ti, :2]).norm(dim=-1).mean().item()
            accuracy = (outs[-1]["logits"][bi, pi].argmax(-1) == batch["cls"][bi, ti]).float().mean().item()
            print(
                f"step {step}: loss {float(loss):.3f} centre {centre:.3f} m class acc {accuracy:.3f}",
                flush=True,
            )
            if centre < 0.5 and accuracy > 0.95:
                print(f"overfit check passed at step {step}")
                return
    raise SystemExit("overfit check failed: 4,000 steps without centre < 0.5 m and class accuracy > 95%")


def agents(cfg, args) -> None:
    """The agents pass: an A1 run's kept epoch decoded on every row, for phase B's arms F and S.

    Stored in the format arm O builds from the labels (train_scorer.truth_agents), with the slots
    under THR masked out: the scorer reads only agents a person would be shown (plan A3).
    """
    from gallops.drivejepa_b2d.provenance import git_sha

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    run = Path(args.run)
    meta = json.loads((run / "run.json").read_text())
    epoch = json.loads((run / "summary.json").read_text())["kept_epoch"]
    run_args = meta["args"]
    tokens, slot, rows, projections, _ = load(
        args.tokens, args.labels, run_args["tap"], run_args.get("vit_tokens", "")
    )
    device = torch.device("cuda")
    model = sidecar.SideCar(
        d_in=TAP_DIMS[run_args["tap"]], layers=run_args["layers"], queries=run_args["queries"]
    )
    model.load_state_dict(torch.load(run / f"epoch_{epoch:02d}.pt", map_location="cpu")["state_dict"])
    model = model.to(device).eval()
    n = len(projections)
    q = run_args["queries"]
    keys = {
        "cls": ((q,), np.int64),
        "conf": ((q,), np.float32),
        "xyz": ((q, 3), np.float32),
        "lwh": ((q, 3), np.float32),
        "yaw": ((q,), np.float32),
        "vel": ((q, 2), np.float32),
        "path": ((q, tg.PATH_STEPS, 2), np.float32),
        "mask": ((q,), bool),
    }
    arrays = {k: np.zeros((n, *shape), dtype) for k, (shape, dtype) in keys.items()}
    every = np.arange(n)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, n, 256):
            chunk = every[i : i + 256]
            tok = torch.from_numpy(
                np.stack([np.asarray(tokens[slot(int(r))], dtype=np.float16) for r in chunk])
            ).to(device)
            out_last = model(tok, torch.from_numpy(projections[chunk]).to(device))[-1]
            o = {k: v.float().cpu().numpy() for k, v in out_last.items()}
            prob = np.exp(o["logits"] - o["logits"].max(-1, keepdims=True))
            prob /= prob.sum(-1, keepdims=True)
            conf = 1.0 - prob[..., tg.NONE]
            arrays["conf"][chunk] = conf
            arrays["cls"][chunk] = prob[..., : tg.NUM_CLASSES].argmax(-1)
            arrays["xyz"][chunk] = o["box"][..., :3]
            arrays["lwh"][chunk] = np.exp(o["box"][..., 3:6])
            arrays["yaw"][chunk] = np.arctan2(o["box"][..., 6], o["box"][..., 7])
            arrays["vel"][chunk] = o["vel"]
            arrays["path"][chunk] = o["path"]
            arrays["mask"][chunk] = conf >= evaluate.THR
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    np.savez(staging / "agents.npz", **arrays)
    manifest = {
        "run": str(run),
        "epoch": epoch,
        "gallops_sha": git_sha(),
        "rows": n,
        "train_git_sha": meta["train_git_sha"],
        "threshold": evaluate.THR,
        "agents_per_row": float(arrays["mask"].sum(1).mean()),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(json.dumps(manifest, indent=1))


def refinish(cfg, args) -> None:
    """Write a trained run's summary from its saved epochs, without retraining (its own args)."""
    out = Path(args.out)
    if (out / "summary.json").exists():
        raise SystemExit(f"{out}/summary.json exists")
    run_args = argparse.Namespace(**json.loads((out / "run.json").read_text())["args"])
    run_args.vit_tokens = getattr(run_args, "vit_tokens", "")
    tokens, slot, rows, projections, targets = load(
        run_args.tokens, run_args.labels, run_args.tap, run_args.vit_tokens
    )
    train_rows = rows["tap"] if run_args.rows == "tap" else rows["train"]
    model = sidecar.SideCar(d_in=TAP_DIMS[run_args.tap], layers=run_args.layers, queries=run_args.queries)
    device = torch.device("cuda")
    finish(
        model.to(device),
        out,
        run_args,
        tokens,
        slot,
        rows,
        projections,
        targets,
        prior_gap(targets, train_rows),
        device,
    )


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.train_sidecar")
    parser.add_argument("cmd", choices=("train", "overfit", "agents", "finish"))
    parser.add_argument("--run", default="", help="agents: the A1 run directory to decode")
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tap", choices=tuple(TAP_DIMS), default="post")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--queries", type=int, default=tg.QUERIES)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--rows", choices=("train", "tap"), default="train")
    parser.add_argument("--blind", action="store_true")
    parser.add_argument(
        "--vit-tokens", default="", help="a ViT-out pass over every row (tokens --vit-out-all)"
    )
    args = parser.parse_args(argv)
    cfg = SiteConfig.load()
    {"train": train, "overfit": overfit, "agents": agents, "finish": refinish}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
