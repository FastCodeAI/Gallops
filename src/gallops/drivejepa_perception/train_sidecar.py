"""Phase A: train the side-car alone, on the frozen ViT's cached tokens (A0's probes, A1's runs).

    python -m gallops.drivejepa_perception.train_sidecar overfit --tokens T --labels L --out O
    python -m gallops.drivejepa_perception.train_sidecar train --tokens T --labels L --out O --seed 0
        [--tap post|vit_out|block12] [--variant v1|v2|v3] [--layers 3] [--queries 64] [--epochs 12]
        [--stop-after N] [--no-report] [--rows train|tap] [--blind]
    python -m gallops.drivejepa_perception.train_sidecar reread --tokens T --labels L --run R [--which kept]
    python -m gallops.drivejepa_perception.train_sidecar sight --tokens T --labels L --run R
    python -m gallops.drivejepa_perception.train_sidecar agents --tokens T --labels L --run R \
        [--vit-tokens V] --row-set all|select|train|train+select --out O

Only the side-car trains; the ViT, the projector and the whole planner are not even loaded. The
recipe is H04's (train_agent_decoder.py): AdamW 2e-4, weight decay 0.01, batch 64, fp16 autocast,
cosine over the run per step, gradient clip 35.

**Which epoch, and which numbers.** Every epoch is saved and read on the select logs. The kept
epoch is the one with the best F1 at 2 m within 30 m on select. The report logs are read once,
after training, at the kept epoch and at the fixed last epoch: the kept epoch is the best of
several on select, so the last, which nothing chose, is quoted beside it. `--no-report` skips
them, so a run can be reviewed on select before any report number exists (the rerun's plan,
docs/drivejepa-perception-sidecar/part-a/plan.md).

**How the run evolved.** Every LOG_EVERY steps, `steps.jsonl` gets the window's means: each loss
term of each layer, the learning rate, the gradient norm before clipping, the AMP loss scale, the
matched slots' centre error and class accuracy, and how long the step waited for data. Each
epoch's line in `metrics.jsonl` adds the epoch's means and the full select readout, which from the
rerun's step 5 includes query collapse (duplicates, slot use) and the recall of the targets that
matter (causal under both rules, visible, hidden, the k nearest).

**The pipeline check** is `--stop-after 1`: epoch 1 of the full schedule (the learning rate follows
`--epochs`, not the epochs run), so its curves are the first epoch of the run it stands for.

**Checks.** `preflight.run` before the first step, with no way to skip it. `overfit` is H04's check
1: 64 train samples, the same batch every step, constant LR, until the centre error of matched slots
is under 0.5 m and their class accuracy over 95%, or 4,000 steps. The sight check is read after
training: with the tokens zeroed, the detections should all but vanish.

`--blind` zeroes the tokens throughout: the no-camera baseline, trained identically.

**The agents pass (phase B).** A run's kept epoch decoded on a row set of any source's caches
(`navsim_data` sources: navtrain, navtest, fresh), into the format arm O builds from the labels.
- It decodes only the rows asked for; a navtrain row set never includes a report row (`row_set`).
- Before decoding, the run is read on its own select rows by this pass's decoding, then by
  `evaluate.readout`, and must give the select F1 its training recorded for that epoch (B0-G7 in
  PREDICTIONS.md): the check that the model, the epoch and the decoding are the ones that were
  measured. The new split's inputs are held to the live model by `preflight.run_data`.
- The manifest names its inputs (paths and their manifests' sha256) and the check's numbers.
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
#: The rerun's three architectures (docs/drivejepa-perception-sidecar/part-a/plan.md, section 6). v1 is A1's
#: side-car, 4.39 M on the ViT-out tap; v2 doubles its depth and its queries, 8.37 M; v3 also widens
#: it to 384, 18.44 M, inside the plan's 30.8 M budget.
VARIANTS = {
    "v1": {"d_model": 256, "heads": 8, "ffn": 1024, "layers": 3, "queries": 64},
    "v2": {"d_model": 256, "heads": 8, "ffn": 1024, "layers": 6, "queries": 128},
    "v3": {"d_model": 384, "heads": 12, "ffn": 1536, "layers": 6, "queries": 128},
}
#: Steps between two lines of steps.jsonl; each line holds the means over those steps.
LOG_EVERY = 50


def resolve_architecture(args) -> None:
    """Fill d_model, heads, ffn, layers and queries in place: from `--variant`, else v1's defaults.

    A flag given beside a variant must agree with it: two sources for one number is how a run
    comes to record an architecture it did not train.
    """
    preset = VARIANTS[args.variant] if args.variant else VARIANTS["v1"]
    for key, value in preset.items():
        given = getattr(args, key)
        if given is None:
            setattr(args, key, value)
        elif args.variant and given != value:
            raise SystemExit(
                f"--{key.replace('_', '-')} {given} contradicts --variant {args.variant} ({value})"
            )


def build_model(run_args: dict) -> sidecar.SideCar:
    """The side-car a run's arguments describe. Runs from before the width flags get v1's widths."""
    return sidecar.SideCar(
        d_in=TAP_DIMS[run_args["tap"]],
        d=run_args.get("d_model") or 256,
        heads=run_args.get("heads") or 8,
        ffn=run_args.get("ffn") or 1024,
        layers=run_args["layers"],
        queries=run_args.get("queries") or tg.QUERIES,
    )


def lr_factor(step: int, total: int) -> float:
    """The cosine schedule, per optimizer step, from 1 at step 0 to 0 at `total` (no warmup)."""
    return 0.5 * (1 + math.cos(math.pi * min(step, total) / total))


class Window:
    """Running sums of per-step numbers; `means` reads them out and starts again.

    A non-finite value (an AMP overflow's gradient norm) is counted apart, not averaged, so one
    skipped step cannot make a window's mean infinite.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sums, self.counts, self.nonfinite = {}, {}, {}

    def add(self, values: dict) -> None:
        for key, value in values.items():
            value = float(value)
            if math.isfinite(value):
                self.sums[key] = self.sums.get(key, 0.0) + value
                self.counts[key] = self.counts.get(key, 0) + 1
            else:
                self.nonfinite[key] = self.nonfinite.get(key, 0) + 1

    def means(self) -> dict:
        out = {key: self.sums[key] / self.counts[key] for key in self.sums}
        out.update({f"{key}_nonfinite": n for key, n in self.nonfinite.items()})
        self.reset()
        return out


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
    lmanifest = json.loads((labels_dir / "manifest.json").read_text())
    logs = lmanifest["logs"]
    token_logs = np.array([logs[i] for i in labels["log_id"]])
    tap_rows = np.load(tokens_dir / "tap_rows.npy")
    if navsim_data.source_of(lmanifest) == "navtrain":
        splits = json.loads((REPO / "recipes/drivejepa_perception/splits.json").read_text())
        rows = {
            name: np.flatnonzero(np.isin(token_logs, splits[name])) for name in ("train", "select", "report")
        }
        rows["tap"] = np.intersect1d(tap_rows, rows["train"])
    else:
        # Part B's other sources have no log splits: nothing trains on them, and all of them is read.
        rows = {"all": np.arange(len(token_logs))}
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


def read(model, dataset, targets, rows, device, gap, blind=False, detail=False, subsets=None) -> dict:
    """The readout on `rows`, with query collapse; with `subsets` (`importance_subsets`), the recall of
    the targets that matter too."""
    preds = predict(model, dataset, device, blind=blind)
    gts = [targets[r] for r in rows]
    result = evaluate.readout(preds, gts, gap, detail=detail)
    result.update(evaluate.slot_numbers(preds))
    if subsets is not None:
        result.update(evaluate.recall_of(preds, gts, subsets))
    return result


#: Nearest-k recall reads each sample's k nearest targets (plan §3.8).
NEAREST_K = (1, 3, 5)


def importance_subsets(labels_dir, projections, targets, rows) -> dict:
    """Per row of `rows`, which of its targets matter (plan §3.8, `importance.py`), as
    `evaluate.recall_of` reads them: causal under each of CAUSAL_RULES, visible and hidden, and the k
    nearest. Built once, at start-up: the labels never change. Each row's labels must cover exactly
    its targets, since both come from one `targets.target_index` on the same projection."""
    from gallops.drivejepa_perception import importance as im

    labels = dict(np.load(Path(labels_dir) / "labels.npz"))
    subsets: dict = {}
    for row in rows:
        lab = im.target_importance(
            navsim_data.agents_of(labels, row), projections[row], labels["ego_future"][row]
        )
        if len(lab["rank"]) != int(targets[row]["mask"].sum()):
            raise SystemExit(
                f"row {row}: importance labels for {len(lab['rank'])} targets, the targets have "
                f"{int(targets[row]['mask'].sum())}"
            )
        visible = np.nan_to_num(lab["visible"], nan=0.0) >= im.VISIBLE_MIN
        named = {f"causal_{name}": lab[f"causal_{name}"] for name in im.CAUSAL_RULES}
        named.update(visible=visible, hidden=~visible)
        named.update({f"nearest_{k}": lab["rank"] < k for k in NEAREST_K})
        for name, mask in named.items():
            subsets.setdefault(name, []).append(mask)
    return subsets


def f1(m: dict) -> float:
    r, p = m["recall_2m_within_30m"], m["precision_2m_within_30m"]
    return 0.0 if not (r + p) or math.isnan(r + p) else 2 * r * p / (r + p)


def match_numbers(out: dict, item: dict, match) -> dict:
    """The last layer's Hungarian-matched slots on one batch: how many, their centre error (x-y, m)
    and class accuracy. What the loss sees of the fit, read while training, not a readout."""
    bi, pi, ti = match
    if not len(bi):
        return {"targets": 0.0}
    with torch.no_grad():
        centre = (out["box"][bi, pi, :2] - item["box"][bi, ti, :2]).norm(dim=-1).mean()
        accuracy = (out["logits"][bi, pi].argmax(-1) == item["cls"][bi, ti]).float().mean()
    return {"targets": float(len(bi)), "match_centre_m": float(centre), "match_class_acc": float(accuracy)}


#: A2's draw of train logs (a learning curve): seeded, so every fraction is reproducible.
FRACTION_SEED = 0


def fraction_of_logs(labels_dir, train_rows: np.ndarray, fraction: float) -> np.ndarray:
    """The train rows of a seeded `fraction` of the train logs (A2). By log, not by row: half the
    data is half the scenes, not every scene thinned, since a log's samples are 0.5 s apart."""
    labels = np.load(Path(labels_dir) / "labels.npz")
    logs = np.array(json.loads((Path(labels_dir) / "manifest.json").read_text())["logs"])[labels["log_id"]]
    names = np.unique(logs[train_rows])
    picked = np.random.default_rng(FRACTION_SEED).choice(
        names, int(round(fraction * len(names))), replace=False
    )
    return train_rows[np.isin(logs[train_rows], picked)]


def train(cfg, args) -> None:
    from gallops.drivejepa_perception import preflight

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a run never overwrites one")
    tokens, slot, rows, projections, targets = load(args.tokens, args.labels, args.tap, args.vit_tokens)
    train_rows = rows["tap"] if args.rows == "tap" else rows["train"]
    if args.train_fraction < 1.0:
        train_rows = fraction_of_logs(args.labels, train_rows, args.train_fraction)
    select_rows = rows["select"]
    stop = args.stop_after or args.epochs
    if not 1 <= stop <= args.epochs:
        raise SystemExit(f"--stop-after {args.stop_after} is outside 1..{args.epochs}")
    report = preflight.run(
        cfg,
        Path(args.tokens),
        Path(args.labels),
        {"train": train_rows, "select": select_rows, "report": rows["report"]},
        vit_dir=Path(args.vit_tokens) if args.tap == "vit_out" and args.vit_tokens else None,
    )
    subsets = importance_subsets(args.labels, projections, targets, select_rows)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    model = build_model(vars(args)).to(device)
    criterion = sidecar.SideCarLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_rows) // BATCH
    total = steps_per_epoch * args.epochs
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: lr_factor(s, total))
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
        "steps_per_epoch": steps_per_epoch,
        "schedule_steps": total,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=1, default=str))
    print(
        f"side-car {params / 1e6:.2f} M params; {len(train_rows)} train rows; {steps_per_epoch} steps/epoch; "
        f"{stop} of {args.epochs} epochs",
        flush=True,
    )
    t0, step, window = time.time(), 0, Window()
    for epoch in range(1, stop + 1):
        model.train()
        epoch_window, clipped, skipped = Window(), 0, 0
        waited = time.time()
        for item in loader:
            data_s = time.time() - waited
            item = {k: v.to(device, non_blocking=True) for k, v in item.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                outs = model(item["tokens"], item["projection"])
            loss, parts = criterion(outs, item)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch}, step {step + 1}")
            lr = optimiser.param_groups[0]["lr"]
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
            scale = scaler.get_scale()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            step += 1
            # An overflow makes GradScaler skip the step and halve its scale; it is not a clip.
            was_skipped = scaler.get_scale() < scale
            was_clipped = math.isfinite(grad_norm) and grad_norm > CLIP
            skipped, clipped = skipped + was_skipped, clipped + was_clipped
            values = {k: v for k, v in parts.items() if not k.startswith("_")}
            values.update(match_numbers(outs[-1], item, parts["_match"]))
            values.update(loss=float(loss), lr=lr, grad_norm=grad_norm, data_s=data_s)
            values["step_s"] = time.time() - waited
            window.add(values)
            epoch_window.add(values)
            if step % args.log_every == 0:
                line = {"step": step, "epoch": epoch, "amp_scale": scaler.get_scale(), **window.means()}
                with open(out / "steps.jsonl", "a") as fh:
                    fh.write(json.dumps(line) + "\n")
            waited = time.time()
        means = epoch_window.means()
        metrics = read(model, select_set, targets, select_rows, device, gap, subsets=subsets)
        record = {
            "epoch": epoch,
            "train_loss": means["loss"],
            "train": means,
            "steps": steps_per_epoch,
            "clipped_steps": clipped,
            "skipped_steps": skipped,
            "max_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
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
            f"precision {precision:.3f} f1 {record['f1_30m']:.3f} "
            f"causal {metrics['recall_causal_chosen']:.3f} "
            f"duplicates {metrics['duplicate_pairs_per_sample']:.2f} ({record['seconds']} s)",
            flush=True,
        )
    finish(model, out, args, tokens, slot, rows, projections, targets, gap, device)


def finish(model, out, args, tokens, slot, rows, projections, targets, gap, device) -> None:
    """The kept epoch (best select F1 within 30 m), then the report logs at it and at the last epoch."""
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    kept = max(records, key=lambda r: r["f1_30m"])["epoch"]
    last = records[-1]["epoch"]  # the schedule's last unless --stop-after ended the run early
    report_set = Rows(tokens, slot, rows["report"], projections, targets, blind=args.blind)
    select_set = Rows(tokens, slot, rows["select"], projections, targets, blind=args.blind)
    summary = {
        "kept_epoch": kept,
        "last_epoch": last,
        "schedule_epochs": args.epochs,
        "selection_rule": "best select F1 at 2 m within 30 m",
    }
    # A 1024-d tap exists only for A0's rows (the select logs and 10,000 train samples; tokens.py), so
    # an A0 probe on it is read on the select logs alone, as PREDICTIONS.md P1 says.
    has_report = args.tap == "post" or bool(getattr(args, "vit_tokens", ""))
    no_report = bool(getattr(args, "no_report", False))
    for name, epoch in (("kept", kept), ("last", last)) if has_report and not no_report else ():
        state = torch.load(out / f"epoch_{epoch:02d}.pt", map_location="cpu")["state_dict"]
        model.load_state_dict(state)
        summary[f"report_{name}"] = read(model, report_set, targets, rows["report"], device, gap)
    if not has_report:
        summary["report"] = "none: this tap has no report rows (A0 reads the select logs)"
    elif no_report:
        summary["report"] = "not read: --no-report (read once, later, after the run is reviewed on select)"
    model.load_state_dict(torch.load(out / f"epoch_{kept:02d}.pt", map_location="cpu")["state_dict"])
    summary["sight_select"] = {
        "with_tokens": read(model, select_set, targets, rows["select"], device, gap)["detections_per_sample"],
        "tokens_zeroed": read(model, select_set, targets, rows["select"], device, gap, blind=True)[
            "detections_per_sample"
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("kept_epoch", "sight_select")}), flush=True)


def reread(cfg, args) -> None:
    """A run's saved epochs read again with the current readout, beside the run's as-run files.

    For runs read with an earlier `evaluate.readout` ("within 30 m" ran to 35 m until 4 Nov). The
    epoch a run kept is not chosen again: `--which kept` reads the one its summary names. Writes
    `reread.json` in the run: per epoch, the select readout, and the report readout where the tap
    has report rows.
    """
    run = Path(args.run)
    target = run / "reread.json"
    if target.exists():
        raise SystemExit(f"{target} exists; a reread never overwrites one")
    meta = json.loads((run / "run.json").read_text())["args"]
    summary = json.loads((run / "summary.json").read_text())
    vit_tokens = meta.get("vit_tokens", "")
    tokens, slot, rows, projections, targets = load(args.tokens, args.labels, meta["tap"], vit_tokens)
    has_report = meta["tap"] == "post" or bool(vit_tokens)
    last = summary.get("last_epoch", meta["epochs"])
    epochs = {
        "kept": [summary["kept_epoch"]],
        "last": [last],
        "all": list(range(1, last + 1)),
    }
    device = torch.device("cuda")
    model = build_model(meta).to(device)
    train_rows = rows["tap"] if meta["rows"] == "tap" else rows["train"]
    gap = prior_gap(targets, train_rows)
    blind = bool(meta.get("blind"))
    out = {"evaluate_range_bins": [list(b) for b in evaluate.RANGE_BINS], "kept_epoch": summary["kept_epoch"]}
    from gallops.drivejepa_perception.b_report import clustered_se

    labels = np.load(Path(args.labels) / "labels.npz")
    logs = np.array(json.loads((Path(args.labels) / "manifest.json").read_text())["logs"])[labels["log_id"]]
    details = {}
    for epoch in epochs[args.which]:
        model.load_state_dict(torch.load(run / f"epoch_{epoch:02d}.pt", map_location="cpu")["state_dict"])
        entry = {}
        for split in ("select", "report") if has_report else ("select",):
            dataset = Rows(tokens, slot, rows[split], projections, targets, blind=blind)
            entry[split] = read(model, dataset, targets, rows[split], device, gap, blind=blind, detail=True)
            entry[f"{split}_f1_30m"] = f1(entry[split])
            d = entry[split].pop("detail")
            # P4: the 1 s path against constant velocity from the predicted velocity, moving agents,
            # paired per agent, SE clustered by log.
            moving = d["path_moving"]
            diff = (d["fde_model"] - d["fde_cv_pred"])[moving]
            clusters = logs[rows[split][d["path_row"][moving]]]
            entry[f"{split}_p4"] = {
                "agents": int(moving.sum()),
                "fde_model_moving": float(d["fde_model"][moving].mean()) if moving.any() else float("nan"),
                "fde_cv_pred_moving": float(d["fde_cv_pred"][moving].mean())
                if moving.any()
                else float("nan"),
                "model_minus_cv_pred": float(diff.mean()) if len(diff) else float("nan"),
                "se_by_log": clustered_se(diff, clusters) if len(diff) > 1 else float("nan"),
            }
            details.update({f"epoch_{epoch:02d}_{split}_{k}": v for k, v in d.items()})
        out[f"epoch_{epoch:02d}"] = entry
        print(
            json.dumps({"epoch": epoch, **{k: v for k, v in entry.items() if k.endswith("f1_30m")}}),
            flush=True,
        )
    target.write_text(json.dumps(out, indent=1))
    np.savez(run / "reread_detail.npz", **details)


def sight(cfg, args) -> None:
    """The sight check in distribution: each report row read with another log's real tokens.

    The pre-registered check zeroes the tokens, an input no run ever saw, and seed 0 answered it with
    27 detections a sample. Here every row keeps its own camera geometry and its own targets, but its
    tokens come from a row of another log (train_scorer.shuffled_rows, the S arm's draw). A side-car
    that sees should lose its recall: detections that still hit the row's agents were not made from
    its image. Writes `sight.json` in the run: the same readout with its own tokens and with another's.
    """
    from gallops.drivejepa_perception.train_scorer import shuffled_rows

    run = Path(args.run)
    target = run / "sight.json"
    if target.exists():
        raise SystemExit(f"{target} exists; a sight check never overwrites one")
    meta = json.loads((run / "run.json").read_text())
    run_args = meta["args"]
    tokens, slot, rows, projections, targets = load(
        args.tokens, args.labels, run_args["tap"], run_args.get("vit_tokens", "")
    )
    labels = np.load(Path(args.labels) / "labels.npz")
    logs = np.array(json.loads((Path(args.labels) / "manifest.json").read_text())["logs"])[labels["log_id"]]
    other = shuffled_rows(logs)
    model, kept = load_run(run)
    device = torch.device("cuda")
    model = model.to(device)
    out = {
        "kept_epoch": kept,
        "rows": int(len(rows["report"])),
        "other_log_share": float((logs[other] != logs).mean()),
    }
    for name, slot_of in (("own_tokens", slot), ("another_logs_tokens", lambda r: slot(int(other[r])))):
        dataset = Rows(tokens, slot_of, rows["report"], projections, targets)
        out[name] = read(model, dataset, targets, rows["report"], device, meta["prior_gap_m"])
        print(json.dumps({name: out[name]["recall_2m_within_30m"]}), flush=True)
    target.write_text(json.dumps(out, indent=1))


def overfit(cfg, args) -> None:
    """H04's check 1 on 64 train samples; exits 1 if it does not pass."""
    from gallops.drivejepa_perception import preflight

    tokens, slot, rows, projections, targets = load(args.tokens, args.labels, args.tap, args.vit_tokens)
    picked = np.random.default_rng(0).choice(rows["train"], 64, replace=False)
    preflight.run(
        cfg,
        Path(args.tokens),
        Path(args.labels),
        {"train": picked, "select": rows["select"][:64]},
        vit_dir=Path(args.vit_tokens) if args.tap == "vit_out" and args.vit_tokens else None,
    )
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = build_model(vars(args)).to(device)
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


def load_run(run: Path) -> tuple:
    """An A1 run's side-car at its kept epoch (on the CPU), and that epoch."""
    run = Path(run)
    run_args = json.loads((run / "run.json").read_text())["args"]
    epoch = json.loads((run / "summary.json").read_text())["kept_epoch"]
    model = build_model(run_args)
    model.load_state_dict(torch.load(run / f"epoch_{epoch:02d}.pt", map_location="cpu")["state_dict"])
    return model.eval(), epoch


@torch.no_grad()
def agent_arrays(model, tokens, slot, rows, projections, device, chunk: int = 256) -> dict:
    """The side-car's last-layer output on `rows`, decoded into the agents format phase B reads.

    The format arm O builds from the labels (train_scorer.truth_agents): class, confidence, centre,
    size, yaw, velocity, path offsets, and a mask of the slots at or over THR, so the scorer reads
    only agents a person would be shown (plan A3). Rows not in `rows` stay zero and masked.
    """
    model = model.to(device).eval()
    n, q = len(projections), model.query.shape[0]
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
    rows = np.asarray(rows)
    step = chunk
    with torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(rows), step):
            chunk = rows[i : i + step]
            # fp32 in, so the CPU runs it too; on the GPU autocast casts back to fp16 exactly.
            tok = (
                torch.from_numpy(
                    np.stack([np.asarray(tokens[slot(int(r))], dtype=np.float16) for r in chunk])
                )
                .to(device)
                .float()
            )
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
    return arrays


def row_set(spec: str, rows: dict, source: str) -> np.ndarray:
    """The rows an agents pass decodes. Off navtrain, `all`; on navtrain, `select`, `train` or
    `train+select`, never a report row (the report logs are read once, by Part A's step 6)."""
    if source != "navtrain":
        if spec != "all":
            raise SystemExit(f"a {source} pass decodes all its rows, not {spec!r}")
        return rows["all"]
    named = {"select": ["select"], "train": ["train"], "train+select": ["train", "select"]}
    if spec not in named:
        raise SystemExit(f"a navtrain pass decodes select, train or train+select, not {spec!r}")
    picked = np.sort(np.concatenate([rows[name] for name in named[spec]]))
    if np.isin(picked, rows["report"]).any():
        raise SystemExit("a report-log row is in the row set")
    return picked


def decoded(arrays: dict, row: int) -> dict:
    """One row of `agent_arrays`' output in `evaluate.decode`'s terms (absolute path points)."""
    xyz = arrays["xyz"][row]
    return {
        "conf": arrays["conf"][row],
        "cls": arrays["cls"][row],
        "xyz": xyz,
        "vel": arrays["vel"][row],
        "path": xyz[:, None, :2] + arrays["path"][row],
    }


def select_f1_check(model, run: Path, epoch: int, device) -> dict:
    """The run's kept epoch decoded by `agent_arrays` on its own select rows, read by `evaluate.readout`:
    the select F1 its training recorded for that epoch, within 1e-6 (B0-G7).

    The rows are decoded in batches of 128 in their order, as `predict` read them in training, so the
    two forwards see identical batches.
    """
    run_args = json.loads((run / "run.json").read_text())["args"]
    tokens, slot, rows, projections, targets = load(
        run_args["tokens"], run_args["labels"], run_args["tap"], run_args.get("vit_tokens", "")
    )
    select = rows["select"]
    arrays = agent_arrays(model, tokens, slot, select, projections, device, chunk=128)
    result = evaluate.readout(
        [decoded(arrays, r) for r in select], [targets[r] for r in select], prior_gap(targets, rows["train"])
    )
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    recorded = [r["f1_30m"] for r in records if r["epoch"] == epoch][0]
    now = f1(result)
    return {
        "select_rows": int(len(select)),
        "epoch": epoch,
        "recorded": recorded,
        "now": now,
        "gap": abs(now - recorded),
    }


def _manifest_sha(directory) -> str:
    import hashlib

    path = Path(directory) / "manifest.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def agents(cfg, args) -> None:
    """The agents pass: a run's kept epoch decoded on a row set, for phase B (module docstring)."""
    from gallops.drivejepa_b2d.provenance import git_sha

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    sha = git_sha()
    run = Path(args.run)
    meta = json.loads((run / "run.json").read_text())
    run_args = meta["args"]
    vit_tokens = args.vit_tokens or run_args.get("vit_tokens", "")
    device = torch.device("cuda")
    model, epoch = load_run(run)
    check = select_f1_check(model, run, epoch, device)
    print(f"select F1 check: {check}", flush=True)
    if check["gap"] > 1e-6:
        raise SystemExit(f"the run, decoded here, does not give the select F1 it recorded: {check}")
    tokens, slot, rows, projections, _ = load(args.tokens, args.labels, run_args["tap"], vit_tokens)
    source = navsim_data.source_of(json.loads((Path(args.labels) / "manifest.json").read_text()))
    picked = row_set(args.row_set, rows, source)
    arrays = agent_arrays(model, tokens, slot, picked, projections, device)
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    np.savez(staging / "agents.npz", **arrays)
    manifest = {
        "run": str(run),
        "epoch": epoch,
        "gallops_sha": sha,
        "train_git_sha": meta["train_git_sha"],
        "source": source,
        "row_set": args.row_set,
        "rows": len(projections),
        "rows_decoded": int(len(picked)),
        "inputs": {
            name: {"path": str(path), "manifest_sha256": _manifest_sha(path)}
            for name, path in (("tokens", args.tokens), ("labels", args.labels), ("vit_tokens", vit_tokens))
        },
        "select_f1_check": check,
        "threshold": evaluate.THR,
        "agents_per_row": float(arrays["mask"][picked].sum(1).mean()) if len(picked) else 0.0,
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
    model = build_model(vars(run_args))
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
    parser.add_argument("cmd", choices=("train", "overfit", "agents", "finish", "reread", "sight"))
    parser.add_argument("--run", default="", help="agents: the A1 run directory to decode")
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tap", choices=tuple(TAP_DIMS), default="post")
    parser.add_argument(
        "--variant", choices=tuple(VARIANTS), default=None, help="one of the plan's architectures"
    )
    parser.add_argument("--d-model", type=int, default=None, help="width (v1: 256)")
    parser.add_argument("--heads", type=int, default=None, help="attention heads (v1: 8)")
    parser.add_argument("--ffn", type=int, default=None, help="feed-forward width (v1: 1024)")
    parser.add_argument("--layers", type=int, default=None, help="decoder layers (v1: 3)")
    parser.add_argument("--queries", type=int, default=None, help="agent queries (v1: 64)")
    parser.add_argument("--epochs", type=int, default=12, help="the schedule's length")
    parser.add_argument(
        "--stop-after", type=int, default=0, help="stop after this many epochs of the --epochs schedule"
    )
    parser.add_argument("--no-report", action="store_true", help="read no report row (select only)")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY, help="steps per line of steps.jsonl")
    parser.add_argument("--rows", choices=("train", "tap"), default="train")
    parser.add_argument(
        "--train-fraction", type=float, default=1.0, help="A2: a seeded share of the train logs"
    )
    parser.add_argument("--blind", action="store_true")
    parser.add_argument(
        "--which", choices=("kept", "last", "all"), default="kept", help="reread: which epochs"
    )
    parser.add_argument(
        "--vit-tokens", default="", help="a ViT-out pass over every row (tokens --vit-out-all)"
    )
    parser.add_argument(
        "--row-set",
        default="all",
        help="agents: all (navtest, fresh) or select|train|train+select (navtrain)",
    )
    args = parser.parse_args(argv)
    resolve_architecture(args)
    cfg = SiteConfig.load()
    {
        "train": train,
        "overfit": overfit,
        "agents": agents,
        "finish": refinish,
        "reread": reread,
        "sight": sight,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
