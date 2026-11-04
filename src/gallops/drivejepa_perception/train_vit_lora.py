"""B3.1: does LoRA on the ViT's last four blocks, read by the side-car alone, beat the frozen tap?

    python -m gallops.drivejepa_perception.train_vit_lora train --block20 B --tokens T --labels L \
        --run {out}/runs/a1-s0 --arm lora --seeds 0,1,2 --out O
    python -m gallops.drivejepa_perception.train_vit_lora shared --block20 B --tokens T --labels L \
        --pdm P --from-run {out}/b3/lora --seed 0 --out O      (B3.2, reduced: `shared`)

PREDICTIONS.md P13 compares the LoRA branch with A1's side-car. Both arms here start from the same
A1 run's kept epoch (`--run`) and train for the same steps on the same rows, the block-20 cache's
train subset (tokens.py --block20, vit_lora.py). They differ only in the branch:
- `lora`: `vit_lora.LateBranch` with rank-16 LoRA (1.05 M), trained with the side-car;
- `control`: the frozen branch, so only the side-car trains: A1 continued on the same budget.
Against A1 alone, a gain could be A1's own extra training; against the control, it cannot.

The planner never runs and its tokens never change: the branch is the side-car's alone.

**Readouts** are A1's (train_sidecar.read): every epoch on the select logs, the kept epoch by F1 at
2 m within 30 m, then the report logs at the kept and the last epoch.

**Preflight** (`preflight.run_vit_lora`): code; the frozen branch on the cache reproduces the
ViT-out cache on 32 rows; and the A1 run read through the frozen branch on the select logs
reproduces its select F1 at that epoch, read with the same readout (its reread.json), within
preflight.VIT_LORA_F1_LIMIT.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gallops.drivejepa_perception import sidecar, train_sidecar, vit_lora

REPO = Path(__file__).resolve().parents[3]
LR_SIDECAR, LR_LORA, EPOCHS = 5e-5, 2e-4, 4
ARMS = ("lora", "control")


class Tail(nn.Module):
    """The branch, then the side-car: what the side-car reads in B3.1, from the block-20 cache."""

    def __init__(self, branch: nn.Module, car: nn.Module):
        super().__init__()
        self.branch, self.car = branch, car

    def forward(self, tokens, projection):
        return self.car(self.branch(tokens.float()), projection)


def release_vit(entered: dict) -> nn.Module:
    """The release's ViT, with the release's weights, on the CPU."""
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    return agent._pad_model._backbone.img_backbone.eval()


def a1_sidecar(run: Path) -> tuple:
    """The A1 run's side-car at its kept epoch, and that epoch's select F1 under the current readout.

    The F1 is the run's `reread.json` (train_sidecar reread), not its metrics.jsonl: the readout
    changed after A1 trained ("within 30 m" counted to 35 m), and the first B3 preflight, comparing
    the new readout's 0.669 with the old one's 0.618, refused a branch that was in fact exact.
    """
    meta = json.loads((run / "run.json").read_text())["args"]
    kept = json.loads((run / "summary.json").read_text())["kept_epoch"]
    car = sidecar.SideCar(
        d_in=train_sidecar.TAP_DIMS[meta["tap"]], layers=meta["layers"], queries=meta["queries"]
    )
    car.load_state_dict(torch.load(run / f"epoch_{kept:02d}.pt", map_location="cpu")["state_dict"])
    reread = run / "reread.json"
    if not reread.exists():
        raise SystemExit(f"{reread} is missing: read the A1 run again with the current readout first")
    f1 = json.loads(reread.read_text())[f"epoch_{kept:02d}"]["select_f1_30m"]
    return car, kept, f1, meta


def data(block20: Path, tokens: Path, labels: Path) -> dict:
    """The cache, its rows by split, and A1's projections and targets."""
    cache = np.load(Path(block20) / "block20.npy", mmap_mode="r")
    cache_rows = np.load(Path(block20) / "rows.npy")
    _, _, rows, projections, targets = train_sidecar.load(tokens, labels, "post")
    where = {int(r): i for i, r in enumerate(cache_rows)}
    held = {k: rows[k] for k in ("select", "report")}
    missing = {k: int((~np.isin(v, cache_rows)).sum()) for k, v in held.items()}
    if any(missing.values()):
        raise SystemExit(f"the block-20 cache lacks held-out rows: {missing}")
    return {
        "cache": cache,
        "slot": where.__getitem__,
        "rows": {"train": np.intersect1d(cache_rows, rows["train"]), **held},
        "projections": projections,
        "targets": targets,
    }


def train(cfg, args) -> None:
    from gallops.drivejepa_perception import env, preflight

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a run never overwrites one")
    entered = env.enter(cfg)
    d = data(Path(args.block20), Path(args.tokens), Path(args.labels))
    rows = d["rows"]
    vit = release_vit(entered)
    device = torch.device("cuda")
    car, kept_a1, f1_a1, a1_args = a1_sidecar(Path(args.run))
    report = preflight.run_vit_lora(cfg, args, d, vit, car, f1_a1)
    gap = train_sidecar.prior_gap(d["targets"], rows["train"])

    def rows_set(split):
        return train_sidecar.Rows(d["cache"], d["slot"], rows[split], d["projections"], d["targets"])

    out.mkdir(parents=True)
    summary = {
        "args": vars(args),
        "train_git_sha": report["code"]["sha"],
        "preflight": report,
        "a1": {"run": str(args.run), "kept_epoch": kept_a1, "select_f1_recorded": f1_a1},
        "rows": {k: int(len(v)) for k, v in rows.items()},
        "prior_gap_m": gap,
    }
    for seed in [int(s) for s in args.seeds.split(",")]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        car, _, _, _ = a1_sidecar(Path(args.run))
        model = Tail(vit_lora.LateBranch(vit, lora=args.arm == "lora"), car).to(device)
        lora = [p for n, p in model.branch.named_parameters() if p.requires_grad]
        groups = [{"params": list(model.car.parameters()), "lr": LR_SIDECAR}]
        if lora:
            groups.append({"params": lora, "lr": LR_LORA})
        trainable = sum(p.numel() for g in groups for p in g["params"])
        optimiser = torch.optim.AdamW(groups, weight_decay=train_sidecar.WEIGHT_DECAY)
        loader = torch.utils.data.DataLoader(
            rows_set("train"),
            batch_size=train_sidecar.BATCH,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            generator=torch.Generator().manual_seed(seed),
            pin_memory=True,
        )
        total = len(loader) * args.epochs
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimiser, lambda s, total=total: 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
        )
        scaler = torch.cuda.amp.GradScaler()
        criterion = sidecar.SideCarLoss()
        run_dir = out / f"seed{seed}"
        run_dir.mkdir()
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            model.branch.eval()  # no dropout or drop-path in the ViT's tail, trained or not
            for item in loader:
                item = {k: v.to(device, non_blocking=True) for k, v in item.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    outs = model(item["tokens"], item["projection"])
                loss, _ = criterion(outs, item)
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], train_sidecar.CLIP)
                scaler.step(optimiser)
                scaler.update()
                schedule.step()
                if not torch.isfinite(loss):
                    raise SystemExit(f"non-finite loss at epoch {epoch}")
            metrics = train_sidecar.read(model, rows_set("select"), d["targets"], rows["select"], device, gap)
            record = {"epoch": epoch, "select": metrics, "f1_30m": train_sidecar.f1(metrics)}
            record["seconds"] = round(time.time() - t0)
            with open(run_dir / "metrics.jsonl", "a") as fh:
                fh.write(json.dumps(record) + "\n")
            state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            torch.save({"trained": state, "epoch": epoch}, run_dir / f"epoch_{epoch:02d}.pt")
            print(
                f"{args.arm} seed {seed} epoch {epoch}: select recall {metrics['recall_2m_within_30m']:.3f} "
                f"f1 {record['f1_30m']:.3f} ({record['seconds']} s)",
                flush=True,
            )
        records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
        kept = max(records, key=lambda r: r["f1_30m"])["epoch"]
        result = {"kept_epoch": kept, "trainable_params": trainable}
        for name, epoch in (("kept", kept), ("last", args.epochs)):
            state = torch.load(run_dir / f"epoch_{epoch:02d}.pt", map_location="cpu")["trained"]
            model.load_state_dict(state, strict=False)
            result[f"report_{name}"] = train_sidecar.read(
                model, rows_set("report"), d["targets"], rows["report"], device, gap
            )
        summary[f"seed{seed}"] = result
        print(
            json.dumps(
                {"seed": seed, "kept": kept, "report_kept_f1": train_sidecar.f1(result["report_kept"])}
            )
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


@torch.no_grad()
def shared(cfg, args) -> None:
    """B3.2, reduced: B3.1's LoRA put in the trunk the planner reads, with the planner unchanged.

    The plan's B3.2 retrains the planner on the adapted tokens. This asks the part that costs no
    training: does the perception-tuned tail move the release's plan? On the select and report logs,
    the release's projector, refiner and scorer run on the branch's output, twice:
    - with the frozen branch, which must reproduce the release's cached proposals and picks within
      preflight.REFINER_LIMITS (B2's), or nothing is written;
    - with `--from`'s LoRA (seed `--seed`, its kept epoch).
    Both sets of candidates are PDM-scored offline, and the report logs' paired difference in the
    pick's EPDMS is given with its log-clustered SE.
    """
    from multiprocessing import get_context

    from gallops.drivejepa_b2d import batch_fix
    from gallops.drivejepa_b2d.head import Head, head_tensors_from_checkpoint
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_b2d.train_generator import refine_from_cache
    from gallops.drivejepa_perception import b_report, env, pdm_labels, preflight, train_refiner, train_scorer
    from gallops.train.preflight import code_checks

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a readout never overwrites one")
    code, problems = code_checks(REPO)
    if problems:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(problems))
    entered = env.enter(cfg)
    data = train_scorer.load(Path(args.tokens), Path(args.labels), Path(args.pdm), "C", None)
    cache = np.load(Path(args.block20) / "block20.npy", mmap_mode="r")
    where = {int(r): i for i, r in enumerate(np.load(Path(args.block20) / "rows.npy"))}
    lidar2img = np.load(Path(args.tokens) / "lidar2img.npy")
    img_shape = np.load(Path(args.tokens) / "img_shape.npy")
    spatial, level = torch.tensor([[16, 32]], dtype=torch.long), torch.tensor([0], dtype=torch.long)
    batch_fix.apply_all()
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.eval()
    vit = model._backbone.img_backbone
    projector = model._backbone.projector.cuda()
    model._backbone = model._backbone.cpu()
    model = model.cuda()
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    head = head.cuda().eval()
    run = Path(args.from_run)
    kept = json.loads((run / "summary.json").read_text())[f"seed{args.seed}"]["kept_epoch"]
    trained = torch.load(run / f"seed{args.seed}" / f"epoch_{kept:02d}.pt", map_location="cpu")["trained"]
    lora_state = {k[len("branch.") :]: v for k, v in trained.items() if k.startswith("branch.")}
    labels = np.load(Path(args.labels) / "labels.npz")
    root = Path(cfg.resolve("{data.navsim_metric_cache}"))
    result = {"args": vars(args), "gallops_sha": git_sha(), "code": code, "from_kept_epoch": kept}
    picks = {}
    for variant in ("frozen", "lora"):
        branch = vit_lora.LateBranch(vit, lora=variant == "lora")
        if variant == "lora":
            missing = branch.load_state_dict(lora_state, strict=False)
            if missing.unexpected_keys or not lora_state:
                raise SystemExit(f"the B3.1 run's LoRA does not fit the branch: {missing.unexpected_keys}")
        branch = branch.cuda().eval()
        for split in ("select", "report"):
            rows = data["rows"][split]
            proposals = np.zeros((len(rows), 32, 8, 3), np.float32)
            pick = np.zeros(len(rows), np.int64)
            for i in range(0, len(rows), train_refiner.BATCH):
                chunk = rows[i : i + train_refiner.BATCH]
                x = torch.from_numpy(np.stack([np.asarray(cache[where[int(r)]]) for r in chunk])).cuda()
                feat = projector(branch(x.float()))
                props, bev = refine_from_cache(
                    model,
                    feat,
                    torch.from_numpy(lidar2img[chunk][:, 1:2]),
                    torch.from_numpy(img_shape[chunk]),
                    torch.from_numpy(data["ego"][chunk]),
                    spatial,
                    level,
                )
                proposals[i : i + len(chunk)] = props[-1].float().cpu().numpy()
                pooled = bev.reshape(len(chunk), 32, 8, -1).amax(2)
                pick[i : i + len(chunk)] = head(pooled)[..., -1].argmax(1).cpu().numpy()
            if variant == "frozen":
                gap = float(np.abs(proposals - data["proposals"][rows]).max())
                share = float((pick == data["pick_release"][rows]).mean())
                result[f"identity_{split}"] = {"proposals_max_m": gap, "picks_equal_share": share}
                limits = preflight.REFINER_LIMITS
                if gap > limits["proposals_m"] or share < limits["picks_share"]:
                    raise SystemExit(f"the frozen branch does not reproduce the release: {result}")
            jobs = [
                (
                    pdm_labels.metric_cache_path(root, data["token_logs"][r], str(labels["token"][r])),
                    proposals[j],
                )
                for j, r in enumerate(rows)
            ]
            with get_context("fork").Pool(48) as pool:
                scored = np.stack(pool.starmap(train_refiner._score_only, jobs, chunksize=8))
            human = data["human"][rows]
            chosen = scored[np.arange(len(rows)), pick]
            result[f"{variant}_{split}"] = {
                "pick_epdms": float(chosen[:, -1].mean()),
                "best_of_32_epdms": float(scored[:, :, -1].max(1).mean()),
                "pick_terms": dict(
                    zip(  # noqa: B905 -- Python 3.9
                        ("nc", "dac", "tlc", "ddc", "ep", "ttc", "lk", "hc"),
                        chosen[:, :8].mean(0).round(4).tolist(),
                    )
                ),
                "l2_3s": float(
                    np.linalg.norm(
                        proposals[np.arange(len(rows)), pick, 5, :2] - human[:, 5, :2], axis=-1
                    ).mean()
                ),
            }
            if split == "report":
                picks[variant] = chosen[:, -1]
        print(json.dumps({k: v for k, v in result.items() if k.startswith(variant)}), flush=True)
    diff = picks["lora"] - picks["frozen"]
    logs = data["token_logs"][data["rows"]["report"]]
    result["report_lora_minus_frozen"] = {"mean": float(diff.mean()), "se": b_report.clustered_se(diff, logs)}
    out.mkdir(parents=True)
    (out / "shared.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result["report_lora_minus_frozen"]))


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.train_vit_lora")
    parser.add_argument("command", choices=("train", "shared"))
    for flag in ("--block20", "--tokens", "--labels", "--out"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--run", default="", help="train: the A1 run both arms start from")
    parser.add_argument("--arm", choices=ARMS, default="lora")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--pdm", default="", help="shared: the PDM labels pass")
    parser.add_argument("--from-run", default="", help="shared: the B3.1 lora run whose LoRA is shared")
    parser.add_argument("--seed", type=int, default=0, help="shared: which seed of --from-run")
    args = parser.parse_args(argv)
    if args.command == "train":
        if not args.run:
            parser.error("train needs --run (the A1 run)")
        train(SiteConfig.load(), args)
    else:
        if not (args.pdm and args.from_run):
            parser.error("shared needs --pdm and --from-run")
        shared(SiteConfig.load(), args)


if __name__ == "__main__":
    main()
