"""B1: train one scorer mode on one arm, every seed, and read the pick (PREDICTIONS.md P7-P11).

    python -m gallops.drivejepa_perception.train_scorer --tokens T --labels L --pdm P --mode frozen --arm C \
        --loss best --seeds 0,1,2 --out O [--agents AGENTS_DIR]
    python -m gallops.drivejepa_perception.train_scorer --tokens T --labels L --pdm P --choose-loss --out O

Everything is cached:
- the release's pooled scorer input, candidates, logits and picks (tokens pass);
- every candidate's PDM labels (pdm pass);
- the agents an arm reads: the side-car's decoded agents for F, the same agents from another log for
  S, and the labels' agents in the camera's view for O. C reads none.

The ViT and the refiner never run. One process trains every seed of one (mode, arm) after one
preflight (`preflight.run_scorer`).

**Readouts.** Each epoch is read on the select logs, which choose the kept epoch by the pick's mean
EPDMS. After training, the report logs are read at the kept epoch and at the last one:
- the pick's EPDMS (the final PDM score of the chosen candidate) and its terms;
- best-of-32, and the gap between it and the pick;
- L2 to the human at 1, 2 and 3 s;
- the same on the three pre-registered subsets (PREDICTIONS.md).

Per-row picks are saved, so arms can be compared with paired, log-clustered SEs (`b1_report`). The
release (R) is read on the same rows from its cached picks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from gallops.drivejepa_perception import navsim_data, scorers
from gallops.drivejepa_perception import targets as tg

REPO = Path(__file__).resolve().parents[3]
ARMS = ("C", "F", "O", "S")
BATCH, EPOCHS = 256, 10
LR, LR_HEAD = 1e-4, 2e-5
L2_STEPS = {"1s": 1, "2s": 3, "3s": 5}
S_SEED = 7


def load(tokens_dir: Path, labels_dir: Path, pdm_dir: Path, arm: str, agents_dir) -> dict:
    """Every array an arm reads, in the labels' row order, plus the splits' rows and the subsets."""
    labels = dict(np.load(Path(labels_dir) / "labels.npz"))
    logs = json.loads((Path(labels_dir) / "manifest.json").read_text())["logs"]
    token_logs = np.array([logs[i] for i in labels["log_id"]])
    splits = json.loads((REPO / "recipes/drivejepa_perception/splits.json").read_text())
    pdm = np.load(Path(pdm_dir) / "pdm.npz")["candidates"]
    scored = np.isfinite(pdm[:, 0, -1])
    rows = {k: np.flatnonzero(np.isin(token_logs, splits[k]) & scored) for k in ("train", "select", "report")}
    proposals = np.load(Path(tokens_dir) / "proposals.npy")
    human = labels["ego_future"].astype(np.float32)
    data = {
        "token_logs": token_logs,
        "rows": rows,
        "pooled": np.load(Path(tokens_dir) / "pooled.npy"),  # 1.7 GB fp16: in RAM
        "proposals": proposals,
        "logits": np.load(Path(tokens_dir) / "logits.npy"),
        "pick_release": np.load(Path(tokens_dir) / "pick.npy").astype(np.int64),
        "ego": np.load(Path(tokens_dir) / "ego_status.npy")[:, -1].astype(np.float32),
        "pdm": np.nan_to_num(pdm, nan=0.0),
        "human": human,
        "ade": np.linalg.norm(proposals[..., :2] - human[:, None, :, :2], axis=-1)
        .mean(-1)
        .astype(np.float32),
        "shape": shape_of(proposals),
        "subsets": subsets(labels, proposals),
    }
    if arm in ("F", "S"):
        stored = np.load(Path(agents_dir) / "agents.npz")
        data["agents"] = {k: stored[k] for k in stored.files}
        if arm == "S":
            data["agents"] = {k: v[shuffled_rows(token_logs)] for k, v in data["agents"].items()}
    elif arm == "O":
        data["agents"] = truth_agents(labels, np.load(Path(tokens_dir) / "lidar2img.npy"))
    return data


def shape_of(proposals: np.ndarray) -> np.ndarray:
    """(N, 32, 8, 3) candidates -> (N, 32, 24) the scorer adapter's view of their shape, scaled."""
    return (
        np.concatenate([proposals[..., 0:1] / 40.0, proposals[..., 1:2] / 10.0, proposals[..., 2:3]], axis=-1)
        .reshape(len(proposals), 32, 24)
        .astype(np.float32)
    )


def truth_agents(labels: dict, lidar2img: np.ndarray) -> dict:
    """Arm O: the labels' agents the camera sees, in the side-car's decoded format, confidence 1."""
    return truth_agents_projected(labels, tg.front_projection(lidar2img))


def truth_agents_projected(labels: dict, projections: np.ndarray) -> dict:
    """`truth_agents` through given token-image projections (nuScenes' differ from NAVSIM's). The rule
    lives in `targets.truth_agents` (numpy); arm O keeps the first pass's zero fill and its eight keys."""
    out = tg.truth_agents(labels, projections, fill="zero")
    out.pop("path_ok")
    return out


def shuffled_rows(token_logs: np.ndarray) -> np.ndarray:
    """Arm S: every row reads the agents of a row from another log (seeded, recorded as S_SEED)."""
    rng = np.random.default_rng(S_SEED)
    perm = rng.permutation(len(token_logs))
    for _ in range(10):
        same = np.flatnonzero(token_logs[perm] == token_logs)
        if not len(same):
            break
        perm[same] = perm[rng.permutation(same)]
    return perm


#: The three pre-registered subsets, now numpy-only in navsim_data (tested without torch).
subsets = navsim_data.subsets


def batch_of(data: dict, rows: np.ndarray, device) -> dict:
    out = {
        "pooled": torch.from_numpy(data["pooled"][rows]),
        "shape": torch.from_numpy(data["shape"][rows]),
        "ego": torch.from_numpy(data["ego"][rows]),
        "logits": torch.from_numpy(data["logits"][rows]),
        "candidates": torch.from_numpy(data["proposals"][rows]),
        "pdm": torch.from_numpy(data["pdm"][rows]),
        "ade": torch.from_numpy(data["ade"][rows]),
    }
    out = {k: v.to(device) for k, v in out.items()}
    if "agents" in data:
        out["agents"] = {
            k: torch.from_numpy(np.asarray(v[rows])).to(device) for k, v in data["agents"].items()
        }
        out["agents"]["mask"] = out["agents"]["mask"].bool()
    return out


@torch.no_grad()
def scores(model, data: dict, rows: np.ndarray, device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(rows), 1024):
        out.append(model(batch_of(data, rows[i : i + 1024], device))["score"].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 32))


def readout(pick: np.ndarray, data: dict, rows: np.ndarray) -> dict:
    """The plan the scorer picks, on `rows`, overall and on each subset."""
    pdm, proposals, human = data["pdm"][rows], data["proposals"][rows], data["human"][rows]
    chosen = pdm[np.arange(len(rows)), pick]
    best = pdm[:, :, -1].max(-1)
    out = {
        "n": int(len(rows)),
        "pick_epdms": float(chosen[:, -1].mean()),
        "best_of_32_epdms": float(best.mean()),
        "gap_to_best": float((best - chosen[:, -1]).mean()),
        "pick_terms": dict(
            zip(  # noqa: B905 -- Python 3.9
                ("nc", "dac", "tlc", "ddc", "ep", "ttc", "lk", "hc"), chosen[:, :8].mean(0).round(4).tolist()
            )
        ),
    }
    for name, step in L2_STEPS.items():
        poses = proposals[np.arange(len(rows)), pick, step, :2]
        out[f"l2_{name}"] = float(np.linalg.norm(poses - human[:, step, :2], axis=-1).mean())
    for name, mask in data["subsets"].items():
        m = mask[rows]
        out[f"pick_epdms_{name}"] = float(chosen[m, -1].mean()) if m.any() else float("nan")
        out[f"n_{name}"] = int(m.sum())
    return out


def _fit(model, data: dict, seed: int, epochs: int, kind: str, run_dir: Path, device) -> tuple:
    """Train one seed with loss `kind`, reading the select logs and saving the weights every epoch.

    Returns the epoch records and the number of trained parameters.
    """
    rows = data["rows"]
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")],
            "lr": LR_HEAD,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")
            ],
            "lr": LR,
        },
    ]
    trainable = sum(p.numel() for g in groups for p in g["params"])
    optimiser = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.01)
    steps = (len(rows["train"]) // BATCH) * epochs
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s, steps=steps: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    rng = np.random.default_rng(seed)
    run_dir.mkdir(parents=True)
    t0, records = time.time(), []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(rows["train"])
        for i in range(0, len(order) - BATCH + 1, BATCH):
            batch = batch_of(data, order[i : i + BATCH], device)
            total, _ = scorers.loss(model(batch), batch, kind)
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimiser.step()
            schedule.step()
        pick = scores(model, data, rows["select"], device).argmax(1)
        record = {
            "epoch": epoch,
            "select": readout(pick, data, rows["select"]),
            "seconds": round(time.time() - t0),
        }
        records.append(record)
        with open(run_dir / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
        state = {k: v for k, v in model.state_dict().items() if not k.startswith("reference.")}
        torch.save({"state_dict": state, "epoch": epoch}, run_dir / f"epoch_{epoch:02d}.pt")
        print(
            f"{run_dir.name} epoch {epoch}: select pick EPDMS {record['select']['pick_epdms']:.4f}",
            flush=True,
        )
    return records, trainable


def train(cfg, args) -> None:
    from gallops.drivejepa_b2d.head import Head, head_tensors_from_checkpoint
    from gallops.drivejepa_perception import env, preflight

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a run never overwrites one")
    data = load(Path(args.tokens), Path(args.labels), Path(args.pdm), args.arm, args.agents)
    device = torch.device("cuda")
    entered = env.enter(cfg)
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    report = preflight.run_scorer(cfg, args, data, lambda: scorers.B1Model(args.mode, args.arm != "C", head))
    out.mkdir(parents=True)
    rows = data["rows"]
    summary = {
        "args": vars(args),
        "train_git_sha": report["code"]["sha"],
        "preflight": report,
        "rows": {k: int(len(v)) for k, v in rows.items()},
        "release": {s: readout(data["pick_release"][rows[s]], data, rows[s]) for s in ("select", "report")},
    }
    picks = {"rows": rows["report"], "release": data["pick_release"][rows["report"]]}
    picks.update({f"subset_{k}": v[rows["report"]] for k, v in data["subsets"].items()})
    for seed in [int(s) for s in args.seeds.split(",")]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = scorers.B1Model(args.mode, args.arm != "C", head).to(device)
        records, trainable = _fit(model, data, seed, args.epochs, args.loss, out / f"seed{seed}", device)
        run_dir = out / f"seed{seed}"
        kept = max(records, key=lambda r: r["select"]["pick_epdms"])["epoch"]
        result = {"kept_epoch": kept, "trainable_params": trainable}
        for name, epoch in (("kept", kept), ("last", args.epochs)):
            state = torch.load(run_dir / f"epoch_{epoch:02d}.pt", map_location="cpu")["state_dict"]
            model.load_state_dict(state, strict=False)
            pick = scores(model, data, rows["report"], device).argmax(1)
            result[f"report_{name}"] = readout(pick, data, rows["report"])
            picks[f"seed{seed}_{name}"] = pick
        summary[f"seed{seed}"] = result
        print(
            json.dumps({"seed": seed, "kept": kept, "report_kept": result["report_kept"]["pick_epdms"]}),
            flush=True,
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    np.savez(out / "report_picks.npz", **picks)


def choose_loss(cfg, args) -> None:
    """Choose B1's loss (scorers.LOSSES) on the select logs, C arm only, before any arm reads an agent.

    Seed 0 of every mode trains with each loss. A loss scores the mean, over the four modes, of the
    select pick EPDMS at each run's kept epoch; the highest is chosen. The report logs are not read.
    """
    from gallops.drivejepa_b2d.head import Head, head_tensors_from_checkpoint
    from gallops.drivejepa_perception import env, preflight

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a run never overwrites one")
    data = load(Path(args.tokens), Path(args.labels), Path(args.pdm), "C", None)
    device = torch.device("cuda")
    entered = env.enter(cfg)
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    rows = data["rows"]
    result = {
        "args": vars(args),
        "rule": "max over losses of the mean over modes of seed 0's select pick EPDMS at its kept epoch",
        "release_select_pick_epdms": readout(data["pick_release"][rows["select"]], data, rows["select"])[
            "pick_epdms"
        ],
        "preflight": {},
        "select": {},
    }
    # Every mode's preflight before any training: the code check then names the commit that runs
    # all twelve, whatever is edited while they train (a mid-run check refused one, 913b671).
    for mode in scorers.MODES:
        mode_args = argparse.Namespace(**{**vars(args), "mode": mode, "arm": "C"})
        result["preflight"][mode] = preflight.run_scorer(
            cfg, mode_args, data, lambda mode=mode: scorers.B1Model(mode, False, head)
        )
    out.mkdir(parents=True)
    for mode in scorers.MODES:
        for kind in scorers.LOSSES:
            torch.manual_seed(0)
            np.random.seed(0)
            model = scorers.B1Model(mode, False, head).to(device)
            records, _ = _fit(model, data, 0, args.epochs, kind, out / kind / f"{mode}-C", device)
            kept = max(records, key=lambda r: r["select"]["pick_epdms"])
            result["select"].setdefault(kind, {})[mode] = {
                "kept_epoch": kept["epoch"],
                "pick_epdms": kept["select"]["pick_epdms"],
                "l2_1s": kept["select"]["l2_1s"],
                "pick_terms": kept["select"]["pick_terms"],
            }
    result["mean_over_modes"] = {
        kind: float(np.mean([v["pick_epdms"] for v in result["select"][kind].values()]))
        for kind in scorers.LOSSES
    }
    result["chosen"] = max(result["mean_over_modes"], key=result["mean_over_modes"].get)
    (out / "choice.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: result[k] for k in ("mean_over_modes", "chosen", "release_select_pick_epdms")}))


def release_only(args) -> None:
    """S0.8: the release's own picks read on the select and report logs; nothing trains."""
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a readout never overwrites one")
    data = load(Path(args.tokens), Path(args.labels), Path(args.pdm), "C", None)
    rows = data["rows"]
    result = {s: readout(data["pick_release"][rows[s]], data, rows[s]) for s in ("select", "report")}
    result["note"] = "in-sample for the release: it trained on all of navtrain"
    out.mkdir(parents=True)
    (out / "release.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.train_scorer")
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pdm", required=True)
    parser.add_argument("--agents", default="")
    parser.add_argument("--mode", choices=scorers.MODES, default="frozen")
    parser.add_argument("--arm", choices=ARMS, default="C")
    parser.add_argument("--release-only", action="store_true", help="S0.8: read the release's own picks")
    parser.add_argument("--choose-loss", action="store_true", help="choose the loss: select logs, C arm")
    parser.add_argument("--loss", choices=scorers.LOSSES, default="", help="required to train")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.release_only:
        release_only(args)
        return
    if args.choose_loss:
        choose_loss(SiteConfig.load(), args)
        return
    if not args.loss:
        parser.error("training needs --loss (scorers.LOSSES; choose it with --choose-loss)")
    if args.arm in ("F", "S") and not args.agents:
        parser.error("arms F and S read the side-car's agents: pass --agents")
    train(SiteConfig.load(), args)


if __name__ == "__main__":
    main()
