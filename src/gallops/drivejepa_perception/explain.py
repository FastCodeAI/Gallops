"""B1.6: are the scorer's reasons faithful? (PREDICTIONS.md P11)

    python -m gallops.drivejepa_perception.explain --run {out}/b1/<mode>-F --seed 0 --tokens T --labels L \
        --pdm P --agents A --out {out}/b1/explain.json

**The named agent.** For each report row, the explanation names the agent the trained scorer
attended to most when it scored its pick: the candidate -> agent attention of the agent half,
averaged over heads. It is read from the model, not chosen after the fact.

**Deletion fidelity.** Delete the named agent from the scorer's input and score again; separately,
delete one random other agent the side-car showed. Compared, over rows with at least two agents:
- how often the pick changes;
- how much the margin between the pick and the runner-up shrinks.

A faithful explanation names agents whose deletion matters more than a random one's (P11: ratio
>= 3).

**Grounding.** Is the named agent real? Within 2 m (centre, x-y) of a labelled agent of its class
in the camera's view (P11: >= 85%).

**Calibration.** The AUC of the scorer's own TTC output (`frozen`, `lora`, `full`: head column TTC)
or NC output (`replace`) against the PDM labels, on every report candidate (P11: >= 0.75).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gallops.drivejepa_perception import navsim_data, scorers
from gallops.drivejepa_perception import targets as tg
from gallops.drivejepa_perception.train_scorer import batch_of, load

GROUNDING_M = 2.0
#: Head column of TTC among the release head's six (scorers.HEAD_COLUMNS: DDC, EP, TTC, LK, HC, final).
HEAD_TTC = 2


@torch.no_grad()
def run(args) -> dict:
    from gallops.config import SiteConfig
    from gallops.drivejepa_b2d.head import Head, auc, head_tensors_from_checkpoint
    from gallops.drivejepa_perception import env

    entered = env.enter(SiteConfig.load())
    data = load(Path(args.tokens), Path(args.labels), Path(args.pdm), "F", args.agents)
    run_args = json.loads((Path(args.run) / "summary.json").read_text())["args"]
    kept = json.loads((Path(args.run) / "summary.json").read_text())[f"seed{args.seed}"]["kept_epoch"]
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    model = scorers.B1Model(run_args["mode"], True, head)
    state = torch.load(Path(args.run) / f"seed{args.seed}" / f"epoch_{kept:02d}.pt", map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    device = torch.device("cuda")
    model = model.to(device).eval()
    labels = dict(np.load(Path(args.labels) / "labels.npz"))
    projections = tg.front_projection(np.load(Path(args.tokens) / "lidar2img.npy"))
    rows = data["rows"]["report"]
    rng = np.random.default_rng(0)
    out = {"pick_changes": {"named": 0, "random": 0}, "margin_drop": {"named": [], "random": []}}
    grounded, considered, scores_all, truth_all = 0, 0, [], []
    for i in range(0, len(rows), 256):
        chunk = rows[i : i + 256]
        batch = batch_of(data, chunk, device)
        result = model(batch)
        score = result["score"]
        pick = score.argmax(1)
        top2 = score.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        attention = result["attention"]  # B, K, A
        if "logits" in result:
            scores_all.append(torch.sigmoid(result["logits"][..., HEAD_TTC]).cpu().numpy().ravel())
            truth_all.append(batch["pdm"][..., 5].cpu().numpy().ravel() >= 0.5)  # TTC label
        else:
            scores_all.append(result["sub"][..., 0].cpu().numpy().ravel())
            truth_all.append(batch["pdm"][..., 0].cpu().numpy().ravel() >= 0.5)  # NC label
        mask = batch["agents"]["mask"]
        weights = attention[torch.arange(len(chunk)), pick].masked_fill(~mask, -1.0)
        named = weights.argmax(1)
        for name, choose in (("named", lambda b, named=named: int(named[b])), ("random", None)):
            deleted = {k: v.clone() for k, v in batch["agents"].items()}
            keep = torch.ones(len(chunk), dtype=torch.bool, device=device)
            for b in range(len(chunk)):
                shown = torch.nonzero(mask[b]).flatten().tolist()
                if len(shown) < 2:
                    keep[b] = False
                    continue
                if choose is None:
                    others = [a for a in shown if a != int(named[b])]
                    victim = int(rng.choice(others))
                else:
                    victim = choose(b)
                deleted["mask"][b, victim] = False
            again = model({**batch, "agents": deleted})["score"]
            changed = (again.argmax(1) != pick) & keep
            out["pick_changes"][name] += int(changed.sum())
            new_margin = (
                again[torch.arange(len(chunk)), pick]
                - again.masked_fill(torch.nn.functional.one_hot(pick, again.shape[1]).bool(), -1e9)
                .max(1)
                .values
            )
            out["margin_drop"][name] += (margin - new_margin)[keep].cpu().numpy().tolist()
        for b, row in enumerate(chunk):
            if mask[b].sum() < 2:
                continue
            considered += 1
            a = int(named[b])
            t = tg.build(navsim_data.agents_of(labels, int(row)), projections[int(row)])
            gxy, gcls = t["box"][t["mask"], :2], t["cls"][t["mask"]]
            xy = batch["agents"]["xyz"][b, a, :2].cpu().numpy()
            same = gcls == int(batch["agents"]["cls"][b, a])
            if same.any() and np.hypot(*(gxy[same] - xy).T).min() <= GROUNDING_M:
                grounded += 1
    report = {
        "rows_with_two_or_more_agents": considered,
        "pick_change_rate_named": out["pick_changes"]["named"] / max(considered, 1),
        "pick_change_rate_random": out["pick_changes"]["random"] / max(considered, 1),
        "mean_margin_drop_named": float(np.mean(out["margin_drop"]["named"])) if considered else float("nan"),
        "mean_margin_drop_random": float(np.mean(out["margin_drop"]["random"]))
        if considered
        else float("nan"),
        "grounded_share": grounded / max(considered, 1),
        "calibration_auc": auc(np.concatenate(scores_all), np.concatenate(truth_all)),
        "calibrated_output": "TTC head" if run_args["mode"] != "replace" else "NC head",
        "run": str(args.run),
        "seed": args.seed,
        "kept_epoch": kept,
    }
    named_rate = report["pick_change_rate_named"]
    random_rate = report["pick_change_rate_random"]
    report["pick_change_ratio"] = named_rate / random_rate if random_rate else float("inf")
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report


@torch.no_grad()
def calibration(args) -> dict:
    """The calibration part alone, for any arm: C has the same trained sub-score outputs as F.

    B1.6's AUC is F's; whether the agents improve it needs C's beside it. Every seed at its kept
    epoch, every report candidate: the AUC of the TTC output (residual modes) or the NC output
    (replace) against the PDM label, as `run` computes it.
    """
    from gallops.config import SiteConfig
    from gallops.drivejepa_b2d.head import Head, auc, head_tensors_from_checkpoint
    from gallops.drivejepa_perception import env

    entered = env.enter(SiteConfig.load())
    summary = json.loads((Path(args.run) / "summary.json").read_text())
    run_args = summary["args"]
    arm = run_args["arm"]
    data = load(Path(args.tokens), Path(args.labels), Path(args.pdm), arm, run_args.get("agents") or None)
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    device = torch.device("cuda")
    rows = data["rows"]["report"]
    out = {"run": str(args.run), "arm": arm, "mode": run_args["mode"]}
    for key in sorted(k for k in summary if k.startswith("seed")):
        kept = summary[key]["kept_epoch"]
        model = scorers.B1Model(run_args["mode"], arm != "C", head)
        state = torch.load(Path(args.run) / key / f"epoch_{kept:02d}.pt", map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)
        model = model.to(device).eval()
        scores_all, truth_all = [], []
        for i in range(0, len(rows), 256):
            batch = batch_of(data, rows[i : i + 256], device)
            result = model(batch)
            if "logits" in result:
                scores_all.append(torch.sigmoid(result["logits"][..., HEAD_TTC]).cpu().numpy().ravel())
                truth_all.append(batch["pdm"][..., 5].cpu().numpy().ravel() >= 0.5)
            else:
                scores_all.append(result["sub"][..., 0].cpu().numpy().ravel())
                truth_all.append(batch["pdm"][..., 0].cpu().numpy().ravel() >= 0.5)
        out[key] = {"kept_epoch": kept, "auc": auc(np.concatenate(scores_all), np.concatenate(truth_all))}
        print(json.dumps({key: out[key]}), flush=True)
    out["calibrated_output"] = "TTC head" if run_args["mode"] != "replace" else "NC head"
    Path(args.out).write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.explain")
    parser.add_argument("--calibration-only", action="store_true", help="any arm: the AUC part alone")
    for flag in ("--run", "--tokens", "--labels", "--pdm", "--out"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--agents", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.calibration_only:
        calibration(args)
        return
    if not args.agents:
        parser.error("the deletion test reads the side-car's agents: pass --agents")
    run(args)


if __name__ == "__main__":
    main()
