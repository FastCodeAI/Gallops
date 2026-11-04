"""B2: give the refiner the side-car's agents, in three modes, against the arm that withholds them.

    python -m gallops.drivejepa_perception.train_refiner --tokens T --labels L --mode frozen --arm C \
        --seeds 0,1,2 --out O [--agents AGENTS_DIR]

The refiner runs from cached tokens (`drivejepa_b2d.train_generator.refine_from_cache`), batched with
`batch_fix`.

**The adapter.** A forward pre-hook on the one shared `Bev_refiner` adds to its input queries on
every pass. The pass-k proposals it is handed give the clearances.
- The own half (MLP over the query and its proposal pose) runs in every arm.
- F, O and S add the agent half: query -> agent attention, biased by the analytic clearances to that
  query's candidate (geometry.py).
- Its output layers start at zero.

**Modes:**
- `frozen` (B2.1): only the adapter trains.
- `lora` (B2.2): plus rank-16 LoRA on the refiner's Linear layers.
- `full` (B2.3): plus the refiner path itself (init queries, ego encoder, refiner), at 2e-5.

**Loss.** The release's own planning loss: `train_generator.mtd_loss` (winner-takes-all to the human
plus 0.5 x up to four teachers, on every pass, earlier passes at 0.1). The teachers are drawn from
the release's good anchors (`anchors_scores_index_v2`, indices into ./data/8192.npy), as its
training drew them.

**Readouts.** Every run is read at its last epoch, which nothing chose. The new candidates of the
select and report logs are scored offline by the vendored PDM scorer (`pdm_labels.score`), giving:
- (a) best-of-32 EPDMS and minADE: the refiner's own effect;
- (b) the pick of the frozen release scorer on the new pass-4 queries.

Backward through grid_sample is nondeterministic, so seeds do not reproduce bit for bit; the seed
range absorbs that.
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

from gallops.drivejepa_perception import geometry, scorers
from gallops.drivejepa_perception.train_scorer import ARMS, load, shuffled_rows, truth_agents  # noqa: F401

REPO = Path(__file__).resolve().parents[3]
BATCH, EPOCHS = 64, 3
LR, LR_REFINER = 1e-4, 2e-5
TEACHERS = 4
REFINER_PREFIXES = ("_trajectory_head", "init_feature", "hist_encoding")


class RefinerAdapter(nn.Module):
    """Adds to the refiner's queries (32 candidates x 8 poses, 256 each): own half, then agents."""

    def __init__(self, agents: bool):
        super().__init__()
        self.own = scorers.mlp(256 + 3, 256, 256, zero=True)
        self.agents = agents
        if agents:
            self.read = scorers.AgentRead(256)
            self.out = nn.Linear(128 + geometry.N_FEATURES, 256)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        self.context: dict = {}

    def forward(self, proposals, queries):
        b = queries.shape[0]
        poses = proposals.reshape(b, -1, 3)  # B, 256, 3: each query's own pose
        delta = self.own(torch.cat([queries, poses], dim=-1))
        if self.agents:
            a = self.context
            features, pairwise, _ = geometry.terms(
                proposals.detach(), a["xyz"][..., :2], a["lwh"][..., :2], a["vel"], a["path"], a["mask"]
            )
            k, t = proposals.shape[1], proposals.shape[2]

            def per_query(x):  # one value per candidate -> the same for each of its 8 queries
                return x[:, :, None].expand(-1, -1, t, *x.shape[2:]).reshape(b, k * t, *x.shape[2:])

            read, _ = self.read(queries, scorers.describe_agents(a), per_query(pairwise), a["mask"])
            delta = delta + self.out(torch.cat([read, per_query(features)], dim=-1))
        return delta


def build(entered: dict, mode: str, agents: bool, rank: int = 16):
    """The release's model, batch-safe, with the adapter hooked into every refiner pass."""
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    from gallops.drivejepa_b2d import batch_fix

    batch_fix.apply_all()
    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.eval()
    model._backbone = model._backbone.cpu()  # never run: the tokens are cached
    for p in model.parameters():
        p.requires_grad_(False)
    refiner = model._trajectory_head[0]
    if mode == "lora":
        for module in list(refiner.modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(module, child_name, scorers.LoRALinear(child, rank))
    elif mode == "full":
        for name, p in model.named_parameters():
            if name.startswith(REFINER_PREFIXES) and not name.startswith("_backbone"):
                p.requires_grad_(True)
    adapter = RefinerAdapter(agents)

    def hook(module, args):
        proposals, queries, image_feature = args
        return proposals, queries + adapter(proposals, queries), image_feature

    refiner.Bev_refiner.register_forward_pre_hook(hook)
    return agent, model, adapter


def teachers(rows, logs, tokens, anchors_root: Path, vocabulary: np.ndarray, rng) -> torch.Tensor:
    """Up to TEACHERS of each row's good anchors, drawn as the release's training drew them."""
    out = np.zeros((len(rows), TEACHERS, 8, 3), np.float32)
    for i, row in enumerate(rows):
        path = anchors_root / logs[row] / "unknown" / tokens[row] / "metric_cache.pkl.npy"
        good = np.load(path) if path.exists() else np.zeros(0, np.int64)
        if len(good):
            out[i] = vocabulary[rng.choice(good, TEACHERS, replace=len(good) < TEACHERS)]
    return torch.from_numpy(out)


def forward(model, data, rows, device):
    """All four passes' proposals and the pass-4 queries from cached tokens, batched with batch_fix."""
    from gallops.drivejepa_b2d.train_generator import refine_from_cache

    feat = torch.from_numpy(np.asarray(data["tokens"][rows], dtype=np.float32)).to(device)
    l2i = torch.from_numpy(data["lidar2img"][rows][:, 1:2]).to(device)
    img_shape = torch.from_numpy(data["img_shape"][rows]).to(device)
    ego = torch.from_numpy(data["ego"][rows]).to(device)
    return refine_from_cache(
        model, feat, l2i, img_shape, ego, data["spatial"].to(device), data["level"].to(device)
    )


def train(cfg, args) -> None:
    from gallops.drivejepa_b2d.head import Head, head_tensors_from_checkpoint
    from gallops.drivejepa_b2d.train_generator import mtd_loss
    from gallops.drivejepa_perception import env, preflight

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a run never overwrites one")
    data = load(Path(args.tokens), Path(args.labels), Path(args.pdm), args.arm, args.agents)
    data["tokens"] = np.load(Path(args.tokens) / "tokens.npy", mmap_mode="r")
    data["lidar2img"] = np.load(Path(args.tokens) / "lidar2img.npy")
    data["img_shape"] = np.load(Path(args.tokens) / "img_shape.npy")
    data["spatial"] = torch.tensor([[16, 32]], dtype=torch.long)
    data["level"] = torch.tensor([0], dtype=torch.long)
    labels = np.load(Path(args.labels) / "labels.npz")
    tokens_list = [str(t) for t in labels["token"]]
    entered = env.enter(cfg)
    vocabulary = np.load(Path(entered["code"]) / "navsim_v2/data/8192.npy")[:, 4::5].astype(np.float32)
    anchors_root = Path(cfg.resolve("{data.navsim_metric_cache}")).parent / "anchors_scores_index_v2"
    device = torch.device("cuda")
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    head = head.to(device).eval()
    report = preflight.run_refiner(
        cfg, args, data, lambda: build(entered, args.mode, args.arm != "C"), forward
    )
    out.mkdir(parents=True)
    rows = data["rows"]
    summary = {"args": vars(args), "train_git_sha": report["code"]["sha"], "preflight": report}
    picks = {"rows": rows["report"], "release": data["pick_release"][rows["report"]]}
    picks.update({f"subset_{k}": v[rows["report"]] for k, v in data["subsets"].items()})
    for seed in [int(s) for s in args.seeds.split(",")]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        agent, model, adapter = build(entered, args.mode, args.arm != "C")
        model, adapter = model.to(device), adapter.to(device)
        model._backbone = model._backbone.cpu()
        refiner_params = [p for n, p in model.named_parameters() if p.requires_grad]
        groups = [{"params": refiner_params, "lr": LR_REFINER if args.mode == "full" else LR}]
        groups.append({"params": list(adapter.parameters()), "lr": LR})
        optimiser = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.01)
        steps = (len(rows["train"]) // BATCH) * args.epochs
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimiser, lambda s, steps=steps: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
        )
        run_dir = out / f"seed{seed}"
        run_dir.mkdir()
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(rows["train"])
            for i in range(0, len(order) - BATCH + 1, BATCH):
                chunk = np.sort(order[i : i + BATCH])
                if args.arm != "C":
                    adapter.context = {
                        k: torch.from_numpy(np.asarray(v[chunk])).to(device)
                        for k, v in data["agents"].items()
                    }
                    adapter.context["mask"] = adapter.context["mask"].bool()
                props, _ = forward(model, data, chunk, device)
                target = torch.from_numpy(data["human"][chunk]).to(device)
                teach = teachers(chunk, data["token_logs"], tokens_list, anchors_root, vocabulary, rng).to(
                    device
                )
                tn = torch.full((len(chunk),), TEACHERS, device=device)
                loss = mtd_loss(props, target, teach, tn, prev_weight=0.1)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 10.0)
                optimiser.step()
                schedule.step()
            print(f"seed {seed} epoch {epoch}: loss {float(loss):.4f} ({time.time() - t0:.0f} s)", flush=True)
        state = {
            "adapter": adapter.state_dict(),
            "refiner": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        }
        torch.save(state, run_dir / f"epoch_{args.epochs:02d}.pt")
        summary[f"seed{seed}"] = readouts(
            model, adapter, head, data, rows, device, cfg, args, picks, f"seed{seed}"
        )
        print(json.dumps({"seed": seed, **summary[f"seed{seed}"]["report"]}), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    np.savez(out / "report_picks.npz", **picks)


@torch.no_grad()
def readouts(model, adapter, head, data, rows, device, cfg, args, picks, key) -> dict:
    """The new candidates on select and report, PDM-scored offline, and the frozen scorer's pick."""
    from multiprocessing import get_context

    from gallops.drivejepa_perception import pdm_labels

    model.eval()
    result = {}
    root = Path(cfg.resolve("{data.navsim_metric_cache}"))
    labels = np.load(Path(args.labels) / "labels.npz")
    for split in ("select", "report"):
        split_rows = rows[split]
        proposals, pick = (
            np.zeros((len(split_rows), 32, 8, 3), np.float32),
            np.zeros(len(split_rows), np.int64),
        )
        for i in range(0, len(split_rows), BATCH):
            chunk = split_rows[i : i + BATCH]
            if args.arm != "C":
                adapter.context = {
                    k: torch.from_numpy(np.asarray(v[chunk])).to(device) for k, v in data["agents"].items()
                }
                adapter.context["mask"] = adapter.context["mask"].bool()
            props, bev = forward(model, data, chunk, device)
            proposals[i : i + len(chunk)] = props[-1].float().cpu().numpy()
            pooled = bev.reshape(len(chunk), 32, 8, -1).amax(2)
            pick[i : i + len(chunk)] = head(pooled)[..., -1].argmax(1).cpu().numpy()
        jobs = [
            (pdm_labels.metric_cache_path(root, data["token_logs"][r], str(labels["token"][r])), proposals[j])
            for j, r in enumerate(split_rows)
        ]
        with get_context("fork").Pool(48) as pool:
            scored = pool.starmap(_score_only, jobs, chunksize=8)
        labels_new = np.stack(scored)  # n, 32, 9
        human = data["human"][split_rows]
        best = labels_new[:, :, -1].max(1)
        chosen = labels_new[np.arange(len(split_rows)), pick, -1]
        min_ade = np.linalg.norm(proposals[..., :2] - human[:, None, :, :2], axis=-1).mean(-1).min(-1)
        result[split] = {
            "best_of_32_epdms": float(best.mean()),
            "pick_epdms_frozen_scorer": float(chosen.mean()),
            "min_ade_m": float(min_ade.mean()),
        }
        if split == "report":
            picks[f"{key}_last"] = pick
            np.savez(
                Path(args.out) / f"{key}_report_candidates.npz",
                proposals=proposals,
                pdm=labels_new,
                pick=pick,
            )
    return result


def _score_only(path, proposals):
    from gallops.drivejepa_perception import pdm_labels

    return pdm_labels.score(path, np.asarray(proposals, dtype=np.float64))[0]


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.train_refiner")
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pdm", required=True)
    parser.add_argument("--agents", default="")
    parser.add_argument("--mode", choices=("frozen", "lora", "full"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.arm in ("F", "S") and not args.agents:
        parser.error("arms F and S read the side-car's agents: pass --agents")
    train(SiteConfig.load(), args)


if __name__ == "__main__":
    main()
