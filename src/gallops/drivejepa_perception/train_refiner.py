"""B2: give the refiner the side-car's agents, in three modes, against the arm that withholds them.

    python -m gallops.drivejepa_perception.train_refiner --gather-anchors --labels L --out A
    python -m gallops.drivejepa_perception.train_refiner --tokens T --labels L --pdm P --anchors A \
        --mode frozen --arm C --seeds 0,1,2 --out O [--agents AGENTS_DIR]

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


#: The longest good-anchor list a row may have (every file read so far holds exactly 256).
MAX_ANCHORS = 256


def _anchor_file(root: Path, log: str, token: str) -> Path:
    return root / log / "unknown" / token / "metric_cache.pkl.npy"


def _read_anchors(path: Path):
    return np.load(path) if path.exists() else np.zeros(0, np.int64)


def gather_anchors(cfg, labels_dir: Path, out: Path, workers: int = 48) -> None:
    """B2's teachers, gathered once: every row's good anchors (anchors_scores_index_v2), one array.

    Read per row from NFS they took 33 ms each, 2.1 s per batch of 64 (measured), which would
    dominate a B2 step. `anchors.npy` (n, MAX_ANCHORS) int16 holds each row's indices into
    8192.npy, padded with -1; `count.npy` how many (0 for a missing file: the row gets no teacher,
    as before). `preflight.run_refiner` holds the array to the files on random rows.
    """
    from multiprocessing import get_context

    from gallops.drivejepa_b2d.provenance import git_sha

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    labels = np.load(Path(labels_dir) / "labels.npz")
    logs = json.loads((Path(labels_dir) / "manifest.json").read_text())["logs"]
    root = anchors_root(cfg)
    paths = [_anchor_file(root, logs[i], str(t)) for i, t in zip(labels["log_id"], labels["token"])]  # noqa: B905 -- Python 3.9
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        lists = pool.map(_read_anchors, paths, chunksize=64)
    longest = max(len(a) for a in lists)
    if longest > MAX_ANCHORS:
        raise SystemExit(f"a row has {longest} good anchors, more than MAX_ANCHORS = {MAX_ANCHORS}")
    anchors = np.full((len(lists), MAX_ANCHORS), -1, np.int16)
    count = np.array([len(a) for a in lists], np.int16)
    for i, a in enumerate(lists):
        anchors[i, : len(a)] = a
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    np.save(staging / "anchors.npy", anchors)
    np.save(staging / "count.npy", count)
    manifest = {
        "pass": "anchors",
        "gallops_sha": git_sha(),
        "root": str(root),
        "rows": len(lists),
        "rows_without_a_file": int((count == 0).sum()),
        "counts": {str(k): int(v) for k, v in zip(*np.unique(count, return_counts=True))},  # noqa: B905 -- Python 3.9
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(json.dumps(manifest))


def anchors_root(cfg) -> Path:
    """Drive-JEPA's good-anchor index, extracted beside the metric cache (S0.5)."""
    return Path(cfg.resolve("{data.navsim_metric_cache}")).parent / "anchors_scores_index_v2"


def teachers(rows, anchors: np.ndarray, count: np.ndarray, vocabulary: np.ndarray, rng) -> torch.Tensor:
    """Up to TEACHERS of each row's good anchors, drawn as the release's training drew them."""
    out = np.zeros((len(rows), TEACHERS, 8, 3), np.float32)
    for i, row in enumerate(rows):
        good = anchors[row, : count[row]]
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
    data["anchors"] = np.load(Path(args.anchors) / "anchors.npy")
    data["anchor_count"] = np.load(Path(args.anchors) / "count.npy")
    entered = env.enter(cfg)
    vocabulary = np.load(Path(entered["code"]) / "navsim_v2/data/8192.npy")[:, 4::5].astype(np.float32)
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
                teach = teachers(chunk, data["anchors"], data["anchor_count"], vocabulary, rng).to(device)
                tn = torch.full((len(chunk),), TEACHERS, device=device)
                loss = mtd_loss(props, target, teach, tn, prev_weight=0.1)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 10.0)
                optimiser.step()
                schedule.step()
                step = i // BATCH + 1
                if step % 200 == 0:
                    rate = (time.time() - t0) / ((epoch - 1) * (len(order) // BATCH) + step)
                    print(
                        f"seed {seed} epoch {epoch} step {step}: loss {float(loss):.4f}, {rate:.2f} s/step",
                        flush=True,
                    )
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
    parser.add_argument("--gather-anchors", action="store_true", help="gather B2's teachers into one array")
    parser.add_argument("--tokens", default="")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pdm", default="")
    parser.add_argument("--anchors", default="", help="the --gather-anchors output")
    parser.add_argument("--agents", default="")
    parser.add_argument("--mode", choices=("frozen", "lora", "full"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.gather_anchors:
        gather_anchors(SiteConfig.load(), Path(args.labels), Path(args.out))
        return
    if not (args.tokens and args.pdm and args.anchors and args.mode and args.arm):
        parser.error("training needs --tokens, --pdm, --anchors, --mode and --arm")
    if args.arm in ("F", "S") and not args.agents:
        parser.error("arms F and S read the side-car's agents: pass --agents")
    train(SiteConfig.load(), args)


if __name__ == "__main__":
    main()
