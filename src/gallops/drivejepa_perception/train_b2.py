"""B-2: the side-car's hidden layer into the refiner, the planner fine-tuned downstream. One run per process.

    python -m gallops.drivejepa_perception.train_b2 train --recipe joint --arm HD --seed 0 --out RUN
    python -m gallops.drivejepa_perception.train_b2 train --recipe lora2 --stage 1 --arm D --seed 0 --out RUN
    python -m gallops.drivejepa_perception.train_b2 dump --run RUN --split train --out RUN/dump-train
    python -m gallops.drivejepa_perception.train_b2 train --recipe lora2 --stage 3 --stage1 RUN ...
    python -m gallops.drivejepa_perception.train_b2 dump --run RUN --split navtest --out RUN/dump-navtest

(stage 1, its stage-2 dump of the train rows, then stage 3, for the two-stage recipes.)

docs/drivejepa-perception-sidecar/part-b/B-2-hidden-refiner.md; PREDICTIONS.md, Part B · B-2 (committed
first).

**The model** is the release, run from the cached fp16 tokens with `batch_fix` (the R-tilde path), with an
adapter hooked before the shared `Bev_refiner` on every pass. Its own half reads each query and its pose
(every arm); its read gives each of the 256 queries attention over the arm's tokens with a learned "no agent"
slot; its output layers start at zero, so the untrained model is R-tilde, bit for bit (the preflight checks
it).

**The arms' tokens** (`Source.context`, `B2Adapter`):
- C: none. O and D: the labelled agents (B-1-2's) or the side-car's decoded list (confidence >= 0.5), encoded,
  the attention biased by -clearance / 5 from each pose to each agent at its time, with the 13 clearance
  features;
- HD: D, each kept slot's layer-6 hidden vector added to its token (LayerNorm, then its fold's input map);
- H6, H3: all 128 slots of that layer, hidden vector + a code of the slot's centre, biased by log(confidence).

The hidden vectors come from the five cross-fit side-cars (`b2_inputs`), each row its fold's. Their spaces do
not align (B2-G4: identity R-squared about -0.15, ridge 0.47-0.55), so each fold has its own input map.

**The recipes:**
- `joint` (B-2-1): the release's objective, resumed. The planning loss (`mtd_loss`) + the release's own
  `score_loss` (final-score BCE, the collision- and TTC-agent heads, the area head) against labels its own
  scorer computes online (`compute_navsim_score`, test=False), on CPU workers. The adapter, refiner LoRA and
  scorer LoRA train together; the auxiliary heads stay frozen.
- `lora2` and `full2` (B-2-2, B-2-3): stage 1 trains the adapter and the refiner (LoRA, or its own weights) on
  the planning loss alone; `dump --split train` writes every train row's new 32 and their vendored labels;
  stage 3 freezes passes 1-3 and trains a pass-4 copy of the read (starting as stage 1's) with the scorer
  (LoRA, or its weights) on the final-score BCE. The proposals, and so their labels, do not move in stage 3.

Train rows only (splits.json; the preflight refuses any other); the last epoch is kept: nothing is chosen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gallops.drivejepa_perception import geometry, scorers

REPO = Path(__file__).resolve().parents[3]
ARMS = ("C", "O", "D", "HD", "H6", "H3")
RECIPES = {"joint": "B-2-1", "lora2": "B-2-2", "full2": "B-2-3"}
AGENT_ARMS, HIDDEN_ARMS = ("O", "D", "HD"), ("HD", "H6", "H3")
BATCH, EPOCHS = 64, 3
LR, LR_SCORER_LORA, LR_FULL = 1e-4, 2e-5, 2e-5
WEIGHT_DECAY, CLIP, RANK = 0.01, 10.0, 16
FOLDS, D_HIDDEN, D_READ, HEADS = 5, 384, 128, 4
LOG_EVERY = 50
#: Each split's directories under {out} (the token, labels, PDM and hidden passes) and its metric cache.
SPLITS = {
    "navtrain": ("tokens-v1", "labels-v1", "pdm-v1", "b2-hidden-navtrain-v1"),
    "navtest": ("tokens-navtest-v1", "labels-navtest-v1", "pdm-navtest-v1", "b2-hidden-navtest-v1"),
    "fresh": ("tokens-fresh-v1", "labels-fresh-v1", "pdm-fresh-v1", "b2-hidden-fresh-v1"),
}
_BATCH_FIXED: list = []


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------- the data


class Source:
    """Everything one split's passes wrote, by row: what the refiner, the arms and the losses read."""

    def __init__(self, out_root: Path, split: str, arm: str):
        from gallops.drivejepa_perception import navsim_data
        from gallops.drivejepa_perception import targets as tg

        tok, lab, pdm, hid = SPLITS[split]
        root = Path(out_root)
        self.split, self.arm = split, arm
        self.tokens_dir, self.labels_dir, self.pdm_dir, self.hidden_dir = (
            root / d for d in (tok, lab, pdm, hid)
        )
        self.labels = dict(np.load(self.labels_dir / "labels.npz"))
        manifest = json.loads((self.labels_dir / "manifest.json").read_text())
        if navsim_data.source_of(manifest) != split:
            raise SystemExit(f"{self.labels_dir} is not {split}'s labels pass")
        self.token_logs = np.array(manifest["logs"])[self.labels["log_id"]]
        self.tokens = [str(t) for t in self.labels["token"]]
        self.data = {
            "tokens": np.load(self.tokens_dir / "tokens.npy", mmap_mode="r"),
            "lidar2img": np.load(self.tokens_dir / "lidar2img.npy"),
            "img_shape": np.load(self.tokens_dir / "img_shape.npy"),
            "ego": np.load(self.tokens_dir / "ego_status.npy")[:, -1].astype(np.float32),
            "spatial": torch.tensor([[16, 32]], dtype=torch.long),
            "level": torch.tensor([0], dtype=torch.long),
        }
        self.human = self.labels["ego_future"].astype(np.float32)
        self.release = {
            "proposals": np.load(self.tokens_dir / "proposals.npy", mmap_mode="r"),
            "pick": np.load(self.tokens_dir / "pick.npy").astype(np.int64),
        }
        self.hidden_manifest = json.loads((self.hidden_dir / "manifest.json").read_text())
        self.fold = np.load(self.hidden_dir / "fold.npy")
        self.agents = None
        if arm in ("D", "HD"):
            self.agents = dict(np.load(self.hidden_dir / "agents.npz"))
        elif arm == "O":
            projections = tg.front_projection(self.data["lidar2img"])
            self.agents = tg.truth_agents(self.labels, projections, fill="constant_velocity")
            self.agents.pop("path_ok")
        layer = {"HD": 6, "H6": 6, "H3": 3}.get(arm)
        self.layer = layer
        if layer is not None:
            self.hidden = np.load(self.hidden_dir / f"hidden{layer}.npy", mmap_mode="r")
            self.centre = np.load(self.hidden_dir / f"centre{layer}.npy")
            self.conf = np.load(self.hidden_dir / f"conf{layer}.npy")

    def context(self, rows: np.ndarray, device, swap=None) -> dict:
        """The arm's inputs for `rows`, on `device`. `swap` (test only) replaces each row's inputs by another
        row's (another log's), or blanks them ("none")."""
        if self.arm == "C":
            return {}
        rows = np.asarray(rows)
        src = rows if swap is None or isinstance(swap, str) else np.asarray(swap)[rows]
        out = {}
        if self.agents is not None:
            for key, value in self.agents.items():
                out[key] = torch.from_numpy(np.asarray(value[src])).to(device)
            out["mask"] = out["mask"].bool()
        if self.layer is not None:
            out["hidden"] = torch.from_numpy(np.asarray(self.hidden[src], dtype=np.float32)).to(device)
            out["centre"] = torch.from_numpy(np.asarray(self.centre[src])).to(device)
            out["conf"] = torch.from_numpy(np.asarray(self.conf[src])).to(device)
            out["fold"] = torch.from_numpy(self.fold[src].astype(np.int64)).to(device)
            if (out["fold"] < 0).any():
                raise SystemExit(f"{self.split}: a row without a fold's slots was read")
        if swap == "none":
            if "mask" in out:
                out["mask"] = torch.zeros_like(out["mask"])
            if "conf" in out:
                out["blank"] = torch.ones(len(rows), dtype=torch.bool, device=device)
        return out


def train_rows(src: Source) -> np.ndarray:
    """navtrain's train rows with a metric cache (PDM-labelled), as `train_scorer.load` reads them."""
    splits = json.loads((REPO / "recipes/drivejepa_perception/splits.json").read_text())
    cand = np.load(src.pdm_dir / "pdm.npz")["candidates"]
    return np.flatnonzero(np.isin(src.token_logs, splits["train"]) & np.isfinite(cand[:, 0, -1]))


def select_rows(src: Source) -> np.ndarray:
    splits = json.loads((REPO / "recipes/drivejepa_perception/splits.json").read_text())
    return np.flatnonzero(np.isin(src.token_logs, splits["select"]))


# ------------------------------------------------------------------------------------------ the model


class SlotRead(nn.Module):
    """Query -> token attention with a per-head scaled bias and a learned null token always attendable
    (`scorers.AgentRead`, the tokens given rather than encoded inside)."""

    def __init__(self, d_query: int = 256, d: int = D_READ, heads: int = HEADS):
        super().__init__()
        self.heads, self.d = heads, d
        self.q, self.k, self.v = nn.Linear(d_query, d), nn.Linear(d, d), nn.Linear(d, d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.bias_scale = nn.Parameter(torch.ones(heads))

    def forward(self, query, tokens, bias, keep):
        b, k, _ = query.shape
        tokens = torch.cat([tokens, self.null.expand(b, 1, -1)], dim=1)
        keep = torch.cat([keep, torch.ones(b, 1, dtype=torch.bool, device=keep.device)], dim=1)
        bias = torch.cat([bias, torch.zeros(b, k, 1, device=bias.device, dtype=bias.dtype)], dim=-1)
        h, dh = self.heads, self.d // self.heads
        q = self.q(query).view(b, k, h, dh).transpose(1, 2)
        kk = self.k(tokens).view(b, -1, h, dh).transpose(1, 2)
        v = self.v(tokens).view(b, -1, h, dh).transpose(1, 2)
        logits = q @ kk.transpose(-1, -2) / dh**0.5 + bias[:, None] * self.bias_scale[None, :, None, None]
        logits = logits.masked_fill(~keep[:, None, None, :], float("-inf"))
        weights = logits.softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(b, k, self.d)
        return out, weights.mean(dim=1)[..., -1].mean()  # the share of attention on the null token


class B2Adapter(nn.Module):
    """Adds to the refiner's 256 queries: an own half (every arm), and the read of the arm's tokens."""

    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.own = scorers.mlp(256 + 3, 256, 256, zero=True)
        self.context: dict = {}
        self.null_share = float("nan")
        if arm == "C":
            return
        self.read = SlotRead()
        if arm in AGENT_ARMS:
            self.encode = scorers.mlp(scorers.AGENT_FEATURES, D_READ, D_READ)
        if arm in HIDDEN_ARMS:
            self.norm = nn.LayerNorm(D_HIDDEN)
            # One input map per fold: the folds' hidden spaces do not align (B2-G4).
            self.maps = nn.Parameter(torch.randn(FOLDS, D_HIDDEN, D_READ) / math.sqrt(D_HIDDEN))
            self.map_bias = nn.Parameter(torch.zeros(FOLDS, D_READ))
        if arm in ("H6", "H3"):
            self.centre = scorers.mlp(3, D_READ, D_READ)
        width = D_READ + (geometry.N_FEATURES if arm in AGENT_ARMS else 0)
        self.out = nn.Linear(width, 256)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def hidden_tokens(self, c: dict):
        x = self.norm(c["hidden"])
        return torch.einsum("bad,bde->bae", x, self.maps[c["fold"]]) + self.map_bias[c["fold"]][:, None]

    def forward(self, proposals, queries):
        b = queries.shape[0]
        poses = proposals.reshape(b, -1, 3)
        delta = self.own(torch.cat([queries, poses], dim=-1))
        if self.arm == "C":
            return delta
        c = self.context
        k, t = proposals.shape[1], proposals.shape[2]

        def per_query(x):  # one value per proposal -> the same for each of its 8 queries
            return x[:, :, None].expand(-1, -1, t, *x.shape[2:]).reshape(b, k * t, *x.shape[2:])

        if self.arm in AGENT_ARMS:
            features, pairwise, _ = geometry.terms(
                proposals.detach(), c["xyz"][..., :2], c["lwh"][..., :2], c["vel"], c["path"], c["mask"]
            )
            tokens = self.encode(scorers.describe_agents(c))
            if self.arm == "HD":
                tokens = tokens + self.hidden_tokens(c)
            read, self.null_share = self.read(queries, tokens, per_query(-pairwise / 5.0), c["mask"])
            return delta + self.out(torch.cat([read, per_query(features)], dim=-1))
        centre = c["centre"] * torch.tensor([1 / 20.0, 1 / 20.0, 1 / 2.0], device=queries.device)
        tokens = self.hidden_tokens(c) + self.centre(centre)
        bias = torch.log(c["conf"].clamp(min=1e-3))[:, None, :].expand(-1, k * t, -1)
        keep = torch.ones_like(c["conf"], dtype=torch.bool)
        if "blank" in c:
            keep = keep & ~c["blank"][:, None]
        read, self.null_share = self.read(queries, tokens, bias, keep)
        return delta + self.out(read)


class PassHook:
    """The pre-hook on `Bev_refiner`: adds the adapter's δ before every pass's BEVFormer layer; with `pass4`,
    a separate copy writes pass 4 (stage 3), so passes 1-3, and the proposals, stay as stage 1 left them."""

    def __init__(self, adapter: B2Adapter, pass4=None):
        self.adapter, self.pass4, self.k = adapter, pass4, 0

    def __call__(self, module, args):
        proposals, queries, image_feature = args
        self.k += 1
        a = self.pass4 if (self.pass4 is not None and self.k == 4) else self.adapter
        return proposals, queries + a(proposals, queries), image_feature


def lora_linears(module: nn.Module, rank: int = RANK) -> None:
    """Every nn.Linear under `module` replaced by a frozen copy plus rank-r LoRA (B at zero)."""
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, scorers.LoRALinear(child, rank))


def build(entered: dict, recipe: str, arm: str, stage: int):
    """The release, frozen, with the recipe's trainable parts and the adapter hooked. Returns (agent, model,
    adapter, hook)."""
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    from gallops.drivejepa_b2d import batch_fix

    if not _BATCH_FIXED:
        batch_fix.apply_all()
        _BATCH_FIXED.append(True)
    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.eval()
    model._backbone = model._backbone.cpu()  # never run: the tokens are cached
    for p in model.parameters():
        p.requires_grad_(False)
    refiner = model._trajectory_head[0]
    head = model.scorer.pred_score.mlp
    if recipe in ("joint", "lora2"):
        lora_linears(refiner)
        if recipe == "joint" or stage == 3:
            lora_linears(head)
    adapter = B2Adapter(arm)
    hook = PassHook(adapter)
    refiner.Bev_refiner.register_forward_pre_hook(hook)
    return agent, model, adapter, hook


def trainable(model, adapter, hook, recipe: str, stage: int) -> list:
    """The optimiser's groups for this recipe and stage; everything else stays frozen."""
    refiner = model._trajectory_head[0]
    head = model.scorer.pred_score.mlp
    groups = []
    if stage in (0, 1):
        groups.append({"params": list(adapter.parameters()), "lr": LR, "name": "adapter"})
        if recipe in ("joint", "lora2"):
            ps = [p for n, p in refiner.named_parameters() if n.endswith((".a", ".b"))]
            groups.append({"params": ps, "lr": LR, "name": "refiner LoRA"})
        else:
            from gallops.drivejepa_perception.train_refiner import REFINER_PREFIXES

            ps = [p for n, p in model.named_parameters() if n.startswith(REFINER_PREFIXES)]
            groups.append({"params": ps, "lr": LR_FULL, "name": "refiner weights"})
        if recipe == "joint":
            ps = [p for n, p in head.named_parameters() if n.endswith((".a", ".b"))]
            groups.append({"params": ps, "lr": LR_SCORER_LORA, "name": "scorer LoRA"})
    else:  # stage 3
        groups.append({"params": list(hook.pass4.parameters()), "lr": LR, "name": "pass-4 read"})
        if recipe == "lora2":
            ps = [p for n, p in head.named_parameters() if n.endswith((".a", ".b"))]
            groups.append({"params": ps, "lr": LR_SCORER_LORA, "name": "scorer LoRA"})
        else:
            groups.append({"params": list(head.parameters()), "lr": LR_FULL, "name": "scorer weights"})
    for g in groups:
        for p in g["params"]:
            p.requires_grad_(True)
    return groups


def forward(model, src: Source, rows: np.ndarray, device, hook: PassHook, swap=None) -> tuple:
    """The four passes from cache and the scorer: (proposals per pass, the pass-4 queries, the scorer's
    outputs). The hook's pass counter restarts for every forward."""
    from gallops.drivejepa_perception import train_refiner

    ctx = src.context(rows, device, swap)
    hook.adapter.context = ctx
    if hook.pass4 is not None:
        hook.pass4.context = ctx
    hook.k = 0
    props, bev = train_refiner.forward(model, src.data, rows, device)
    scored = model.scorer(props[-1], bev)
    return props, bev, scored


# ----------------------------------------------------------------------------------- online labels


def _online(job) -> tuple:
    """The release's own scorer on one row's 32 plans, with the auxiliary targets (test=False). A job carries
    its metric cache's path: run as `-m`, this module is also `__main__`, so worker globals would split."""
    from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as cns

    path, poses = job
    cache = cns.load_metric_cache(path)
    states = cns.before_score_proposals(cache, poses, cache.ego_state)
    final = cns.scorer.score_proposals(
        states,
        cache.observation,
        cache.centerline,
        cache.route_lane_ids,
        cache.drivable_area_map,
        cache.map_parameters,
        None,
        cache.past_human_trajectory,
    )
    scores, corners, labels, areas = cns.after_score_proposals(cns.scorer, final, cache.ego_state, False)
    return (
        np.asarray(scores, np.float32),
        np.asarray(corners, np.float32),
        np.asarray(labels, bool),
        np.asarray(areas, bool),
    )


def online_targets(pool, src: Source, root: Path, rows: np.ndarray, proposals: np.ndarray, device) -> tuple:
    from gallops.drivejepa_perception import pdm_labels

    jobs = [
        (
            pdm_labels.metric_cache_path(root, src.token_logs[r], src.tokens[r]),
            proposals[i].astype(np.float64),
        )
        for i, r in enumerate(rows)
    ]
    out = pool.map(_online, jobs)
    stack = [np.stack([o[j] for o in out]) for j in range(4)]
    return tuple(torch.from_numpy(x).to(device) for x in stack)


# ------------------------------------------------------------------------------------------ training


def lr_lambda(total: int):
    return lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total))


def save_weights(model, adapter, hook, path: Path) -> str:
    state = {
        "adapter": adapter.state_dict(),
        "pass4": hook.pass4.state_dict() if hook.pass4 is not None else None,
        "model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
    }
    torch.save(state, path)
    return sha256_of(path)


def load_weights(model, adapter, hook, path: Path, pass4: bool = True) -> None:
    state = torch.load(path, map_location="cpu")
    adapter.load_state_dict(state["adapter"])
    if pass4 and state.get("pass4") is not None and hook.pass4 is not None:
        hook.pass4.load_state_dict(state["pass4"])
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state["model"].items():
            params[name].copy_(value.to(params[name].device))


def train(cfg, args) -> None:
    from multiprocessing import get_context

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_b2d.train_generator import mtd_loss
    from gallops.drivejepa_perception import env, train_refiner

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a run never overwrites one")
    stage = args.stage
    if args.recipe == "joint" and stage != 0:
        raise SystemExit("the joint recipe has no stages")
    if args.recipe != "joint" and stage not in (1, 3):
        raise SystemExit("a two-stage recipe trains --stage 1 or --stage 3")
    entered = env.enter(cfg)
    out_root = Path(cfg.resolve("{runs}/drivejepa_perception"))
    src = Source(out_root, "navtrain", args.arm)
    rows = train_rows(src)
    pool = None
    root = Path(cfg.resolve("{data.navsim_metric_cache}"))
    if args.recipe == "joint":
        # The workers fork before CUDA is touched, and each scores one row's 32 plans per job.
        pool = get_context("fork").Pool(args.workers)
    labels3 = None
    if stage == 3:
        dump = Path(args.stage1) / "dump-train"
        labels3 = np.load(dump / "labels.npz")["candidates"]
    from gallops.drivejepa_perception import preflight

    report = preflight.run_b2(cfg, args, src, rows, pool, root)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    agent, model, adapter, hook = build(entered, args.recipe, args.arm, stage)
    if stage == 3:
        load_weights(model, adapter, hook, Path(args.stage1) / "weights.pt", pass4=False)
        hook.pass4 = copy.deepcopy(adapter)
    model, adapter = model.to(device), adapter.to(device)
    if hook.pass4 is not None:
        hook.pass4 = hook.pass4.to(device)
    model._backbone = model._backbone.cpu()
    groups = trainable(model, adapter, hook, args.recipe, stage)
    counts = {g["name"]: int(sum(p.numel() for p in g["params"])) for g in groups}
    optimiser = torch.optim.AdamW(
        [{"params": g["params"], "lr": g["lr"]} for g in groups], weight_decay=WEIGHT_DECAY
    )
    steps = (len(rows) // BATCH) * args.epochs
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda(steps))
    anchors = np.load(out_root / "anchors-v1" / "anchors.npy")
    count = np.load(out_root / "anchors-v1" / "count.npy")
    vocabulary = np.load(Path(entered["code"]) / "navsim_v2/data/8192.npy")[:, 4::5].astype(np.float32)
    conf = agent._config
    out.mkdir(parents=True)
    run = {
        "args": vars(args),
        "sub_version": RECIPES[args.recipe],
        "stage": stage,
        "train_git_sha": report["code"]["sha"],
        "gallops_sha": git_sha(),
        "preflight": report,
        "rows": int(len(rows)),
        "trainable": counts,
        "steps": steps,
        "smoke": bool(args.max_steps),
    }
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str))
    params = [p for g in groups for p in g["params"]]
    t0, step, window = time.time(), 0, {}
    model.train()  # dropout is 0 everywhere; train mode makes the scorer compute its auxiliary heads
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(rows)
        for i in range(0, len(order) - BATCH + 1, BATCH):
            if args.max_steps and step >= args.max_steps:
                break
            chunk = np.sort(order[i : i + BATCH])
            t_step = time.time()
            props, _, scored = forward(model, src, chunk, device, hook)
            pred_logit, pred_agents, pred_area = scored[0], scored[1], scored[2]
            parts = {}
            if stage in (0, 1):
                target = torch.from_numpy(src.human[chunk]).to(device)
                teach, tn = train_refiner.teachers(chunk, anchors, count, vocabulary, rng)
                parts["planning"] = mtd_loss(props, target, teach.to(device), tn.to(device), prev_weight=0.1)
            if args.recipe == "joint":
                t_wait = time.time()
                scores_t, corners, valid, areas = online_targets(
                    pool, src, root, chunk, props[-1].detach().float().cpu().numpy(), device
                )
                parts["_wait_s"] = time.time() - t_wait
                sub, final, ce, l1, area = agent.score_loss(
                    pred_logit, pred_agents, pred_area, scores_t, corners, valid, areas
                )
                parts.update({"final": final, "agents_ce": ce, "agents_l1": l1, "area": area, "sub": sub})
                loss = (
                    conf.trajectory_weight * parts["planning"]
                    + conf.sub_score_weight * sub
                    + conf.final_score_weight * final
                    + conf.pred_ce_weight * ce
                    + conf.pred_l1_weight * l1
                    + conf.pred_area_weight * area
                )
            elif stage == 1:
                loss = parts["planning"]
            else:
                target = torch.from_numpy(labels3[chunk][..., -1]).to(device)
                parts["final"] = F.binary_cross_entropy_with_logits(pred_logit[..., -1].float(), target)
                loss = parts["final"]
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(params, CLIP))
            optimiser.step()
            schedule.step()
            step += 1
            record = {k: float(v) for k, v in parts.items()}
            record.update(loss=float(loss), grad_norm=grad, step_s=time.time() - t_step)
            if args.arm != "C":
                record["null_share"] = float(hook.adapter.null_share if stage != 3 else hook.pass4.null_share)
            for k, v in record.items():
                window.setdefault(k, []).append(v)
            if step % LOG_EVERY == 0:
                line = {k: float(np.mean(v)) for k, v in window.items()}
                line.update(
                    step=step, epoch=epoch, lr=schedule.get_last_lr()[0], seconds=round(time.time() - t0)
                )
                with open(out / "steps.jsonl", "a") as fh:
                    fh.write(json.dumps(line) + "\n")
                window = {}
                if step % (LOG_EVERY * 10) == 0:
                    print(
                        json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in line.items()}),
                        flush=True,
                    )
        print(f"epoch {epoch} done ({time.time() - t0:.0f} s)", flush=True)
    model.eval()
    sha = save_weights(model, adapter, hook, out / "weights.pt")
    run.update(weights_sha256=sha, seconds=round(time.time() - t0), ended=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str))
    if pool is not None:
        pool.close()
    print(json.dumps({k: run[k] for k in ("sub_version", "stage", "trainable", "weights_sha256", "seconds")}))


# --------------------------------------------------------------------------------------- the dumps


def restore(entered: dict, run: Path, device):
    """A finished run's model, from its run.json and weights (a stage-3 run brings its stage 1 along)."""
    meta = json.loads((run / "run.json").read_text())
    a = meta["args"]
    stage = meta["stage"]
    agent, model, adapter, hook = build(entered, a["recipe"], a["arm"], stage)
    if stage == 3:
        hook.pass4 = copy.deepcopy(adapter)
    load_weights(model, adapter, hook, run / "weights.pt")
    model, adapter = model.to(device).eval(), adapter.to(device).eval()
    if hook.pass4 is not None:
        hook.pass4 = hook.pass4.to(device).eval()
    model._backbone = model._backbone.cpu()
    return meta, model, hook


@torch.no_grad()
def dump(cfg, args) -> dict:
    """A run's 32 proposals, scores and pick on a split's rows, and their vendored labels; optionally with the
    arm's inputs swapped at test (another log's, or none)."""
    from multiprocessing import get_context

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env, pdm_labels, train_scorer

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    entered = env.enter(cfg)
    out_root = Path(cfg.resolve("{runs}/drivejepa_perception"))
    run = Path(args.run)
    meta = json.loads((run / "run.json").read_text())
    split = {"train": "navtrain", "select": "navtrain"}.get(args.split, args.split)
    src = Source(out_root, split, meta["args"]["arm"])
    if args.split == "train":
        rows = train_rows(src)
    elif args.split == "select":
        rows = select_rows(src)
    else:
        rows = np.arange(len(src.tokens))
    swap = None
    if args.swap == "none":
        swap = "none"
    elif args.swap == "log":
        swap = train_scorer.shuffled_rows(src.token_logs)
    root = {
        "navtrain": Path(cfg.resolve("{data.navsim_metric_cache}")),
        "fresh": out_root / "metric-cache-fresh-v2",
    }.get(split)
    if split == "navtest":
        from gallops.drivejepa_perception import navtest

        root = navtest.verified_cache(out_root / "part-b/b0/navtest-cache-verify")
    device = torch.device("cuda")
    _, model, hook = restore(entered, run, device)
    n = len(rows)
    proposals = np.zeros((n, 32, 8, 3), np.float32)
    logits = np.zeros((n, 32, 6), np.float32)
    t0 = time.time()
    for i in range(0, n, BATCH):
        chunk = rows[i : i + BATCH]
        props, _, scored = forward(model, src, chunk, device, hook, swap=swap)
        proposals[i : i + len(chunk)] = props[-1].float().cpu().numpy()
        logits[i : i + len(chunk)] = scored[0].float().cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits[..., -1]))
    pick = scores.argmax(1)
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    np.savez(staging / "dump.npz", rows=rows, proposals=proposals, logits=logits, scores=scores, pick=pick)
    labelled = None
    if not args.no_labels:
        jobs = [
            (
                pdm_labels.metric_cache_path(root, src.token_logs[r], src.tokens[r]),
                proposals[j].astype(np.float64),
            )
            for j, r in enumerate(rows)
        ]
        with get_context("fork").Pool(args.workers) as pool:
            done = pool.starmap(_label, jobs, chunksize=4)
        labelled = {k: np.stack([d[k] for d in done]) for k in done[0]}
        np.savez(
            staging / "labels.npz",
            candidates=labelled["scores"],
            **{k: v for k, v in labelled.items() if k != "scores"},
        )
    record = {
        "pass": "b2-dump",
        "run": str(run),
        "split": args.split,
        "swap": args.swap or None,
        "rows": int(n),
        "weights_sha256": sha256_of(run / "weights.pt"),
        "gallops_sha": git_sha(),
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(record, indent=1))
    staging.rename(out)
    print(json.dumps(record))
    return record


def _label(path, proposals) -> dict:
    from gallops.drivejepa_perception import pdm_labels

    if not Path(path).exists():
        raise SystemExit(f"no metric cache at {path}")
    return pdm_labels.score_detail(path, proposals)


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.train_b2")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--recipe", required=True, choices=tuple(RECIPES))
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", type=int, default=0, help="0 for joint; 1 or 3 for a two-stage recipe")
    p.add_argument("--stage1", default="", help="stage 3: its stage-1 run (with dump-train)")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--workers", type=int, default=40, help="joint: the online scorer's CPU workers")
    p.add_argument("--max-steps", type=int, default=0, help="a pipeline check (smoke): stop after N steps")
    p.add_argument("--out", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--run", required=True)
    p.add_argument("--split", required=True, choices=("train", "select", "navtest", "fresh"))
    p.add_argument("--swap", default="", choices=("", "none", "log"))
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "train":
        if args.stage == 3 and not args.stage1:
            parser.error("stage 3 needs --stage1")
        train(SiteConfig.load(), args)
        return
    dump(SiteConfig.load(), args)


if __name__ == "__main__":
    # Run as the package's module, never as __main__: the preflight imports this module by name, and two
    # copies would keep two sets of process state (batch_fix's once-per-process patch, among others).
    from gallops.drivejepa_perception import train_b2

    train_b2.main()
