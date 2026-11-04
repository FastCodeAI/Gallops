"""One training step of Drive-JEPA's refiner with the auxiliary heads, and the model it runs on.

Runs in Drive-JEPA's interpreter (torch, Python 3.9) with its cwd at navsim_v2/ -- the agent
opens ./data/8192.npy and ./vjepa2/... relative to it -- and paddock's src/ importable.

One step, in order:

1. **The vendor's own forward**, unchanged, in eval mode: the frozen ViT-L and projector turn two
   front frames into 512 image tokens; four passes of the one shared refiner turn 32 x 8
   trajectory queries into query features and 32 proposals per pass; the frozen scorer picks
   one. Eval mode turns off GridMask in the encoder (as the cached-feature trainer has no
   GridMask either); the refiner's dropout is 0, so it computes the same in either mode.
2. **The heads** on the last pass's query features (captured as the scorer's input): upstream's
   two, called as `Scorer.forward` calls them in training mode, and `SceneHeads`.
3. **The loss**: the planning term `drivejepa_b2d.train_generator.mtd_loss` (expert winner-takes-
   all + 0.5 x pseudo-teachers, every pass, imported, not restated) plus the weighted heads.
4. **One optimizer step.** Only the refiner path (`swerve.TRAINABLE_REFINER`) and the heads have
   requires_grad; `report` says how large each part's gradient and update were.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from gallops.drivejepa_aux import bev, heads, targets
from gallops.drivejepa_aux.publish import SERVED, UPSTREAM_HEAD_KEYS

#: What a run changes and saves: the refiner path -- the one shared refiner under its first name,
#: as `train_generator.gen_state` keeps it -- and upstream's heads.
TRAINED_KEYS = SERVED + UPSTREAM_HEAD_KEYS
#: The refiner path trains at the Bench2Drive generator stage's rate; the heads, which start from
#: random weights, at the rate the release itself was trained at (DriveJEPAAgent lr).
LR_REFINER = 2e-5
LR_HEADS = 1e-4
#: The released checkpoint, where paddock's model.yaml finds it.
RELEASE = "{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"


def enter(cfg) -> str:
    """Drive-JEPA's environment, as paddock's model.yaml gives it to its node; returns the release.

    Changes into navsim_v2/: the agent opens ./data/8192.npy and ./vjepa2/... relative to its cwd.
    NAVSIM's data need not exist here -- nothing reads it -- so its configured path is passed as it
    is, not resolved (resolving checks it exists). paddock's src/ supplies `Command`.
    """
    navsim_v2 = Path(cfg.resolve("{code.drivejepa}")) / "navsim_v2"
    exp = cfg.resolve("{data.navsim.exp}")
    navsim_data = str(cfg.datasets["navsim"].path)
    os.environ.update(
        NAVSIM_DEVKIT_ROOT=str(navsim_v2),
        NAVSIM_EXP_ROOT=exp,
        OPENSCENE_DATA_ROOT=navsim_data,
        NUPLAN_MAPS_ROOT=navsim_data + "/maps",
        NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
        DRIVE_JEPA_MSDA="auto",
    )
    for path in (cfg.resolve("{code.paddock}") + "/src", str(navsim_v2)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(navsim_v2)
    return cfg.resolve(RELEASE)


def check_grid(config, aux: bool) -> None:
    """The label grid and box count in `bev` and `targets` are the model's; refuse if not."""
    got = (config.bev_pixel_height, config.bev_pixel_width, config.bev_pixel_size, config.num_bounding_boxes)
    want = (bev.ROWS, bev.COLS, bev.CELL_M, targets.MAX_AGENTS)
    if got != want:
        raise SystemExit(f"DriveJEPAConfig BEV grid / box count {got} != the labels' {want}")
    if config.latent:
        raise SystemExit(
            "config.latent filters boxes by angle in the agent loss; the labels assume it is off"
        )
    if aux and config.num_bev_classes != len(bev.CLASSES):
        raise SystemExit(f"num_bev_classes {config.num_bev_classes} != {len(bev.CLASSES)} label classes")


def build_model(checkpoint: str, device: str, aux: bool = True, seed: int = 0, scene: bool = True):
    """The released model with upstream's heads switched on, and the new heads.

    Loads the released state dict with strict=False and refuses unless the only missing keys are
    the two switched-on heads' and nothing is unexpected. Applies the batch fix, puts the model in
    eval mode, and leaves requires_grad on the refiner path and the heads alone. `aux` False is
    the release as it is (arm C); `scene` False keeps upstream's two heads and not ours (arm T1).
    """
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    from gallops.drivejepa_b2d import batch_fix
    from gallops.train.swerve import TRAINABLE_REFINER

    config = DriveJEPAConfig()
    if aux:
        heads.configure(config)
    check_grid(config, aux)
    torch.manual_seed(seed)  # the new heads' initial weights
    agent = DriveJEPAAgent(config=config, lr=1e-4, checkpoint_path=checkpoint)
    state = torch.load(checkpoint, map_location="cpu")["state_dict"]
    result = agent.load_state_dict(
        {k.replace("agent._pad_model", "_pad_model"): v for k, v in state.items()}, strict=False
    )
    missing = [k for k in result.missing_keys if not k[len("_pad_model.") :].startswith(UPSTREAM_HEAD_KEYS)]
    if missing or result.unexpected_keys:
        raise SystemExit(
            f"checkpoint mismatch: missing {missing[:5]}, unexpected {result.unexpected_keys[:5]}"
        )
    batch_fix.apply_all()

    model = agent._pad_model.to(device).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(TRAINABLE_REFINER) or name.startswith(UPSTREAM_HEAD_KEYS)
    scene_heads = heads.SceneHeads(config.tf_d_model).to(device).eval() if aux and scene else None
    return agent, model, scene_heads, config


def trainable_state(model, scene_heads) -> dict:
    """What a run changed, on the CPU: `model` (TRAINED_KEYS) and `scene_heads`."""
    trained = {k: v for k, v in model.state_dict().items() if k.startswith(TRAINED_KEYS)}
    state = {"model": {k: v.detach().cpu().clone() for k, v in trained.items()}}
    if scene_heads is not None:
        state["scene_heads"] = {k: v.detach().cpu().clone() for k, v in scene_heads.state_dict().items()}
    return state


def load_trainable(model, scene_heads, state: dict) -> None:
    """`trainable_state`'s inverse. Loading `_trajectory_head.0.*` sets the one shared refiner, so
    the four passes move together; refuses a key the model does not have."""
    result = model.load_state_dict(state["model"], strict=False)
    if result.unexpected_keys:
        raise SystemExit(f"trained state has keys the model lacks: {result.unexpected_keys[:5]}")
    if scene_heads is not None:
        scene_heads.load_state_dict(state["scene_heads"])


def cached_forward(
    model, scene_heads, batch: dict, spatial, level, detach_heads: bool = False, score: bool = False
):
    """`forward` from the encoder's cached output: the same result, without the frozen ViT.

    `batch` holds the cache's tokens (B, 512, 256), l2i (B, 1, 4, 4), img_shape (B, 4, 3) and ego
    (B, 11). The four passes are `train_generator.refine_from_cache`, the generator stage's own;
    `score` also runs the frozen scorer and picks a plan, as the vendor's forward does. B > 1 needs
    the batch fix (`build_model` applies it); the preflight holds this path to the live one.
    """
    from gallops.drivejepa_b2d.train_generator import refine_from_cache

    proposals, query = refine_from_cache(
        model, batch["tokens"], batch["l2i"], batch["img_shape"], batch["ego"], spatial, level
    )
    out = {"proposal_list": proposals, "proposals": proposals[-1]}
    if score:
        out["pred_logit"] = model.scorer(proposals[-1], query)[0]
        out["pdm_score"] = torch.sigmoid(out["pred_logit"])[:, :, -1]
        pick = out["pdm_score"].argmax(1)
        out["trajectory"] = proposals[-1][torch.arange(len(pick), device=pick.device), pick]
    head_input = query.detach() if detach_heads else query
    aux = heads.run_upstream_heads(model.scorer, head_input)
    if scene_heads is not None:
        aux.update(scene_heads(head_input))
    return {"out": out, "query_features": query, "aux": aux}


def make_optimizer(model, scene_heads) -> torch.optim.Optimizer:
    from gallops.train.swerve import TRAINABLE_REFINER

    refiner = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(TRAINABLE_REFINER)]
    head_params = [
        p for n, p in model.named_parameters() if p.requires_grad and n.startswith(UPSTREAM_HEAD_KEYS)
    ]
    if scene_heads is not None:
        head_params += list(scene_heads.parameters())
    groups = [{"params": refiner, "lr": LR_REFINER, "name": "refiner path"}]
    if head_params:
        groups.append({"params": head_params, "lr": LR_HEADS, "name": "heads"})
    return torch.optim.Adam(groups)


def forward(model, scene_heads, features: dict, keep: bool = False, detach_heads: bool = False) -> dict:
    """The vendor's forward plus the heads. `keep` also returns the image tokens and each pass's
    query features, for looking at. `detach_heads` feeds the heads a detached copy of the query
    features, so their losses train the heads and do not reach the refiner (a warm-up)."""
    captured = {"passes": []}

    def scorer_input(module, args):
        captured["proposals"], captured["query_features"] = args[0], args[1]

    def tokens(module, args, output):
        captured["tokens"] = output[0]  # (cameras, 512, B, 256)
        captured["spatial"], captured["level"] = output[1], output[2]  # what `cached_forward` needs beside

    def refined(module, args, output):
        captured["passes"].append(output[0])

    hooks = [model.scorer.register_forward_pre_hook(scorer_input)]
    if keep:
        hooks.append(model._backbone.register_forward_hook(tokens))
        hooks.append(model._trajectory_head[0].register_forward_hook(refined))
    try:
        # the vendor's forward slices features["lidar2img"] in place, so it gets a shallow copy
        out = model(dict(features))
    finally:
        for hook in hooks:
            hook.remove()
    query = captured["query_features"]
    head_input = query.detach() if detach_heads else query
    outputs = heads.run_upstream_heads(model.scorer, head_input)
    if scene_heads is not None:
        outputs.update(scene_heads(head_input))
    result = {"out": out, "query_features": query, "scorer_proposals": captured["proposals"], "aux": outputs}
    if keep:
        for key in ("tokens", "spatial", "level", "passes"):
            result[key] = captured[key]
    return result


def release_teachers(out: dict, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """The pseudo-teachers: the k proposals the model's own scorer ranks highest, detached.

    Taken from the released model before any training, they are the release's own best plans,
    so the teacher term pulls the candidates toward what it already did well.
    """
    order = out["pdm_score"].argsort(dim=1, descending=True)[:, :k]
    teachers = torch.stack([out["proposals"][b, order[b]] for b in range(len(order))]).detach()
    return teachers, torch.full((len(order),), k, device=teachers.device)


def loss(result: dict, labels, target, teachers, n_teachers, config, teacher_weight: float = 0.5) -> dict:
    """Every term of the total, and the total. `labels` None trains the planning term alone (arm C)."""
    from gallops.drivejepa_b2d.train_generator import mtd_loss

    terms = {
        "planning": mtd_loss(
            result["out"]["proposal_list"], target, teachers, n_teachers, config.prev_weight, teacher_weight
        )
    }
    total = terms["planning"]
    if labels is not None:
        weight = heads.weights(config)
        for name, value in heads.losses(result["aux"], labels, config).items():
            terms[name] = value
            total = total + weight[name] * value
    terms["total"] = total
    return terms


def _groups(model, scene_heads) -> dict:
    named = dict(model.named_parameters())
    groups = {
        "refiner": [p for n, p in named.items() if n.startswith("_trajectory_head")],
        "init_feature": [named["init_feature.weight"]],
        "hist_encoding": [p for n, p in named.items() if n.startswith("hist_encoding")],
        "encoder (ViT + projector)": [p for n, p in named.items() if n.startswith("_backbone")],
        "scorer (picks the plan)": [
            p for n, p in named.items() if n.startswith("scorer.") and not n.startswith(UPSTREAM_HEAD_KEYS)
        ],
        "map head (upstream)": [p for n, p in named.items() if n.startswith("scorer.map_head.")],
        "box head (upstream)": [p for n, p in named.items() if n.startswith("scorer._agent_head.")],
    }
    if scene_heads is not None:
        groups["lead + lane-change heads (new)"] = list(scene_heads.parameters())
    return {k: v for k, v in groups.items() if v}


def check_heads(model, result: dict) -> float:
    """Largest difference between upstream's heads as `forward` runs them and as `Scorer.forward`
    runs them in training mode, on the same query features. Should be 0."""
    scorer, was_training = model.scorer, model.scorer.training
    scorer.train()
    try:
        with torch.no_grad():
            _, _, _, bev_map, agent_states, agent_labels = scorer(
                result["scorer_proposals"], result["query_features"]
            )
    finally:
        scorer.train(was_training)
    pairs = [(bev_map, "bev"), (agent_states, "agent_states"), (agent_labels, "agent_labels")]
    return max(float((a - result["aux"][key]).abs().max()) for a, key in pairs if a is not None)


def _checksum(params) -> float:
    return float(sum(p.detach().double().sum() for p in params))


def train_step(model, scene_heads, opt, features, labels, target, teachers, n_teachers, config) -> dict:
    """One step. Returns the loss terms, and per parameter group its size, gradient and update.

    A trainable group's update is measured exactly; a frozen group's (the 304 M-parameter encoder
    among them) by a checksum before and after, which is enough to show it did not move.
    """
    groups = _groups(model, scene_heads)
    trainable = {name for name, params in groups.items() if all(p.requires_grad for p in params)}
    before = {name: [p.detach().clone() for p in groups[name]] for name in trainable}
    sums = {name: _checksum(params) for name, params in groups.items() if name not in trainable}
    opt.zero_grad(set_to_none=True)
    result = forward(model, scene_heads, features)
    terms = loss(result, labels, target, teachers, n_teachers, config)
    terms["total"].backward()
    grad = {}
    for name, params in groups.items():
        grads = [p.grad for p in params if p.grad is not None]
        grad[name] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else None
    opt.step()
    report = {}
    for name, params in groups.items():
        if name in trainable:
            pairs = zip(params, before[name])  # noqa: B905 -- Python 3.9
            update = float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in pairs)))
        else:
            update = abs(_checksum(params) - sums[name])
        report[name] = {
            "tensors": len(params),
            "parameters": sum(p.numel() for p in params),
            "trainable": name in trainable,
            "grad_norm": grad[name],
            "update_norm": update,
        }
    return {"terms": {k: float(v) for k, v in terms.items()}, "groups": report, "result": result}


def gradient_split(model, scene_heads, features, labels, target, teachers, n_teachers, config) -> dict:
    """Where each part of the loss reaches, without stepping.

    Per parameter group: the gradient norm of the planning term alone, of the weighted auxiliary
    terms together, and of each weighted auxiliary term alone; and, on the refiner, the cosine
    between the planning and the auxiliary gradient (positive: they pull the same way).
    """
    groups = {name: params for name, params in _groups(model, scene_heads).items() if params[0].requires_grad}
    params = [p for ps in groups.values() for p in ps]
    result = forward(model, scene_heads, features)
    terms = loss(result, labels, target, teachers, n_teachers, config)
    weight = heads.weights(config)
    parts = {"planning": terms["planning"], "auxiliary (all)": terms["total"] - terms["planning"]}
    parts.update({f"aux: {k}": weight[k] * terms[k] for k in heads.weights(config) if k in terms})
    norms, refiner = {}, {}
    for label, term in parts.items():
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        by_param = dict(zip(map(id, params), grads))  # noqa: B905 -- Python 3.9
        norms[label] = {}
        for name, ps in groups.items():
            g = [by_param[id(p)] for p in ps if by_param[id(p)] is not None]
            norms[label][name] = float(torch.sqrt(sum((x.float() ** 2).sum() for x in g))) if g else 0.0
        if label in ("planning", "auxiliary (all)"):
            refiner[label] = torch.cat(
                [
                    torch.zeros_like(p).flatten() if by_param[id(p)] is None else by_param[id(p)].flatten()
                    for p in groups["refiner"]
                ]
            )
    a, b = refiner["planning"], refiner["auxiliary (all)"]
    cosine = float((a @ b) / (a.norm() * b.norm() + 1e-12))
    return {"norms": norms, "refiner_cosine": cosine}
