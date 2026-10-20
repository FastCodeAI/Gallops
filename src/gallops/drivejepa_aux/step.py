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

import torch

from gallops.drivejepa_aux import bev, heads, targets

#: Prefixes of upstream's switched-on heads inside the model, the only keys the release lacks.
UPSTREAM_HEAD_KEYS = ("scorer.map_head.", "scorer._agent_head.")
#: The refiner path trains at the Bench2Drive generator stage's rate; the heads, which start from
#: random weights, at the rate the release itself was trained at (DriveJEPAAgent lr).
LR_REFINER = 2e-5
LR_HEADS = 1e-4


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


def build_model(checkpoint: str, device: str, aux: bool = True, seed: int = 0):
    """The released model with upstream's heads switched on, and the new heads.

    Loads the released state dict with strict=False and refuses unless the only missing keys are
    the two switched-on heads' and nothing is unexpected. Applies the batch fix, puts the model in
    eval mode, and leaves requires_grad on the refiner path and the heads alone.
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
    scene_heads = heads.SceneHeads(config.tf_d_model).to(device).eval() if aux else None
    return agent, model, scene_heads, config


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
        result["tokens"] = captured["tokens"]
        result["passes"] = captured["passes"]
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
