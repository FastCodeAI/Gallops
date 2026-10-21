"""Fine-tune Drive-JEPA's refiner on nuScenes train, with or without auxiliary heads: arms C, T1, T2.

    <drive-jepa python> -m gallops.drivejepa_aux.train --cache <cache> --extract <extract> \\
        --heads all --seed 0 --train-run aux-t2-s0

docs/drivejepa-aux-heads.md is the plan; this is its step 4. One recipe for every arm, and the only
switch is which heads are on:

    --heads none      arm C:  the planning loss alone -- what fine-tuning on nuScenes does by itself
    --heads upstream  arm T1: + upstream's BEV map and box heads (its code, switched on)
    --heads all       arm T2: + the lead-vehicle and lane-change heads

The same in every arm: the release as the start; the refiner path (refiner, query embedding, ego
encoder) trains at 2e-5 and the heads at 1e-4; the encoder and the scorer are frozen; the planning
loss is the Bench2Drive generator stage's -- expert winner-takes-all plus 0.5 x the release's own
top 2, on every pass -- imported; Adam, batch 32 with the batch fix, 10 epochs. **The epoch kept is
the one with the lowest held-out PLANNING loss**, so C and T are chosen by one rule, and the release
itself is epoch 0: a run that never beats it keeps it. For one seed every arm sees the batches in
the same order -- the order has its own generator, not the one that initialises the heads -- so arms
are paired step for step.

It trains from the cache (`cache.py`): the frozen encoder's tokens, so an epoch takes under a minute.
The preflight (`preflight.py`) runs after the model is built and before the first step, and holds the
cache to a live rebuild and the trainer's batched path to the live batch-1 one. There is no way to
skip it.

Each run writes under {runs}/drivejepa_aux/runs/<train_run>/:
    metrics.jsonl      the mean losses every 25 steps, the gradient pull on the refiner every 100,
                       every epoch's held-out losses
    epoch_NN.pt        what each epoch trained (the refiner path and the heads, 32 MB)
    heldout.npz        the release's and the kept epoch's held-out measurements, row by row
    summary.json       the same, aggregated with scene-clustered standard errors
and the kept epoch, served, under {weights}/drivejepa_aux/<train_run>/step_<n>/ (`publish.py`).

Runs in Drive-JEPA's interpreter (Python 3.9), with its cwd and environment (`step.enter`).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np

from gallops import checkpoint as sidecar
from gallops.config import REPO_ROOT, SiteConfig
from gallops.drivejepa_aux import cache, metrics, publish, split

ARMS = {"none": "C", "upstream": "T1", "all": "T2"}
#: Rows per held-out forward: no gradient, so as large as is quick.
EVAL_BATCH = 64
#: Steps between logged means, and between measurements of the heads' pull on the refiner.
LOG_EVERY = 25
PULL_EVERY = 100
#: Held-out rows the sight check rebuilds live, for the release and for the kept weights.
SIGHT_ROWS = 16
#: Held-out rows the reloaded file must reproduce exactly.
PROBE_ROWS = 4


def run_directory(cfg: SiteConfig, train_run: str) -> Path:
    return Path(cfg.resolve("{runs}")) / "drivejepa_aux" / "runs" / train_run


def held_out_losses(model, scene_heads, config, data, rows, spatial, level, device, with_labels) -> dict:
    """Every loss term's mean over `rows`, no gradient. `planning` is what chooses the epoch."""
    import torch

    from gallops.drivejepa_aux import step

    sums: dict = {}
    with torch.no_grad():
        for start in range(0, len(rows), EVAL_BATCH):
            index = rows[start : start + EVAL_BATCH]
            fed = cache.batch(data, index, device, with_labels)
            result = step.cached_forward(model, scene_heads, fed["inputs"], spatial, level)
            terms = step.loss(
                result, fed["labels"], fed["target"], fed["teachers"], fed["n_teachers"], config
            )
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + float(value) * len(index)
    return {key: value / len(rows) for key, value in sums.items()}


def measure(model, scene_heads, data, rows, spatial, level, device, with_heads: bool) -> dict:
    """Row by row on `rows`: the picked plan and the metrics of `metrics`, and each head's output
    reduced to what its metric needs."""
    import torch

    from gallops.drivejepa_aux import heads, step

    labels, target = data["labels"], data["inputs"]["target"]
    plans, best = [], []
    bev_counts, boxes, lead, lane = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), EVAL_BATCH):
            index = rows[start : start + EVAL_BATCH]
            fed = cache.batch(data, index, device, with_labels=False)
            result = step.cached_forward(model, scene_heads, fed["inputs"], spatial, level, score=True)
            out, aux = result["out"], result["aux"]
            plans.append(out["trajectory"].float().cpu().numpy())
            best.append(metrics.min_ade(out["proposals"].float().cpu().numpy(), target[index]))
            if not with_heads:
                continue
            if "bev" in aux:
                predicted = aux["bev"].argmax(1).cpu().numpy().astype(np.uint8)
                bev_counts.append(metrics.bev_counts(predicted, labels["bev"][index]))
            if "agent_states" in aux:
                states = aux["agent_states"].float().cpu().numpy()
                logits = aux["agent_labels"].float().cpu().numpy()
                for j, row in enumerate(index):
                    tp, fp, fn, errors = metrics.box_matches(
                        states[j], logits[j], labels["agent_states"][row], labels["agent_valid"][row]
                    )
                    boxes.append((tp, fp, fn, sum(errors)))
            if "lead" in aux:
                readout = heads.lead_readout(aux["lead"].float())
                lead.append(
                    np.stack(
                        [readout[k].cpu().numpy() for k in ("present_probability", "gap_m", "closing_mps")], 1
                    )
                )
            if "lane_change" in aux:
                lane.append(aux["lane_change"].argmax(1).cpu().numpy())
    plan = np.concatenate(plans)
    out = {
        "rows": np.asarray(rows),
        "scene": labels["scene"][rows],
        "plan": plan,
        "l2": metrics.l2(plan, target[rows]),
        "min_ade": np.concatenate(best),
        "collided": metrics.collided(plan, [cache.future_boxes(labels, r) for r in rows]),
        "off_road": metrics.off_road(plan, cache.road(labels, rows)),
    }
    if bev_counts:
        out["bev_counts"] = np.concatenate(bev_counts)
    if boxes:
        out["boxes"] = np.asarray(boxes, dtype=float)
    if lead:
        out["lead"] = np.concatenate(lead)
    if lane:
        out["lane"] = np.concatenate(lane)
    return out


def baselines(data, train_rows, held_rows) -> dict:
    """What a model that never looks would score on the held-out rows, from the train rows' labels,
    and the human's own path under the plan's metrics (its collision rate is the metric's floor)."""
    from gallops.drivejepa_aux import bev

    labels, expert = data["labels"], data["inputs"]["target"]
    prior = metrics.prior_bev(labels["bev"][train_rows])
    counts = metrics.bev_counts(
        np.broadcast_to(prior, labels["bev"][held_rows].shape), labels["bev"][held_rows]
    )
    present = labels["lead_present"][held_rows].astype(bool)
    train_gap = float(
        np.nanmean(labels["lead_gap"][train_rows][labels["lead_present"][train_rows].astype(bool)])
    )
    lane = labels["lane_change"][held_rows]
    scene = labels["scene"][held_rows]
    return {
        "bev_iou_prior": dict(zip(bev.CLASSES, map(float, metrics.iou(counts)))),  # noqa: B905 -- 3.9
        "lead_present_accuracy_never": float(1.0 - present.mean()),
        "lead_gap_mae_train_mean_m": float(np.abs(labels["lead_gap"][held_rows][present] - train_gap).mean()),
        "lane_f1_always_keep": metrics.lane_f1(np.zeros(len(lane), int), lane),
        "human_collision_3s": _stat(
            metrics.collided(expert[held_rows], [cache.future_boxes(labels, r) for r in held_rows]), scene
        ),
        "human_off_road_3s": _stat(metrics.off_road(expert[held_rows], cache.road(labels, held_rows)), scene),
    }


def plain(value):
    """`value` with every NaN or infinity as None: the sidecar and summary.json stay strict JSON (an
    IoU of a class no held-out row has is NaN, and `json` would write the non-standard `NaN`)."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _stat(values, clusters) -> dict:
    mean, se, scenes = metrics.clustered(values, clusters)
    return {"mean": mean, "se": se, "scenes": scenes}


def summarize(measured: dict, data: dict) -> dict:
    """`measure`'s rows, aggregated: means with scene-clustered standard errors, and the heads' scores."""
    from gallops.drivejepa_aux import bev

    scene = measured["scene"]
    labels = data["labels"]
    rows = measured["rows"]
    out = {f"l2_{name}": _stat(measured["l2"][:, k], scene) for name, k in metrics.L2_WAYPOINTS.items()}
    out["min_ade_3s"] = _stat(measured["min_ade"], scene)
    out["collision_3s"] = _stat(measured["collided"], scene)
    out["off_road_3s"] = _stat(measured["off_road"], scene)
    if "bev_counts" in measured:
        iou = metrics.iou(measured["bev_counts"])
        out["bev_iou"] = dict(zip(bev.CLASSES, map(float, iou)))  # noqa: B905 -- 3.9
    if "boxes" in measured:
        tp, fp, fn, error = measured["boxes"].sum(0)
        out["box_f1"] = metrics.f1(tp, fp, fn)
        out["box_centre_error_m"] = float(error / tp) if tp else float("nan")
        out["box_counts"] = {"tp": int(tp), "fp": int(fp), "fn": int(fn)}
    if "lead" in measured:
        present = labels["lead_present"][rows].astype(bool)
        out["lead_present_accuracy"] = float(((measured["lead"][:, 0] > 0.5) == present).mean())
        out["lead_gap_mae_m"] = float(
            np.abs(measured["lead"][present, 1] - labels["lead_gap"][rows][present]).mean()
        )
    if "lane" in measured:
        out["lane_f1"] = metrics.lane_f1(measured["lane"], labels["lane_change"][rows])
    return out


def gradient_pull(terms: dict, refiner: list) -> dict:
    """How hard the heads pull on the refiner, against the planning loss, and which way: the norms of
    the two gradients on the refiner's parameters and their cosine. No step is taken."""
    import torch

    def flat(loss):
        grads = torch.autograd.grad(loss, refiner, retain_graph=True, allow_unused=True)
        return torch.cat(
            [torch.zeros_like(p).flatten() if g is None else g.flatten() for g, p in zip(grads, refiner)]  # noqa: B905
        )  # noqa: B905

    planning, auxiliary = flat(terms["planning"]), flat(terms["total"] - terms["planning"])
    a, b = float(planning.norm()), float(auxiliary.norm())
    return {
        "planning_norm": a,
        "aux_norm": b,
        "ratio": b / a if a else float("nan"),
        "cosine": float(planning @ auxiliary) / (a * b) if a and b else float("nan"),
    }


def sight(model, extract, data, rows, root, device) -> dict:
    """Swerve's sight check, live, on held-out rows: blanking the camera must move the plan."""
    import torch

    from gallops.drivejepa_aux import inputs
    from gallops.train.preflight import sight_check

    labels = data["labels"]
    by_key = {(t["scene"], t["keyframe"]): t for name in extract["ticks"] for t in extract["ticks"][name]}
    ticks = [by_key[(str(labels["scene"][r]), int(labels["keyframe"][r]))] for r in rows]

    def dataset(chosen, device_):
        return inputs.TablesTickDataset(chosen, device_, extract["scenes"], str(root))

    report, problems = sight_check(model, dataset, ticks, torch, device)
    return {**report, "passes": not problems, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, required=True, help="the cache directory (labels/ and tokens/)")
    parser.add_argument(
        "--extract", type=Path, required=True, help="the `nusc extract` the cache was made from"
    )
    parser.add_argument(
        "--heads", choices=tuple(ARMS), required=True, help="none = C, upstream = T1, all = T2"
    )
    parser.add_argument("--seed", type=int, default=0, help="the heads' initial weights and the batch order")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="steps during which the heads read a detached copy of the query features, so they learn "
        "while the refiner sees the planning loss alone (0: joint from the first step)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="a smoke run over N train rows (and N // 4 held out)"
    )
    parser.add_argument("--nuscenes", type=Path, help="default: gallops.toml's {data.nuscenes}")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch < 4:
        raise SystemExit("--batch must be at least 4: the preflight's loss parity runs the agent's own loss")

    from gallops.drivejepa_b2d.sidecar import refuse_existing_run, sha256
    from gallops.train.swerve import NUSCENES_TRAIN_SCENES, git_sha, head_moved

    cfg = SiteConfig.load()
    # absolute before `step.enter` changes into navsim_v2/
    cache_dir, extract_dir = args.cache.resolve(), args.extract.resolve()
    root = (args.nuscenes or Path(cfg.resolve("{data.nuscenes}"))).resolve()
    run = run_directory(cfg, args.train_run)
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"{run} already holds a fit; one fit is one run -- choose a new --train-run")
    refuse_existing_run(cfg.run_dir(publish.MODEL, args.train_run))
    loaded = git_sha(REPO_ROOT)  # the commit whose modules this process imported

    from gallops.drivejepa_aux import step

    release = step.enter(cfg)
    import torch

    from gallops.drivejepa_aux import inputs
    from gallops.drivejepa_aux import preflight as aux_preflight

    print(f"loading the cache {cache_dir} ...", flush=True)
    data = cache.load(cache_dir)
    held = split.load()
    built = data["inputs"]["built"]
    train_rows, held_rows = split.partition(data["labels"]["scene"], held["held_out"])
    train_rows, held_rows = train_rows[built[train_rows]], held_rows[built[held_rows]]
    if args.limit:
        # evenly spaced, not the first N: rows are in scene order, and a head slice is a few scenes
        train_rows = train_rows[np.linspace(0, len(train_rows) - 1, args.limit).round().astype(int)]
        held_rows = held_rows[
            np.linspace(0, len(held_rows) - 1, max(args.batch, args.limit // 4)).round().astype(int)
        ]
    aux = args.heads != "none"
    print(
        f"  {len(train_rows)} train rows, {len(held_rows)} held out; "
        f"arm {ARMS[args.heads]}, seed {args.seed}",
        flush=True,
    )

    with contextlib.redirect_stdout(io.StringIO()):  # the vendor prints the whole ViT as it builds
        agent, model, scene_heads, config = step.build_model(
            release, args.device, aux=aux, seed=args.seed, scene=args.heads == "all"
        )
    extract = cache.load_extract(extract_dir)
    inputs.use_calibration(extract_dir / "calib")

    # Before any step: the code is committed and its tests pass, the cache is what a live pass gives,
    # the trainer's batched path reproduces the release's live answer, and the labels are the labeller's.
    checked = aux_preflight.run(
        REPO_ROOT, model, scene_heads, agent, config, data, train_rows, held_rows, extract, root, args.device
    )
    if checked["code"]["sha"] != loaded:
        raise SystemExit(
            f"HEAD moved from {loaded} to {checked['code']['sha']} while the model loaded; the "
            "preflight checked a tree this process did not import -- start again"
        )

    spatial = torch.as_tensor(data["inputs"]["spatial"], device=args.device)
    level = torch.as_tensor(data["inputs"]["level"], device=args.device)
    optimizer = step.make_optimizer(model, scene_heads)
    refiner = [p for n, p in model.named_parameters() if n.startswith("_trajectory_head") and p.requires_grad]
    run.mkdir(parents=True, exist_ok=True)
    log = (run / "metrics.jsonl").open("x")

    def record(kind: str, **fields) -> None:
        log.write(json.dumps({"kind": kind, "time": round(time.time(), 1), **fields}) + "\n")
        log.flush()

    record(
        "start",
        run=args.train_run,
        arm=ARMS[args.heads],
        seed=args.seed,
        train_rows=len(train_rows),
        held_out_rows=len(held_rows),
        epochs=args.epochs,
        batch=args.batch,
        warmup_steps=args.warmup_steps,
    )

    # The release, before any step: its held-out answer and losses are epoch 0, which a run must beat.
    print("measuring the release on the held-out rows ...", flush=True)
    release_measured = measure(model, None, data, held_rows, spatial, level, args.device, with_heads=False)
    sight_rows = held_rows[
        np.linspace(0, len(held_rows) - 1, min(SIGHT_ROWS, len(held_rows))).round().astype(int)
    ]
    release_sight = sight(model, extract, data, sight_rows, root, args.device)
    losses = held_out_losses(model, scene_heads, config, data, held_rows, spatial, level, args.device, aux)
    history = [{"epoch": 0, "step": 0, "held_out": losses}]
    record("val", epoch=0, step=0, **losses)
    best = (losses["planning"], 0, 0, step.trainable_state(model, scene_heads))
    print(f"  epoch 0 (the release): held-out planning {losses['planning']:.4f}", flush=True)

    rng = np.random.default_rng(args.seed)  # the batch order, and nothing else: paired across arms
    steps = 0
    steps_per_epoch = len(train_rows) // args.batch
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train_rows)
        started, window = time.time(), []
        for start in range(0, steps_per_epoch * args.batch, args.batch):
            fed = cache.batch(data, order[start : start + args.batch], args.device, with_labels=aux)
            result = step.cached_forward(
                model, scene_heads, fed["inputs"], spatial, level, detach_heads=steps < args.warmup_steps
            )
            terms = step.loss(
                result, fed["labels"], fed["target"], fed["teachers"], fed["n_teachers"], config
            )
            if aux and steps % PULL_EVERY == 0:
                record("pull", epoch=epoch, step=steps, **gradient_pull(terms, refiner))
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            optimizer.step()
            steps += 1
            window.append({k: float(v) for k, v in terms.items()})
            if steps % LOG_EVERY == 0:
                means = {k: float(np.mean([w[k] for w in window])) for k in window[0]}
                record("train", epoch=epoch, step=steps, **means)
                window = []
        losses = held_out_losses(
            model, scene_heads, config, data, held_rows, spatial, level, args.device, aux
        )
        state = step.trainable_state(model, scene_heads)
        torch.save(state, run / f"epoch_{epoch:02d}.pt")
        minutes = (time.time() - started) / 60
        history.append({"epoch": epoch, "step": steps, "held_out": losses, "minutes": round(minutes, 2)})
        record("val", epoch=epoch, step=steps, **losses)
        if losses["planning"] < best[0]:
            best = (losses["planning"], epoch, steps, state)
        print(
            f"  epoch {epoch}: held-out planning {losses['planning']:.4f} total {losses['total']:.4f} "
            f"({minutes:.1f} min)",
            flush=True,
        )

    _, kept_epoch, kept_step, kept_state = best
    step.load_trainable(model, scene_heads, kept_state)
    print(f"kept epoch {kept_epoch} (step {kept_step}); measuring it on the held-out rows ...", flush=True)
    kept_measured = measure(model, scene_heads, data, held_rows, spatial, level, args.device, with_heads=aux)
    kept_sight = sight(model, extract, data, sight_rows, root, args.device)
    summary = {
        "arm": ARMS[args.heads],
        "seed": args.seed,
        "kept_epoch": kept_epoch,
        "held_out_rows": len(held_rows),
        "release": {**summarize(release_measured, data), "sight": release_sight},
        "kept": {**summarize(kept_measured, data), "sight": kept_sight},
        "baselines": baselines(data, train_rows, held_rows),
        "history": history,
    }
    np.savez(
        run / "heldout.npz",
        **{f"release_{k}": v for k, v in release_measured.items()},
        **{f"kept_{k}": v for k, v in kept_measured.items()},
    )
    summary = plain(summary)
    (run / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    record("done", kept_epoch=kept_epoch, kept_step=kept_step)
    log.close()
    if not kept_sight["passes"]:
        raise SystemExit(f"sight: the kept weights do not read the camera: {kept_sight['problems']}")

    probe_rows = held_rows[:PROBE_ROWS]

    def probe(weights: Path) -> dict:
        """The written file, loaded as paddock's policy loads it, must be the model that trained."""
        from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
        from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

        with contextlib.redirect_stdout(io.StringIO()):
            served_agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=str(weights))
            served_agent.initialize()
        served = served_agent._pad_model.to(args.device).eval()
        fed = cache.batch(data, probe_rows, args.device, with_labels=False)
        with torch.no_grad():
            ours = step.cached_forward(model, None, fed["inputs"], spatial, level, score=True)["out"]
            theirs = step.cached_forward(served, None, fed["inputs"], spatial, level, score=True)["out"]
        gap = float((ours["proposals"] - theirs["proposals"]).abs().max())
        same_pick = bool((ours["pdm_score"].argmax(1) == theirs["pdm_score"].argmax(1)).all())
        del served, served_agent
        torch.cuda.empty_cache()
        if gap != 0.0 or not same_pick:
            raise SystemExit(
                f"PROBE FAILED: the written file's proposals differ by {gap:.2e} m (same pick: {same_pick})"
            )
        return {
            "rows": len(probe_rows),
            "max_proposal_gap_m": gap,
            "how": "reloaded by DriveJEPAAgent.initialize, as paddock's policy loads it; "
            "held-out rows, cached path",
        }

    served_params = sum(
        p.numel() for n, p in model.named_parameters() if not n.startswith(publish.UPSTREAM_HEAD_KEYS)
    )
    train_scenes = sorted(set(data["labels"]["scene"][train_rows]))
    card = sidecar.Checkpoint(
        model=publish.MODEL,
        step=kept_step,
        train_git_sha=checked["code"]["sha"],
        train_run=args.train_run,
        trained_on=(
            sidecar.TrainedOn("nuscenes", "train", round(len(train_scenes) / NUSCENES_TRAIN_SCENES, 6)),
            # inherited: the release was trained on navtrain, so a NAVSIM score of this is not zero-shot
            sidecar.TrainedOn("navsim", "navtrain"),
        ),
        selection_split="nuscenes/train: the held-out scenes of recipes/drivejepa_aux/heldout_scenes.json",
        params_m=round(served_params / 1e6, 2),
        precision="fp32",
        notes={
            "arm": ARMS[args.heads],
            "heads": args.heads,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": {"refiner path": step.LR_REFINER, "heads": step.LR_HEADS},
            "warmup_steps": args.warmup_steps,
            "kept_epoch": kept_epoch,
            "steps_per_epoch": steps_per_epoch,
            "selection": "lowest held-out planning loss (mtd_loss) over epochs 0..N, epoch 0 the release; "
            "one rule for every arm",
            "loss": "drivejepa_b2d.train_generator.mtd_loss (winner-takes-all + 0.5 x the release's top-2, "
            "every pass)" + (" + heads.losses, weighted by heads.weights" if aux else ""),
            "init": {"checkpoint": release, "sha256": data["manifest"]["tokens"]["checkpoint_sha256"]},
            "served_by": "paddock's drivejepa adapter, "
            f"DRIVEJEPA_CHECKPOINT=<this directory>/{publish.WEIGHTS_FILE}",
            "heldout": summary,
            "preflight": plain(checked),
            "paddock_parity": aux_preflight.PADDOCK_PARITY,
            "cache": data["manifest"],
            "heldout_list_sha256": sha256(split.HELDOUT_FILE),
            "train_scenes": len(train_scenes),
            "rows_without_tokens": int((~built).sum()),
            "limit": args.limit or None,
            "run_dir": str(run),
            "deterministic": "no: the deformable attention's backward adds atomically; "
            "the seed fixes the batch order and the heads' initial weights",
            **head_moved(checked["code"]["sha"], REPO_ROOT),
        },
    )
    out_dir = cfg.checkpoint_dir(publish.MODEL, args.train_run, kept_step)
    written = publish.write(out_dir, card, Path(release), kept_state, probe)
    print(
        f"wrote {out_dir} (sha256 {written.weights_sha256[:16]}...), the run's measurements in {run}",
        flush=True,
    )


if __name__ == "__main__":
    main()
