"""H15's preflight: what a trainer computes equals what was recorded, checked before its first step.

CLAUDE.md: no run trains until the preflight passes, and there is no way to skip it. A check that
can be skipped is skipped the one time it matters. Every trainer in this package calls `run` after
building its data and before its first optimizer step; a failure lists every problem.

- **code:** `src/` and `tests/` committed, the suite green under the training interpreter
  (`gallops.train.preflight.code_checks`), so the sha in the run's files names the code that ran.
- **data:** the splits re-drawn and unedited; the token cache and the labels hold the same rows in the
  same order; the run's rows fall in the splits they claim.
- **labels:** `navsim_data.check_labels` on a few windows, held to NAVSIM's and nuPlan's own code.
- **inputs:** a live batch-1 forward of a few select rows, inputs built by the vendor's code, equals
  the cache: tokens to fp16 rounding, and the release's pick exactly. With `vit_dir` (a side-car
  on the ViT's own output), the same forward's ViT output equals that cache too, within
  tokens.VIT_LIMITS: it is the tap the side-car reads, so it is the one held to the live model.
- **projection:** a point 20 m straight ahead lands near the token image's centre column through
  the cached CAM_F0 projection; a wrong camera slot or a wrong TO_512 puts it elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
#: fp16 rounding of tokens up to about 10 in magnitude (the smoke pass's largest was 10.2).
TOKEN_ATOL = 1e-2


def run(cfg, tokens_dir: Path, labels_dir: Path, rows: dict, live_rows: int = 6, vit_dir=None) -> dict:
    """All checks; returns the report for the run's files or raises SystemExit listing every problem.

    `rows` maps the split names a run uses (train, select, report, tap) to the row arrays it reads;
    `vit_dir`, the ViT-out cache (tokens.py --vit-out-all) when that is what the run reads.
    """
    from gallops.drivejepa_perception import env, navsim_data, targets
    from gallops.train.preflight import code_checks

    report, problems = {}, []
    report["code"], found = code_checks(REPO)
    problems += found

    entered = env.enter(cfg)
    report["entered"] = entered
    tmanifest = json.loads((Path(tokens_dir) / "manifest.json").read_text())
    lmanifest = json.loads((Path(labels_dir) / "manifest.json").read_text())
    labels = np.load(Path(labels_dir) / "labels.npz")
    logs = lmanifest["logs"]
    token_logs = np.array([logs[i] for i in labels["log_id"]])
    splits = navsim_data.load_splits(
        REPO / "recipes/drivejepa_perception/splits.json",
        navsim_data.navtrain_filter(Path(entered["code"]))["log_names"],
    )
    data = {"cache_rows": tmanifest["rows"], "label_rows": int(len(labels["token"]))}
    if tmanifest["rows"] != len(labels["token"]):
        problems.append(f"the token cache has {tmanifest['rows']} rows, the labels {len(labels['token'])}")
    if Path(tmanifest["labels"]).resolve() != Path(labels_dir).resolve():
        problems.append(f"the token cache was built from {tmanifest['labels']}, not {labels_dir}")
    if vit_dir is not None:
        vmanifest = json.loads((Path(vit_dir) / "manifest.json").read_text())
        data["vit_rows"] = vmanifest["rows"]
        if vmanifest["rows"] != len(labels["token"]):
            problems.append(
                f"the ViT-out cache has {vmanifest['rows']} rows, the labels {len(labels['token'])}"
            )
        if Path(vmanifest["labels"]).resolve() != Path(labels_dir).resolve():
            problems.append(f"the ViT-out cache was built from {vmanifest['labels']}, not {labels_dir}")
    for name, picked in rows.items():
        allowed = set(splits["train"]) if name in ("train", "tap") else set(splits[name])
        if name == "tap":
            allowed |= set(splits["select"])
        outside = sum(token_logs[r] not in allowed for r in picked)
        data[f"{name}_rows"] = int(len(picked))
        if outside:
            problems.append(f"{outside} of the {name} rows are from logs outside its split")
    report["data"] = data

    report["labels"] = {"problems": navsim_data.check_labels(cfg, Path(labels_dir), n_windows=8)}
    problems += report["labels"]["problems"]

    select_rows = np.asarray(rows.get("select", []))
    sample = (
        select_rows[np.linspace(0, len(select_rows) - 1, live_rows).astype(int)] if len(select_rows) else []
    )
    report["inputs"], found = _live_equals_cache(entered, tokens_dir, labels, token_logs, sample, vit_dir)
    problems += found

    l2i = np.load(Path(tokens_dir) / "lidar2img.npy", mmap_mode="r")
    centre_u = []
    for r in sample:
        h = targets.front_projection(np.asarray(l2i[r])) @ np.array([20.0, 0.0, 1.0, 1.0])
        centre_u.append(float(h[0] / h[2]))
    report["projection"] = {"u_of_20m_ahead": centre_u}
    if centre_u and max(abs(u - 256.0) for u in centre_u) > 64:
        problems.append(f"20 m straight ahead projects to u = {centre_u}, not near 256: wrong slot or TO_512")

    report["problems"] = problems
    if problems:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(problems))
    return report


def _live_equals_cache(entered: dict, tokens_dir: Path, labels, token_logs, sample, vit_dir=None) -> tuple:
    """Rebuild `sample` rows live (vendor inputs, batch 1) and compare them with the cache."""
    import torch
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder
    from navsim.common.dataclasses import AgentInput

    from gallops.drivejepa_perception import navsim_data, tokens

    cache = {k: np.load(Path(tokens_dir) / f"{k}.npy", mmap_mode="r") for k in ("tokens", "pick")}
    if vit_dir is not None:
        cache["vit_out"] = np.load(Path(vit_dir) / "vit_out.npy", mmap_mode="r")
    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.cuda().eval()
    captured = {}
    hooks = [model._backbone.register_forward_hook(lambda m, i, o: captured.__setitem__("tokens", o[0]))]
    if vit_dir is not None:
        hooks.append(
            model._backbone.img_backbone.register_forward_hook(
                lambda m, i, o: captured.__setitem__("vit_out", o)
            )
        )
    vit_diff = []
    builder = DriveJEPAFeatureBuilder(config=DriveJEPAConfig())
    sensors = DriveJEPAAgent.get_sensor_config(None)
    logs_dir = Path(entered["navsim_root"]) / "navsim_logs/trainval"
    blobs = Path(entered["navsim_root"]) / "sensor_blobs/trainval"
    worst, picks, problems = 0.0, 0, []
    with torch.no_grad():
        for r in sample:
            token = str(labels["token"][r])
            frames = navsim_data.log_windows(str(token_logs[r]), logs_dir, {token})[token]
            agent_input = AgentInput.from_scene_dict_list(frames, blobs, navsim_data.HISTORY, sensors)
            features = builder.compute_features(agent_input)
            out = model({k: v.unsqueeze(0).cuda() for k, v in features.items() if torch.is_tensor(v)})
            live = captured["tokens"][0, :, 0].float().cpu().numpy()
            worst = max(worst, float(np.abs(live - np.asarray(cache["tokens"][r], dtype=np.float32)).max()))
            picks += int(int(out["pdm_score"][0].argmax()) == int(cache["pick"][r]))
            if vit_dir is not None:
                live_vit = captured["vit_out"].reshape(512, 1024).float().cpu().numpy()
                vit_diff.append(np.abs(live_vit - np.asarray(cache["vit_out"][r], dtype=np.float32)))
    for hook in hooks:
        hook.remove()
    del model, agent
    torch.cuda.empty_cache()
    found = {"rows": [int(r) for r in sample], "token_max_diff": worst, "picks_equal": picks}
    if worst > TOKEN_ATOL:
        problems.append(f"live tokens differ from the cache by {worst:.3g} (limit {TOKEN_ATOL})")
    if picks != len(sample):
        problems.append(f"the live pick equals the cached one on {picks} of {len(sample)} rows")
    if vit_diff:
        diff = np.stack(vit_diff)
        found["vit_out"] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())}
        if diff.max() > tokens.VIT_LIMITS["max_abs"] or diff.mean() > tokens.VIT_LIMITS["mean_abs"]:
            problems.append(f"the live ViT output differs from the ViT-out cache: {found['vit_out']}")
    return found, problems


def run_scorer(cfg, args, data: dict, model_factory) -> dict:
    """B1's preflight: code, data, the PDM labels re-scored, and the release reproduced at step 0.

    Everything B1 reads is cached, so the checks hold the caches to what produced them:
    - the PDM labels it trains on equal a re-scoring by the vendored scorer, exactly (5 rows);
    - in the residual modes, the untrained model's logits and picks equal the release's cached ones
      exactly on 512 select rows: the identity the arms' comparison starts from;
    - an agents file (arms F, S) covers every row and names the side-car run that made it.
    """
    import torch

    from gallops.drivejepa_perception import env, navsim_data, pdm_labels
    from gallops.train.preflight import code_checks

    report, problems = {}, []
    report["code"], found = code_checks(REPO)
    problems += found
    entered = env.enter(cfg)
    navsim_data.load_splits(
        REPO / "recipes/drivejepa_perception/splits.json",
        navsim_data.navtrain_filter(Path(entered["code"]))["log_names"],
    )
    rows = data["rows"]
    report["data"] = {k: int(len(v)) for k, v in rows.items()}
    if any(len(v) == 0 for v in rows.values()):
        problems.append(f"a split has no scored rows: {report['data']}")
    if len(data["pdm"]) != len(data["logits"]):
        problems.append("the PDM labels and the token cache hold different numbers of rows")
    report["pdm_recheck"] = pdm_labels.check(cfg, Path(args.pdm), Path(args.tokens), Path(args.labels), 5)
    problems += report["pdm_recheck"]
    if args.arm in ("F", "S"):
        manifest = json.loads((Path(args.agents) / "manifest.json").read_text())
        report["agents"] = {k: manifest[k] for k in ("run", "epoch", "gallops_sha", "rows")}
        if manifest["rows"] != len(data["logits"]):
            problems.append(
                f"the agents file covers {manifest['rows']} rows, the cache {len(data['logits'])}"
            )
    if args.mode != "replace":
        from gallops.drivejepa_perception.train_scorer import batch_of

        model = model_factory().cuda().eval()
        sample = rows["select"][:512]
        with torch.no_grad():
            out = model(batch_of(data, sample, torch.device("cuda")))["logits"].cpu()
        cached = torch.from_numpy(data["logits"][sample])
        picks_equal = int((out[..., -1].argmax(1).numpy() == data["pick_release"][sample]).sum())
        report["identity"] = {
            "rows": int(len(sample)),
            "logits_equal": bool(torch.equal(out, cached)),
            "picks_equal": picks_equal,
        }
        if not torch.equal(out, cached) or picks_equal != len(sample):
            problems.append(f"the untrained model is not the release: {report['identity']}")
    report["problems"] = problems
    if problems:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(problems))
    return report


#: B2's identity limits: the refiner run from fp16 cached tokens against the cached release
#: proposals (computed from fp32 tokens in the live forward).
REFINER_LIMITS = {"proposals_m": 0.05, "picks_share": 0.95}


def run_refiner(cfg, args, data: dict, build_fn, forward_fn) -> dict:
    """B2's preflight: code, splits, the gathered teachers, and the untrained model reproducing the release.

    The gathered good anchors (`train_refiner.gather_anchors`) must equal their files on 16 random
    rows.

    With the adapter at zero, the refiner run from the fp16 token cache must give the cached release
    proposals to within REFINER_LIMITS on 64 select rows, and the frozen scorer's pick on at least
    95% of them. The measured differences go into the run's files.
    """
    import torch

    from gallops.drivejepa_b2d.head import Head, head_tensors_from_checkpoint
    from gallops.drivejepa_perception import env, navsim_data
    from gallops.train.preflight import code_checks

    report, problems = {}, []
    report["code"], found = code_checks(REPO)
    problems += found
    entered = env.enter(cfg)
    navsim_data.load_splits(
        REPO / "recipes/drivejepa_perception/splits.json",
        navsim_data.navtrain_filter(Path(entered["code"]))["log_names"],
    )
    from gallops.drivejepa_perception import train_refiner

    picked = np.random.default_rng(0).choice(len(data["anchor_count"]), 16, replace=False)
    labels = np.load(Path(args.labels) / "labels.npz")
    unequal = []
    for row in picked:
        path = train_refiner._anchor_file(
            train_refiner.anchors_root(cfg), data["token_logs"][row], str(labels["token"][row])
        )
        stored = data["anchors"][row, : data["anchor_count"][row]]
        if not np.array_equal(stored, train_refiner._read_anchors(path)):
            unequal.append(int(row))
    report["anchors"] = {"rows": 16, "unequal": unequal}
    if unequal:
        problems.append(f"the gathered anchors differ from their files on rows {unequal}")
    rows = data["rows"]["select"][:64]
    agent, model, adapter = build_fn()
    device = torch.device("cuda")
    model, adapter = model.to(device), adapter.to(device)
    model._backbone = model._backbone.cpu()
    if args.arm != "C":
        adapter.context = {
            k: torch.from_numpy(np.asarray(v[rows])).to(device) for k, v in data["agents"].items()
        }
        adapter.context["mask"] = adapter.context["mask"].bool()
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    head = head.to(device).eval()
    with torch.no_grad():
        props, bev = forward_fn(model, data, rows, device)
        pick = head(bev.reshape(len(rows), 32, 8, -1).amax(2))[..., -1].argmax(1).cpu().numpy()
    gap = float(np.abs(props[-1].float().cpu().numpy() - data["proposals"][rows]).max())
    share = float((pick == data["pick_release"][rows]).mean())
    report["identity"] = {"rows": int(len(rows)), "proposals_max_m": gap, "picks_equal_share": share}
    if gap > REFINER_LIMITS["proposals_m"] or share < REFINER_LIMITS["picks_share"]:
        problems.append(f"the untrained refiner does not reproduce the release: {report['identity']}")
    del model, agent
    torch.cuda.empty_cache()
    report["problems"] = problems
    if problems:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(problems))
    return report


#: B3.1: how far A1's select F1, read through the frozen branch on the fp16 block-20 cache, may sit
#: from the F1 it recorded on the fp16 ViT-out cache. One point; the measured gap is recorded.
VIT_LORA_F1_LIMIT = 0.01


def run_vit_lora(cfg, args, d: dict, vit, car, f1_recorded: float) -> dict:
    """B3.1's preflight: code, the cache through the frozen branch, and A1 reproduced through it.

    - The frozen branch (vit_lora.LateBranch without LoRA) run on 32 rows of the block-20 cache
      gives the ViT-out cache that A1 trained on, within tokens.VIT_LIMITS.
    - A1's side-car read through it on the select logs gives the select F1 A1 recorded at that
      epoch, within VIT_LORA_F1_LIMIT.
    """
    import torch

    from gallops.drivejepa_perception import tokens, train_sidecar, train_vit_lora, vit_lora
    from gallops.train.preflight import code_checks

    report, problems = {}, []
    report["code"], found = code_checks(REPO)
    problems += found
    manifest = json.loads((Path(args.block20) / "manifest.json").read_text())
    reference = np.load(Path(manifest["reference_vit_out"]) / "vit_out.npy", mmap_mode="r")
    device = torch.device("cuda")
    branch = vit_lora.LateBranch(vit, lora=False).to(device).eval()
    rows = d["rows"]["select"][np.linspace(0, len(d["rows"]["select"]) - 1, 32).astype(int)]
    with torch.no_grad():
        x = torch.from_numpy(np.stack([np.asarray(d["cache"][d["slot"](int(r))]) for r in rows])).to(device)
        tail = branch(x.float()).cpu().numpy()
    diff = np.abs(tail - np.asarray(reference[rows], dtype=np.float32))
    report["frozen_branch"] = {"rows": 32, "max_abs": float(diff.max()), "mean_abs": float(diff.mean())}
    if diff.max() > tokens.VIT_LIMITS["max_abs"] or diff.mean() > tokens.VIT_LIMITS["mean_abs"]:
        problems.append(f"the frozen branch does not give the ViT-out cache: {report['frozen_branch']}")
    model = train_vit_lora.Tail(branch, car).to(device)
    select = train_sidecar.Rows(d["cache"], d["slot"], d["rows"]["select"], d["projections"], d["targets"])
    metrics = train_sidecar.read(model, select, d["targets"], d["rows"]["select"], device, 25.0)
    f1 = train_sidecar.f1(metrics)
    report["a1_through_branch"] = {"select_f1": f1, "recorded": f1_recorded, "gap": abs(f1 - f1_recorded)}
    if abs(f1 - f1_recorded) > VIT_LORA_F1_LIMIT:
        problems.append(
            f"A1 through the frozen branch does not reproduce its select F1: {report['a1_through_branch']}"
        )
    del model, branch
    torch.cuda.empty_cache()
    report["problems"] = problems
    if problems:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(problems))
    return report
