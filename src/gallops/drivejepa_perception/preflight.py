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

**Part B's data preflight** (`run_data`, `python -m gallops.drivejepa_perception.preflight data`) holds
each of navtest's and R-clean's passes to the code and the data before anything reads them: the code
committed and the suite green; the list re-derived; the labels against NAVSIM's code on 8 windows;
a live batch-1 forward of 6 rows, inputs built by the vendor's code from that split's own data split,
equal to the cache (tokens, pick, the release's scores, the ViT output); the projection check; and the
PDM labels re-scored on 5 rows. `report_guard` refuses any row set that touches a report log: Part B
never reads them (B-0's protocol).
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


def report_guard(token_logs: np.ndarray, rows: dict, report_logs) -> list:
    """Every row set that touches a report log, named: Part B reads none of them."""
    report = set(report_logs)
    return [
        f"{name}: {int(np.isin(token_logs[np.asarray(picked)], list(report)).sum())} rows from report logs"
        for name, picked in rows.items()
        if len(picked) and np.isin(token_logs[np.asarray(picked)], list(report)).any()
    ]


def run_data(
    cfg, labels_dir: Path, tokens_dir: Path, vit_dir: Path, pdm_dir=None, live_rows: int = 6
) -> dict:
    """Part B's data preflight for a navtest or R-clean pass set (module docstring); writes nothing.

    Raises SystemExit listing every problem; returns the report otherwise.
    """
    from gallops.drivejepa_perception import env, navsim_data, pdm_labels, targets
    from gallops.train.preflight import code_checks

    report, problems = {}, []
    report["code"], found = code_checks(REPO)
    problems += found
    entered = env.enter(cfg)
    report["entered"] = entered
    lmanifest = json.loads((Path(labels_dir) / "manifest.json").read_text())
    source = navsim_data.source_of(lmanifest)
    if source == "navtrain":
        problems.append("run_data reads Part B's own sources; navtrain passes are held by `run`")
    spec = navsim_data.source_spec(Path(entered["code"]), Path(entered["navsim_root"]), source)
    labels = np.load(Path(labels_dir) / "labels.npz")
    token_logs = np.array(lmanifest["logs"])[labels["log_id"]]
    if sorted(set(token_logs.tolist())) != sorted(spec["logs"]):
        problems.append(f"the labels' logs are not the {source} list's")
    for name, directory in (("tokens", tokens_dir), ("vit", vit_dir)):
        m = json.loads((Path(directory) / "manifest.json").read_text())
        if m["rows"] != len(labels["token"]) or Path(m["labels"]).resolve() != Path(labels_dir).resolve():
            problems.append(f"the {name} pass was not built from these labels, row for row")
    splits = json.loads((REPO / "recipes/drivejepa_perception/splits.json").read_text())
    problems += report_guard(token_logs, {"all": np.arange(len(token_logs))}, splits["report"])
    report["data"] = {
        "source": source,
        "rows": int(len(labels["token"])),
        "logs": len(set(token_logs.tolist())),
    }

    report["labels"] = {"problems": navsim_data.check_labels(cfg, Path(labels_dir), n_windows=8)}
    problems += report["labels"]["problems"]
    sample = np.linspace(0, len(labels["token"]) - 1, live_rows).astype(int)
    report["inputs"], found = _live_equals_cache(
        entered, tokens_dir, labels, token_logs, sample, vit_dir, navsim_data.DATA_SPLIT[source]
    )
    problems += found
    l2i = np.load(Path(tokens_dir) / "lidar2img.npy", mmap_mode="r")
    centre_u = []
    for r in sample:
        h = targets.front_projection(np.asarray(l2i[r])) @ np.array([20.0, 0.0, 1.0, 1.0])
        centre_u.append(float(h[0] / h[2]))
    report["projection"] = {"u_of_20m_ahead": centre_u}
    if max(abs(u - 256.0) for u in centre_u) > 64:
        problems.append(f"20 m straight ahead projects to u = {centre_u}, not near 256: wrong slot or TO_512")
    if pdm_dir is not None:
        report["pdm_recheck"] = pdm_labels.check(cfg, Path(pdm_dir), Path(tokens_dir), Path(labels_dir), 5)
        problems += report["pdm_recheck"]
    report["problems"] = problems
    if problems:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(problems))
    return report


def _live_equals_cache(
    entered: dict, tokens_dir: Path, labels, token_logs, sample, vit_dir=None, data_split: str = "trainval"
) -> tuple:
    """Rebuild `sample` rows live (vendor inputs, batch 1) and compare them with the cache: tokens to fp16
    rounding, the pick exactly, the release's scores (where cached) within 1e-3, the ViT output."""
    import torch
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder
    from navsim.common.dataclasses import AgentInput

    from gallops.drivejepa_perception import navsim_data, tokens

    cache = {k: np.load(Path(tokens_dir) / f"{k}.npy", mmap_mode="r") for k in ("tokens", "pick")}
    if (Path(tokens_dir) / "scores.npy").exists():
        cache["scores"] = np.load(Path(tokens_dir) / "scores.npy", mmap_mode="r")
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
    logs_dir = Path(entered["navsim_root"]) / "navsim_logs" / data_split
    blobs = Path(entered["navsim_root"]) / "sensor_blobs" / data_split
    worst, picks, problems, score_gap = 0.0, 0, [], 0.0
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
            if "scores" in cache:
                live_scores = out["pdm_score"][0].float().cpu().numpy()
                score_gap = max(score_gap, float(np.abs(live_scores - cache["scores"][r]).max()))
            if vit_dir is not None:
                live_vit = captured["vit_out"].reshape(512, 1024).float().cpu().numpy()
                vit_diff.append(np.abs(live_vit - np.asarray(cache["vit_out"][r], dtype=np.float32)))
    for hook in hooks:
        hook.remove()
    del model, agent
    torch.cuda.empty_cache()
    found = {"rows": [int(r) for r in sample], "token_max_diff": worst, "picks_equal": picks}
    if "scores" in cache:
        found["scores_max_diff"] = score_gap
        if score_gap > 1e-3:
            problems.append(
                f"the live release's scores differ from the cache by {score_gap:.3g} (limit 1e-3)"
            )
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


#: B-2's slots, decoded again live in a small batch, against the stored pass (fp16, another batch's rounding).
B2_HIDDEN_ATOL, B2_CENTRE_ATOL = 0.05, 0.05


def run_b2(cfg, args, src, rows: np.ndarray, pool, root) -> dict:
    """B-2's preflight (B2-G3, PREDICTIONS.md, Part B · B-2), before a run's first step; every problem listed.

    - the code committed, and the suite passing under this interpreter;
    - the splits and the fold record re-derived; the hidden pass written under this fold record; every train
      row from a train log, none from a report log, each reading the side-car that did not train on its log;
    - the gathered teachers equal to their files on 16 rows;
    - the untrained model (arm, recipe, stage) equal, bit for bit, to the same cached path without its new
      part on 64 select rows, and that path within REFINER_LIMITS of the token pass;
    - stage 3: its stage 1's proposals recomputed equal to the dump it trains on, and the targets the trainer
      reads for 16 rows equal to the vendored scorer run live on those rows' dumped proposals;
    - the joint recipe: the online labeller giving exactly `pdm-v1`'s labels for the release's own 32 on 16
      rows;
    - an arm that reads the side-car: its stored slots equal to the fold's side-car decoding them live, 8
      rows.
    """
    import torch

    from gallops.drivejepa_perception import (
        b2_inputs,
        env,
        navsim_data,
        pdm_labels,
        train_b2,
        train_refiner,
        train_sidecar,
    )
    from gallops.train.preflight import code_checks

    report, problems = {}, []
    report["code"], found = code_checks(REPO)
    problems += found
    entered = env.enter(cfg)
    splits = navsim_data.load_splits(
        REPO / "recipes/drivejepa_perception/splits.json",
        navsim_data.navtrain_filter(Path(entered["code"]))["log_names"],
    )
    record = b2_inputs.load_folds()
    if src.hidden_manifest["folds_sha256"] != record["sha256"]:
        problems.append(f"{src.hidden_dir} was written under another fold record")
    logs = src.token_logs[rows]
    if not np.isin(logs, splits["train"]).all():
        problems.append("a training row is not from a train log")
    problems += report_guard(src.token_logs, {"train": rows}, splits["report"])
    expected = np.array([record["train"][str(log)] for log in logs], np.int8)
    wrong = int((src.fold[rows] != expected).sum())
    report["folds"] = {"rows": int(len(rows)), "wrong_fold": wrong}
    if wrong:
        problems.append(f"{wrong} train rows read a side-car that trained on their log")

    out_root = Path(cfg.resolve("{runs}/drivejepa_perception"))
    anchors = np.load(out_root / "anchors-v1" / "anchors.npy")
    count = np.load(out_root / "anchors-v1" / "count.npy")
    picked = np.random.default_rng(0).choice(rows, 16, replace=False)
    unequal = []
    for row in picked:
        path = train_refiner._anchor_file(
            train_refiner.anchors_root(cfg), src.token_logs[row], src.tokens[row]
        )
        if not np.array_equal(anchors[row, : count[row]], train_refiner._read_anchors(path)):
            unequal.append(int(row))
    report["anchors"] = {"rows": 16, "unequal": unequal}
    if unequal:
        problems.append(f"the gathered anchors differ from their files on rows {unequal}")

    device = torch.device("cuda")
    select = train_b2.select_rows(src)[:64]
    stage = args.stage
    agent, model, adapter, hook = train_b2.build(entered, args.recipe, args.arm, stage)
    if stage == 3:
        train_b2.load_weights(model, adapter, hook, Path(args.stage1) / "weights.pt", pass4=False)
    model, adapter = model.to(device).eval(), adapter.to(device).eval()
    model._backbone = model._backbone.cpu()
    with torch.no_grad():
        if stage == 3:
            base_props, _, base_scored = train_b2.forward(model, src, select, device, hook)
            hook.pass4 = __import__("copy").deepcopy(adapter)
            props, _, scored = train_b2.forward(model, src, select, device, hook)
        else:
            props, _, scored = train_b2.forward(model, src, select, device, hook)
            saved = adapter.forward
            adapter.forward = lambda proposals, queries: torch.zeros_like(queries)
            base_props, _, base_scored = train_b2.forward(model, src, select, device, hook)
            adapter.forward = saved
    same = float((props[-1] - base_props[-1]).abs().max())
    same_logits = float((scored[0] - base_scored[0]).abs().max())
    release = np.asarray(src.release["proposals"][select])
    gap = float(np.abs(base_props[-1].float().cpu().numpy() - release).max())
    pick = torch.sigmoid(base_scored[0][..., -1]).argmax(1).cpu().numpy()
    share = float((pick == src.release["pick"][select]).mean())
    report["identity"] = {
        "rows": int(len(select)),
        "untrained_minus_base_proposals": same,
        "untrained_minus_base_logits": same_logits,
        "base_minus_token_pass_m": gap,
        "base_picks_equal_share": share,
    }
    if same != 0.0 or same_logits != 0.0:
        problems.append(f"the untrained model is not its base path, bit for bit: {report['identity']}")
    if stage != 3 and (gap > REFINER_LIMITS["proposals_m"] or share < REFINER_LIMITS["picks_share"]):
        problems.append(f"the cached path does not reproduce the token pass: {report['identity']}")
    if stage == 3:
        dumped = np.load(Path(args.stage1) / "dump-train" / "dump.npz")
        at = {int(r): i for i, r in enumerate(dumped["rows"])}
        # Spread over the rows: on the first few, a row id and a position in the dump can coincide.
        some = rows[np.linspace(0, len(rows) - 1, 16).astype(int)]
        with torch.no_grad():
            p3, _, _ = train_b2.forward(model, src, some, device, hook)
        held = dumped["proposals"][[at[int(r)] for r in some]]
        moved = float(np.abs(p3[-1].float().cpu().numpy() - held).max())
        targets = train_b2.stage3_labels(Path(args.stage1), rows, len(src.tokens))[some]
        live = np.stack(
            [
                pdm_labels.score_detail(
                    pdm_labels.metric_cache_path(root, src.token_logs[r], src.tokens[r]),
                    held[j].astype(np.float64),
                )["scores"][:, -1]
                for j, r in enumerate(some)
            ]
        )
        report["stage1_dump"] = {
            "rows": 16,
            "proposals_max_m": moved,
            "targets_unequal": int((live != targets).sum()),
        }
        if moved > 1e-3:
            problems.append(
                f"stage 1's proposals, recomputed, moved {moved:.2g} m from the dump stage 3 reads"
            )
        if report["stage1_dump"]["targets_unequal"]:
            problems.append(
                f"stage 3's targets differ from the vendored scorer on its proposals: {report['stage1_dump']}"
            )
    del model, agent
    torch.cuda.empty_cache()

    if pool is not None:
        pdm = np.load(src.pdm_dir / "pdm.npz")["candidates"]
        some = rows[:16]
        got = train_b2.online_targets(
            pool, src, root, some, np.asarray(src.release["proposals"][some], np.float32), torch.device("cpu")
        )[0].numpy()
        report["online_labels"] = {"rows": 16, "unequal": int((got != pdm[some]).sum())}
        if report["online_labels"]["unequal"]:
            problems.append(f"the online labeller differs from pdm-v1: {report['online_labels']}")

    if args.arm in ("D", "HD", "H6", "H3"):
        manifest = src.hidden_manifest
        runs = [Path(r) for r in manifest["runs"]]
        run_args = json.loads((runs[0] / "run.json").read_text())["args"]
        tokens, slot, _, projections, _ = train_sidecar.load(
            manifest["inputs"]["tokens"],
            manifest["inputs"]["labels"],
            run_args["tap"],
            manifest["inputs"]["vit_tokens"],
        )
        some = select[:8]
        worst = {"hidden": 0.0, "centre": 0.0, "mask_unequal": 0}
        stored = {
            "hidden6": np.load(src.hidden_dir / "hidden6.npy", mmap_mode="r"),
            "centre6": np.load(src.hidden_dir / "centre6.npy", mmap_mode="r"),
            "mask": np.load(src.hidden_dir / "agents.npz")["mask"],
        }
        for k in sorted(set(src.fold[some].tolist())):
            mine = some[src.fold[some] == k]
            model_k, _ = train_sidecar.load_run(runs[k])
            n, q = len(mine), model_k.query.shape[0]
            sink = {key: np.zeros((n, q, *shape), dtype) for key, (shape, dtype) in b2_inputs.KEYS.items()}
            for layer in b2_inputs.LAYERS:
                sink[f"hidden{layer}"] = np.zeros((n, q, model_k.query.shape[1]), np.float16)
                sink[f"centre{layer}"] = np.zeros((n, q, 3), np.float32)
                sink[f"conf{layer}"] = np.zeros((n, q), np.float32)
            b2_inputs.decode_rows(model_k, tokens, slot, mine, projections, device, sink, at=np.arange(n))
            worst["hidden"] = max(
                worst["hidden"],
                float(
                    np.abs(
                        sink["hidden6"].astype(np.float32) - stored["hidden6"][mine].astype(np.float32)
                    ).max()
                ),
            )
            worst["centre"] = max(
                worst["centre"], float(np.abs(sink["centre6"] - stored["centre6"][mine]).max())
            )
            worst["mask_unequal"] += int((sink["mask"] != stored["mask"][mine]).sum())
            del model_k
        report["slots_live"] = {"rows": int(len(some)), **worst}
        if worst["hidden"] > B2_HIDDEN_ATOL or worst["centre"] > B2_CENTRE_ATOL or worst["mask_unequal"]:
            problems.append(
                f"the stored slots differ from the fold side-cars run live: {report['slots_live']}"
            )
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


def main(argv=None) -> None:
    import argparse

    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.preflight")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data", help="Part B's data preflight for a navtest or R-clean pass set")
    p.add_argument("--labels", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--vit-tokens", required=True)
    p.add_argument("--pdm", default="")
    p.add_argument("--out", required=True, help="the report, JSON")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    report = run_data(
        SiteConfig.load(),
        Path(args.labels),
        Path(args.tokens),
        Path(args.vit_tokens),
        Path(args.pdm) if args.pdm else None,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("data", "inputs", "projection")}, indent=1, default=str))


if __name__ == "__main__":
    main()
