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
  the cache: tokens to fp16 rounding, and the release's pick exactly.
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


def run(cfg, tokens_dir: Path, labels_dir: Path, rows: dict, live_rows: int = 6) -> dict:
    """All checks; returns the report for the run's files or raises SystemExit listing every problem.

    `rows` maps the split names a run uses (train, select, report, tap) to the row arrays it reads.
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
    report["inputs"], found = _live_equals_cache(entered, tokens_dir, labels, token_logs, sample)
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


def _live_equals_cache(entered: dict, tokens_dir: Path, labels, token_logs, sample) -> tuple:
    """Rebuild `sample` rows live (vendor inputs, batch 1) and compare them with the cache."""
    import torch
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder
    from navsim.common.dataclasses import AgentInput

    from gallops.drivejepa_perception import navsim_data

    cache = {k: np.load(Path(tokens_dir) / f"{k}.npy", mmap_mode="r") for k in ("tokens", "pick")}
    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.cuda().eval()
    captured = {}
    hook = model._backbone.register_forward_hook(lambda m, i, o: captured.__setitem__("tokens", o[0]))
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
    hook.remove()
    del model, agent
    torch.cuda.empty_cache()
    if worst > TOKEN_ATOL:
        problems.append(f"live tokens differ from the cache by {worst:.3g} (limit {TOKEN_ATOL})")
    if picks != len(sample):
        problems.append(f"the live pick equals the cached one on {picks} of {len(sample)} rows")
    return {"rows": [int(r) for r in sample], "token_max_diff": worst, "picks_equal": picks}, problems
