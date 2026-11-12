"""B-1, the analytic veto: a rule on the agents removes the release's proposals that come too close.

    python -m gallops.drivejepa_perception.veto clearances --root {out} --split fresh \
        --out {out}/part-b/b1/clearance-fresh
    python -m gallops.drivejepa_perception.veto grid --root {out} --split fresh \
        --clearances {out}/part-b/b1/clearance-fresh --out {out}/part-b/b1/grid-fresh
    python -m gallops.drivejepa_perception.veto freeze --grid {out}/part-b/b1/grid-fresh \
        --write recipes/drivejepa_perception/part_b1_frozen.json

docs/drivejepa-perception-sidecar/part-b/B-1-veto.md. Nothing trains: the release still proposes and
scores its 32 plans; B-1 changes only the pick, and only by removing proposals.

**The rule** (B-1, Q4). A proposal is vetoed if, at any plan time within the horizon H, its clearance
to an agent (`geometry.clearances`: footprint circles, the agent at its 0.5 s and 1 s path then constant
velocity) is under the margin m. The pick is the release's highest score among the rest, first index
on ties as the release's argmax; if every proposal is vetoed, the release's own pick stands (D24, B-1
(b)). Its sub-versions differ only in the agents:
- **B-1-1, the ceiling:** no agents; it vetoes exactly the proposals the grader's own (vendored) labels
  fail on NC (0 or 0.5: static objects too) or TTC. It reads the answer and is never a result;
- **B-1-2:** the labelled agents in CAM_F0's view within 50 m (`targets.truth_agents`, a lost track's
  step filled at constant velocity);
- **B-1-3:** the side-car's agents at confidence c or more, seed k on the planner's seed k.

**The settings** (D24, B-1 (a)): margins 0.5, 1, 2 m; horizons 1, 2 s; confidence 0.5, 0.7 for the
side-car. They are chosen on R-clean only (`grid`, then `freeze`), never on navtest.

**The rows chosen on** (D25): R-clean's windows whose driving command is one navtest uses (left,
straight, right). On the other 77% the command is "unknown", which neither navtrain nor navtest ever has,
so the release's input there is out of distribution (its pick fails NC on 4.0% of them, on 0.65% of the
rest): 2,624 windows over 20 logs remain. `known_command` reads the command from the release's own input
(the token pass's `ego_status`).

**The selection value** of a pick on R-clean is its vendored PDM final score under the human filter
(`filtered_final`): every term the human's own plan fails is forgiven, as navsim's filter forgives it,
recombined without EC (the vendored scorer has none). Its mean is reweighted to navtest's city mix
(`city_weights`); Singapore, which R-clean lacks, is left out and the rest renormalised.

**Checks before any number:** with an empty veto, the rule's pick equals the release's recorded pick
on every row; the release's scores reproduce that pick; each clearance file's inputs are the ones on
disk (sha256). A navtrain split is refused unless `--smoke`, and a smoke grid is never frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FROZEN = "recipes/drivejepa_perception/part_b1_frozen.json"
PREDICTIONS = "experiments/H15_perception_sidecar/PREDICTIONS.md"
PREDICTIONS_SECTION = "## Part B · B-1, the analytic veto"

#: D24, B-1 (a).
MARGINS = (0.5, 1.0, 2.0)
HORIZONS = (1.0, 2.0)
CONFIDENCES = (0.5, 0.7)
#: The plan's poses are 0.5 s apart, the first at 0.5 s: H seconds is its first H / 0.5 times.
STEP_S = 0.5
SEEDS = (0, 1, 2)
#: The driving command's one-hot slots in the release's own `ego_status` (its last frame): left, straight,
#: right, unknown. navtrain and navtest never use "unknown"; B-1 chooses only on the others (D25).
COMMAND_SLOTS, UNKNOWN_COMMAND = slice(7, 11), 3
#: The vendored labels' columns (pdm_labels.COLUMNS) and the final score's weights.
NC, DAC, TLC, DDC, EP, TTC, LK, HC, FINAL = range(9)


def horizon_steps(horizon: float) -> int:
    return int(round(horizon / STEP_S))


def settings(source: str) -> list:
    """The grid for a sub-version: (margin, horizon, confidence); the labels have no confidence."""
    if source == "B-1-1":
        return [None]
    confidences = CONFIDENCES if source == "B-1-3" else (None,)
    return [(m, h, c) for m in MARGINS for h in HORIZONS for c in confidences]


def setting_name(setting) -> str:
    if setting is None:
        return "labels"
    m, h, c = setting
    return f"m{m:g}-h{h:g}" + ("" if c is None else f"-c{c:g}")


# ------------------------------------------------------------------------------------ the rule


def vetoed(per_time: np.ndarray, margin: float, horizon: float) -> np.ndarray:
    """(N, K) True where a proposal's clearance (N, K, T, minimum over agents) is under `margin` at any
    plan time within `horizon` seconds."""
    return (np.asarray(per_time)[..., : horizon_steps(horizon)] < margin).any(-1)


def ceiling_vetoed(pdm: np.ndarray) -> np.ndarray:
    """B-1-1: (N, K) True where the vendored labels fail NC (0 or 0.5) or TTC."""
    return (pdm[..., NC] < 1) | (pdm[..., TTC] < 1)


def pick_with_veto(scores: np.ndarray, veto: np.ndarray, release_pick: np.ndarray) -> tuple:
    """The release's highest score among the proposals left; the release's pick where all are vetoed.

    numpy's argmax returns the first index on ties, as the release's torch argmax does.
    """
    masked = np.where(veto, -np.inf, np.asarray(scores, dtype=np.float64))
    pick = masked.argmax(1)
    every = np.asarray(veto).all(1)
    pick[every] = np.asarray(release_pick)[every]
    return pick.astype(np.int64), every


def filtered_final(plan: np.ndarray, human: np.ndarray) -> np.ndarray:
    """The vendored final score of (N, 9) plan labels under the human filter: each of the eight terms the
    human's own plan scores 0 on is set to 1, then NC.DAC.TLC.DDC.(5 EP + 5 TTC + 2 LK + 2 HC) / 14.

    navsim's filter forgives a term where the human fails it. The vendored EP is already zeroed by its
    plan's own gates, so where the human fails a gate the plan's EP is forgiven too, as navsim's is.
    """
    terms = np.where(np.asarray(human)[..., :8] == 0, 1.0, np.asarray(plan, dtype=np.float64)[..., :8])
    gates = terms[..., NC] * terms[..., DAC] * terms[..., TLC] * terms[..., DDC]
    return gates * (5 * terms[..., EP] + 5 * terms[..., TTC] + 2 * terms[..., LK] + 2 * terms[..., HC]) / 14


def city_weights(row_cities: np.ndarray, target_cities: np.ndarray) -> tuple:
    """Per-row weights that make a mean over `row_cities` stand for `target_cities`' city mix.

    A row in city c gets s_c / n_c, where n_c counts c's rows and s_c is c's share of the target among
    the cities the rows have (a city the rows lack, Singapore for R-clean, is dropped and the rest
    renormalised). The weights sum to 1. Returns them and a record of the shares.
    """
    row_cities, target_cities = np.asarray(row_cities), np.asarray(target_cities)
    present = sorted(set(row_cities.tolist()))
    target = {c: float((target_cities == c).mean()) for c in sorted(set(target_cities.tolist()))}
    kept = {c: target.get(c, 0.0) for c in present}
    total = sum(kept.values())
    if total <= 0:
        raise SystemExit("the target shares none of the rows' cities")
    shares = {c: v / total for c, v in kept.items()}
    counts = {c: int((row_cities == c).sum()) for c in present}
    weights = np.array([shares[c] / counts[c] for c in row_cities.tolist()])
    record = {
        "target_share": target,
        "used_share": shares,
        "dropped": sorted(set(target) - set(present)),
        "rows_per_city": counts,
        "kish_n_eff": float(1.0 / (weights**2).sum()),
    }
    return weights, record


def weighted_mean_se(values: np.ndarray, clusters: np.ndarray, weights: np.ndarray) -> tuple:
    """The weighted mean and its SE clustered by `clusters` (b_report.clustered_se_weighted)."""
    from gallops.drivejepa_perception.b_report import clustered_se_weighted

    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    mean = float((weights * values).sum() / weights.sum())
    return mean, clustered_se_weighted(values, clusters, weights)


def confusion(veto: np.ndarray, failing: np.ndarray) -> dict:
    """Proposal-level counts of a veto against the grader's labels (B-1-1's vetoes)."""
    veto, failing = np.asarray(veto, bool), np.asarray(failing, bool)
    return {
        "right": int((veto & failing).sum()),
        "wrong": int((veto & ~failing).sum()),
        "missed": int((~veto & failing).sum()),
        "kept": int((~veto & ~failing).sum()),
    }


def choose(values: dict, veto_rates: dict) -> str:
    """The frozen setting: the highest weighted R-clean value; a tie goes to the fewer vetoes, then to
    the grid's order (`values` is in the grid's order)."""
    best = max(values.values())
    tied = [name for name, v in values.items() if v == best]
    return min(tied, key=lambda name: (veto_rates[name], list(values).index(name)))


# ------------------------------------------------------------------------------------ the data


O_DIRS = {
    "select": ("tokens-v1", "labels-v1", "pdm-v1", "agents-select-v3-s{seed}"),
    "fresh": ("tokens-fresh-v1", "labels-fresh-v1", "pdm-fresh-v1", "agents-fresh-v3-s{seed}"),
    "navtest": ("tokens-navtest-v1", "labels-navtest-v1", "pdm-navtest-v1", "agents-navtest-v3-s{seed}"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def known_command(tokens_dir: Path) -> np.ndarray:
    """(N,) True where the release's input carries a driving command navtest uses (D25); refuses a row
    whose command is not one-hot."""
    command = np.load(Path(tokens_dir) / "ego_status.npy", mmap_mode="r")[:, -1, COMMAND_SLOTS]
    command = np.asarray(command)
    if not (np.isin(command, [0.0, 1.0]).all(1) & (command.sum(1) == 1)).all():
        raise SystemExit(f"{tokens_dir}: a driving command is not one-hot")
    return command.argmax(1) != UNKNOWN_COMMAND


def split_rows(split: str, labels_dir: Path, tokens_dir=None) -> np.ndarray:
    """The rows B-1 reads: R-clean's known-command windows (D25); every navtest row; navtrain's select rows
    for the smoke."""
    from gallops.drivejepa_perception import navsim_data

    labels = np.load(Path(labels_dir) / "labels.npz")
    manifest = json.loads((Path(labels_dir) / "manifest.json").read_text())
    if navsim_data.source_of(manifest) != ("navtrain" if split == "select" else split):
        raise SystemExit(f"{labels_dir} is not {split}'s labels pass")
    if split == "fresh":
        return np.flatnonzero(known_command(tokens_dir))
    if split == "navtest":
        return np.arange(len(labels["token"]))
    splits = json.loads((REPO / "recipes/drivejepa_perception/splits.json").read_text())
    token_logs = np.array(manifest["logs"])[labels["log_id"]]
    return np.flatnonzero(np.isin(token_logs, splits["select"]))


def load_split(out_root: Path, split: str) -> dict:
    """What B-1 reads on a split: proposals, the release's scores and pick, the vendored labels, logs and
    cities per row, restricted to its rows. Refuses unless the scores reproduce the recorded pick."""
    from gallops.drivejepa_perception import tokens

    tok, lab, pdm_name, _ = O_DIRS[split]
    out_root = Path(out_root)
    labels_dir = out_root / lab
    rows = split_rows(split, labels_dir, out_root / tok)
    labels = np.load(labels_dir / "labels.npz")
    manifest = json.loads((labels_dir / "manifest.json").read_text())
    scores = tokens.release_scores(out_root / tok)
    pick = np.load(out_root / tok / "pick.npy").astype(np.int64)
    problems = tokens.pick_problems(scores, pick)
    if problems:
        raise SystemExit("the release's scores do not reproduce its pick: " + "; ".join(problems))
    pdm = np.load(out_root / pdm_name / "pdm.npz")
    logs = np.array(manifest["logs"])[labels["log_id"]]
    cities = manifest.get("cities")
    return {
        "split": split,
        "rows": rows,
        "token": labels["token"][rows],
        "log": logs[rows],
        "city": None if cities is None else np.array(cities)[labels["log_id"]][rows],
        "proposals": np.load(out_root / tok / "proposals.npy", mmap_mode="r"),
        "scores": scores[rows],
        "pick": pick[rows],
        "pdm": pdm["candidates"][rows],
        "human": pdm["human"][rows],
        "inputs": {
            "tokens": str(out_root / tok),
            "labels": str(labels_dir),
            "pdm": str(out_root / pdm_name),
            "pdm_sha256": _sha256(out_root / pdm_name / "pdm.npz"),
        },
    }


def agent_sources(out_root: Path, split: str) -> dict:
    """name -> (xy, size, vel, path, conf, mask-or-None, its file) for B-1-2 and B-1-3's seeds."""
    from gallops.drivejepa_perception import targets as tg

    out_root = Path(out_root)
    tok, lab, _, agents_name = O_DIRS[split]
    labels = dict(np.load(out_root / lab / "labels.npz"))
    projections = tg.front_projection(np.load(out_root / tok / "lidar2img.npy"))
    truth = tg.truth_agents(labels, projections, fill="constant_velocity")
    sources = {"truth": (truth, str(out_root / lab / "labels.npz"))}
    for seed in SEEDS:
        path = out_root / agents_name.format(seed=seed) / "agents.npz"
        sources[f"s{seed}"] = (dict(np.load(path)), str(path))
    return sources


def clearance_arrays(proposals, agents: dict, rows: np.ndarray, confidence, chunk: int = 256) -> tuple:
    """Per row of `rows`: (K, T) the smallest clearance over the agents kept, and which agent it was
    (-1 where none is kept). Agents are kept by their mask, or by confidence >= `confidence`."""
    import torch

    from gallops.drivejepa_perception import geometry

    n, t = len(rows), len(geometry.TIMES)
    per_time = np.zeros((n, 32, t), np.float32)
    who = np.full((n, 32, t), -1, np.int16)
    keep_all = agents["mask"] if confidence is None else agents["conf"] >= confidence
    with torch.no_grad():
        for i in range(0, n, chunk):
            r = rows[i : i + chunk]
            keep = torch.from_numpy(np.asarray(keep_all[r]))
            c = geometry.clearances(
                torch.from_numpy(np.asarray(proposals[r], dtype=np.float32)),
                torch.from_numpy(np.asarray(agents["xyz"][r][..., :2], dtype=np.float32)),
                torch.from_numpy(np.asarray(agents["lwh"][r][..., :2], dtype=np.float32)),
                torch.from_numpy(np.asarray(agents["vel"][r], dtype=np.float32)),
                torch.from_numpy(np.asarray(agents["path"][r], dtype=np.float32)),
                keep,
            )
            values, index = c.min(dim=2)
            per_time[i : i + len(r)] = values.numpy()
            index = torch.where(keep.any(dim=1)[:, None, None], index, torch.full_like(index, -1))
            who[i : i + len(r)] = index.numpy().astype(np.int16)
    return per_time, who


def write_clearances(cfg, out_root: Path, split: str, out: Path, smoke: bool) -> dict:
    """Every source's clearance arrays on a split, written once (`clearance-<split>/<name>.npz`)."""
    from gallops.drivejepa_b2d.provenance import git_sha

    if split == "select" and not smoke:
        raise SystemExit("navtrain's select rows are read only as a smoke run (--smoke), choosing nothing")
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started, sha = time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha()
    data = load_split(out_root, split)
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    files = {}
    for name, (agents, path) in agent_sources(out_root, split).items():
        for confidence in (None,) if name == "truth" else CONFIDENCES:
            key = name if confidence is None else f"{name}-c{confidence:g}"
            per_time, who = clearance_arrays(data["proposals"], agents, data["rows"], confidence)
            np.savez(staging / f"{key}.npz", per_time=per_time, agent=who)
            files[key] = {"agents": path, "agents_sha256": _sha256(Path(path)), "confidence": confidence}
            print(f"{key}: {len(data['rows'])} rows", flush=True)
    manifest = {
        "pass": "veto-clearances",
        "split": split,
        "smoke": smoke,
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "rows": int(len(data["rows"])),
        "inputs": data["inputs"],
        "proposals_sha256": _sha256(Path(data["inputs"]["tokens"]) / "proposals.npy"),
        "files": files,
        "truth_fill": "constant_velocity",
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))
    return manifest


# ------------------------------------------------------------------------------------ the grid


def predictions_committed() -> list:
    """Why B-1 may not read R-clean yet: its PREDICTIONS.md section must be committed, unedited since."""
    problems = []
    head = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{PREDICTIONS}"], capture_output=True, text=True
    ).stdout
    if PREDICTIONS_SECTION not in head:
        problems.append(f"PREDICTIONS.md has no committed '{PREDICTIONS_SECTION}' section")
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", PREDICTIONS], capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        problems.append("PREDICTIONS.md has uncommitted changes")
    return problems


def evaluate_pick(data: dict, pick: np.ndarray, veto, failing: np.ndarray, weights, who=None) -> dict:
    """One rule's pick on a split, against the release's: the selection value, rates and confusion."""
    n = np.arange(len(pick))
    release = data["pick"]
    value = filtered_final(data["pdm"][n, pick], data["human"])
    base = filtered_final(data["pdm"][n, release], data["human"])
    w = np.full(len(pick), 1.0 / len(pick)) if weights is None else weights
    mean, se = weighted_mean_se(value - base, data["log"], w)
    plain = float((value - base).mean())
    out = {
        "value": float((w * value).sum()),
        "release_value": float((w * base).sum()),
        "minus_release": mean,
        "minus_release_se": se,
        "minus_release_unweighted": plain,
        "final_minus_release": float((data["pdm"][n, pick, FINAL] - data["pdm"][n, release, FINAL]).mean()),
        "veto_rate": None if veto is None else float(np.asarray(veto).mean()),
        "all_vetoed_rate": None if veto is None else float(np.asarray(veto).all(1).mean()),
        "switch_rate": float((pick != release).mean()),
        "pick_nc_fail": float((data["pdm"][n, pick, NC] < 1).mean()),
        "pick_ttc_fail": float((data["pdm"][n, pick, TTC] < 1).mean()),
        "pick_ep": float(data["pdm"][n, pick, EP].mean()),
        "release_nc_fail": float((data["pdm"][n, release, NC] < 1).mean()),
        "release_ttc_fail": float((data["pdm"][n, release, TTC] < 1).mean()),
        "release_ep": float(data["pdm"][n, release, EP].mean()),
    }
    if veto is not None:
        out["confusion"] = confusion(veto, failing)
    return out


def run_grid(out_root: Path, split: str, clearance_dir: Path, out: Path, smoke: bool) -> dict:
    """Every setting of every sub-version on a split, scored by lookup in the vendored labels."""
    from gallops.drivejepa_b2d.provenance import git_sha

    if split == "select" and not smoke:
        raise SystemExit("navtrain's select rows are read only as a smoke run (--smoke), choosing nothing")
    if split == "navtest":
        raise SystemExit("settings are never chosen on navtest")
    if not smoke:
        problems = predictions_committed()
        if problems:
            raise SystemExit("B-1 may not read R-clean yet:\n- " + "\n- ".join(problems))
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    data = load_split(out_root, split)
    cmanifest = json.loads((Path(clearance_dir) / "manifest.json").read_text())
    if cmanifest["split"] != split or cmanifest["inputs"] != data["inputs"]:
        raise SystemExit(f"{clearance_dir} was computed on other inputs")
    for key, f in cmanifest["files"].items():
        if _sha256(Path(f["agents"])) != f["agents_sha256"]:
            raise SystemExit(f"{key}: its agents changed since the clearances were computed")
    empty, _ = pick_with_veto(data["scores"], np.zeros_like(data["scores"], bool), data["pick"])
    if not np.array_equal(empty, data["pick"]):
        raise SystemExit(f"an empty veto changes {int((empty != data['pick']).sum())} picks")

    weights, city_record = None, None
    if split == "fresh":
        navtest = np.load(Path(out_root) / O_DIRS["navtest"][1] / "labels.npz")
        nmanifest = json.loads((Path(out_root) / O_DIRS["navtest"][1] / "manifest.json").read_text())
        weights, city_record = city_weights(data["city"], np.array(nmanifest["cities"])[navtest["log_id"]])
    failing = ceiling_vetoed(data["pdm"])
    results, picks = {}, {}

    veto = failing
    pick, _ = pick_with_veto(data["scores"], veto, data["pick"])
    results["B-1-1"] = {"labels": evaluate_pick(data, pick, veto, failing, weights)}
    picks["B-1-1/labels"] = pick
    for source, names in (("B-1-2", ["truth"]), ("B-1-3", [f"s{k}" for k in SEEDS])):
        results[source] = {}
        for setting in settings(source):
            m, h, c = setting
            entry = {}
            for name in names:
                key = name if c is None else f"{name}-c{c:g}"
                arrays = np.load(Path(clearance_dir) / f"{key}.npz")
                veto = vetoed(arrays["per_time"], m, h)
                pick, _ = pick_with_veto(data["scores"], veto, data["pick"])
                entry[name] = evaluate_pick(data, pick, veto, failing, weights)
                picks[f"{source}/{setting_name(setting)}/{name}"] = pick
            if source == "B-1-3":
                entry["mean_over_seeds"] = {
                    k: float(np.mean([entry[s][k] for s in names]))
                    for k in ("value", "minus_release", "veto_rate", "switch_rate", "pick_ep")
                }
            results[source][setting_name(setting)] = entry
    record = {
        "pass": "veto-grid",
        "split": split,
        "smoke": smoke,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": git_sha(),
        "rows": int(len(data["rows"])),
        "logs": int(len(set(data["log"].tolist()))),
        "clearances": str(clearance_dir),
        "inputs": data["inputs"],
        "value": "vendored final, human-filtered, no EC (filtered_final)",
        "weights": city_record or "unweighted",
        "release": evaluate_pick(data, data["pick"], None, failing, weights),
        "results": results,
    }
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    (staging / "grid.json").write_text(json.dumps(record, indent=1))
    np.savez(
        staging / "picks.npz",
        rows=data["rows"],
        release=data["pick"],
        **{k.replace("/", "|"): v for k, v in picks.items()},
    )
    staging.rename(out)
    return record


# ------------------------------------------------------------------------------------ the freeze


def frozen_record(grid: dict, grid_sha256: str) -> dict:
    """Each deployable sub-version's setting, by the rule (`choose`), from a grid's values."""
    chosen, values = {}, {}
    for source in ("B-1-2", "B-1-3"):
        entries = grid["results"][source]
        key = "truth" if source == "B-1-2" else "mean_over_seeds"
        v = {name: entries[name][key]["value"] for name in entries}
        rates = {name: entries[name][key]["veto_rate"] for name in entries}
        chosen[source] = choose(v, rates)
        values[source] = {name: {"value": v[name], "veto_rate": rates[name]} for name in entries}
    return {
        "rule": "per sub-version, the setting with the highest R-clean selection value (vendored final, "
        "human-filtered, reweighted to navtest's cities), B-1-3 by its mean over seeds 0-2; a tie goes "
        "to the fewer vetoes, then to the grid's order (D24; PREDICTIONS.md, Part B · B-1)",
        "grid": {"margins": list(MARGINS), "horizons": list(HORIZONS), "confidences": list(CONFIDENCES)},
        "source": {"grid_sha256": grid_sha256, "split": grid["split"], "rows": grid["rows"]},
        "weights": grid["weights"],
        "values": values,
        "chosen": chosen,
    }


def load_frozen(path: Path, grid_dir=None) -> dict:
    """The committed choice, re-derived from its own values by the rule and refused if edited; with the
    grid, held to it by sha256."""
    stored = json.loads(Path(path).read_text())
    for source, entries in stored["values"].items():
        v = {name: e["value"] for name, e in entries.items()}
        rates = {name: e["veto_rate"] for name, e in entries.items()}
        if choose(v, rates) != stored["chosen"][source]:
            raise SystemExit(f"{path}: {source}'s chosen setting is not what its values give")
    if grid_dir is not None and _sha256(Path(grid_dir) / "grid.json") != stored["source"]["grid_sha256"]:
        raise SystemExit(f"{path}: the grid on disk is not the one it was chosen from")
    return stored


def freeze(grid_dir: Path, write: Path) -> dict:
    grid = json.loads((Path(grid_dir) / "grid.json").read_text())
    if grid["smoke"] or grid["split"] != "fresh":
        raise SystemExit("only R-clean's grid is frozen; a smoke grid chooses nothing")
    write = Path(write)
    if write.exists():
        raise SystemExit(f"{write} exists: B-1's settings are frozen once")
    record = frozen_record(grid, _sha256(Path(grid_dir) / "grid.json"))
    write.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["chosen"], indent=1))
    return record


# ------------------------------------------------------------------------------ navtest, once


def committed_clean(rel: str) -> list:
    """Why a file may not stand behind a navtest read: not committed, or edited since."""
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel], capture_output=True
    )
    if tracked.returncode:
        return [f"{rel} is not committed"]
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", rel], capture_output=True, text=True
    )
    return [f"{rel} has uncommitted changes"] if dirty.stdout.strip() else []


def setting_of(source: str, name: str):
    """The setting a frozen name stands for."""
    named = {setting_name(s): s for s in settings(source)}
    if name not in named:
        raise SystemExit(f"{source}: {name!r} is not a setting of the grid")
    return named[name]


def frozen_picks(out_root: Path, split: str, frozen: dict) -> tuple:
    """Each arm's pick on a split at the frozen settings: (data, arm -> pick, veto, who, setting, source)."""
    data = load_split(out_root, split)
    failing = ceiling_vetoed(data["pdm"])
    pick, _ = pick_with_veto(data["scores"], failing, data["pick"])
    arms = {"B-1-1": {"pick": pick, "veto": failing, "who": None, "setting": None, "source": None}}
    sources = agent_sources(out_root, split)
    for source, names in (("B-1-2", ["truth"]), ("B-1-3", [f"s{k}" for k in SEEDS])):
        m, h, c = setting_of(source, frozen["chosen"][source])
        for name in names:
            agents, path = sources[name]
            per_time, who = clearance_arrays(data["proposals"], agents, data["rows"], c)
            v = vetoed(per_time, m, h)
            pick, _ = pick_with_veto(data["scores"], v, data["pick"])
            arm = source if name == "truth" else f"{source}-{name}"
            arms[arm] = {
                "pick": pick,
                "veto": v,
                "who": who,
                "per_time": per_time,
                "setting": (m, h, c),
                "source": name,
            }
    return data, arms


def verdicts(pairs: dict) -> dict:
    """V1 to V5 and B-1-4's trigger, by PREDICTIONS.md's rules (Part B · B-1)."""

    def d(arm, key="epdms_hf"):
        return pairs[arm][key]["a_minus_b"], pairs[arm][key]["se"]

    seeds = [f"B-1-3-s{k}" for k in SEEDS]
    v1, v1se = d("B-1-1")
    v2, v2se = d("B-1-2")
    v3 = {arm: d(arm) for arm in seeds}
    ep = {arm: d(arm, "EP") for arm in seeds}
    v5 = float(np.mean([-100 * ep[arm][0] for arm in seeds]))
    return {
        "V1": {
            "b11_minus_r": v1,
            "se": v1se,
            "supports": v1 > 2 * v1se,
            "note": "the ceiling: reads the answer",
        },
        "V2": {"b12_minus_r": v2, "se": v2se, "supports": v2 > 2 * v2se, "refutes": v2 < -2 * v2se},
        "V3": {
            "per_seed": {arm: {"b13_minus_r": x, "se": se} for arm, (x, se) in v3.items()},
            "supports": all(x > 2 * se for x, se in v3.values()),
            "refutes": any(x < -2 * se for x, se in v3.values()),
        },
        "V4": {"share_kept": float(np.mean([x for x, _ in v3.values()]) / v2) if v2 else float("nan")},
        "V5": {"ep_lost_points": v5, "refutes": v5 > 0.6},
        "B-1-4_runs": v5 > 0.6 or any(x < -2 * se for x, se in ep.values()),
    }


def read_navtest(
    cfg,
    out_root: Path,
    grid_dir: Path,
    verified: Path,
    anchor_dir: Path,
    r_dir: Path,
    out: Path,
    workers: int,
) -> dict:
    """B-1 on navtest, once: each arm's picks at the frozen settings, scored officially, paired with R."""
    from gallops.drivejepa_perception import navtest

    problems = predictions_committed() + committed_clean(FROZEN)
    frozen = load_frozen(REPO / FROZEN, grid_dir)
    anchor = json.loads((Path(anchor_dir) / "anchor.json").read_text())
    if not all(anchor["gates"].values()):
        problems.append(f"B-0's anchor does not hold: {anchor['gates']}")
    out = Path(out)
    if (out / "navtest.json").exists():
        problems.append(f"{out} holds a finished read: navtest is read once")
    if problems:
        raise SystemExit("B-1 may not read navtest:\n- " + "\n- ".join(problems))
    out.mkdir(parents=True, exist_ok=True)  # resumes arm by arm: a scored arm is never scored again
    data, arms = frozen_picks(out_root, "navtest", frozen)
    empty, _ = pick_with_veto(data["scores"], np.zeros_like(data["scores"], bool), data["pick"])
    if not np.array_equal(empty, data["pick"]):
        raise SystemExit("an empty veto changes a navtest pick")
    labels_dir = Path(out_root) / O_DIRS["navtest"][1]
    tokens = [str(t) for t in data["token"]]
    n = np.arange(len(tokens))
    pairs, rates = {}, {}
    for arm, a in arms.items():
        plans = out / f"plans-{arm}"
        if not plans.exists():
            meta = {
                "arm": arm,
                "setting": None if a["setting"] is None else setting_name(a["setting"]),
                "frozen_sha256": _sha256(REPO / FROZEN),
                "proposals": data["inputs"]["tokens"],
            }
            navtest.write_plans(
                plans,
                tokens,
                np.asarray(data["proposals"][data["rows"], a["pick"]]),
                meta,
                pick=a["pick"],
                row=data["rows"],
            )
        scored = out / f"score-{arm}"
        if not scored.exists():
            navtest.score_plans(cfg, plans, verified, scored, workers, arm)
        pairs[arm] = navtest.paired(scored, r_dir, labels_dir)
        rates[arm] = {
            "veto_rate": float(a["veto"].mean()),
            "all_vetoed_rate": float(a["veto"].all(1).mean()),
            "switch_rate": float((a["pick"] != data["pick"]).mean()),
            "vendored_final_minus_release": float(
                (data["pdm"][n, a["pick"], FINAL] - data["pdm"][n, data["pick"], FINAL]).mean() * 100
            ),
        }
    record = {
        "pass": "veto-navtest",
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen": frozen["chosen"],
        "frozen_sha256": _sha256(REPO / FROZEN),
        "r": str(r_dir),
        "pairs": pairs,
        "rates": rates,
        "verdicts": verdicts(pairs),
    }
    (out / "navtest.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record["verdicts"], indent=1))
    return record


# ------------------------------------------------------------------------------------ analysis


def vetoing_agent(a: dict, data: dict) -> np.ndarray:
    """Per row, the agent index behind the veto of the release's own pick (-1 where it was not vetoed):
    the agent at the time within the horizon where that proposal's clearance is smallest."""
    n = np.arange(len(data["pick"]))
    steps = horizon_steps(a["setting"][1])
    per_time = a["per_time"][n, data["pick"], :steps]
    when = per_time.argmin(1)
    who = a["who"][n, data["pick"], :steps][n, when]
    return np.where(a["veto"][n, data["pick"]], who, -1)


def analyse(out_root: Path, split: str, grid_dir: Path, out: Path, official_dir=None) -> dict:
    """The frozen settings, taken apart on a split: rates, the confusion against B-1-1, EP lost on switched
    rows by the class and range of the agent behind the veto, false alarms (B-1-3), the three subsets."""
    from gallops.drivejepa_perception import evaluate, navsim_data
    from gallops.drivejepa_perception import targets as tg

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    frozen = load_frozen(REPO / FROZEN, grid_dir)
    data, arms = frozen_picks(out_root, split, frozen)
    tok, lab, _, _ = O_DIRS[split]
    labels = dict(np.load(Path(out_root) / lab / "labels.npz"))
    projections = tg.front_projection(np.load(Path(out_root) / tok / "lidar2img.npy"))
    sources = agent_sources(out_root, split)
    # The subsets are read over every row of the labels (navsim_data.subsets walks them all), then kept for
    # the rows B-1 reads: R-clean's are its known-command windows (D25).
    subsets = {
        k: v[data["rows"]] for k, v in navsim_data.subsets(labels, np.asarray(data["proposals"])).items()
    }
    failing = ceiling_vetoed(data["pdm"])
    n = np.arange(len(data["pick"]))
    record = {"split": split, "rows": int(len(n)), "frozen": frozen["chosen"], "arms": {}}
    for arm, a in arms.items():
        switched = a["pick"] != data["pick"]
        ep_lost = (data["pdm"][n, data["pick"], EP] - data["pdm"][n, a["pick"], EP]) * switched
        entry = {
            "veto_rate": float(a["veto"].mean()),
            "all_vetoed_rate": float(a["veto"].all(1).mean()),
            "switch_rate": float(switched.mean()),
            "confusion": confusion(a["veto"], failing),
            "pick_vetoes_right": int((switched & failing[n, data["pick"]]).sum()),
            "pick_vetoes_wrong": int((switched & ~failing[n, data["pick"]]).sum()),
            "ep_lost_points": float(ep_lost.mean() * 100),
            "subsets": {
                name: {
                    "rows": int(m.sum()),
                    "switch_rate": float(switched[m].mean()) if m.any() else float("nan"),
                }
                for name, m in subsets.items()
            },
        }
        if a["who"] is not None:
            agents = sources[a["source"]][0]
            who = vetoing_agent(a, data)
            by = {}
            false_alarm, real = 0, 0
            for r in np.flatnonzero(switched & (who >= 0)):
                row, q = int(data["rows"][r]), int(who[r])
                xy = agents["xyz"][row][q, :2].astype(np.float64)
                reach = float(np.hypot(*xy))
                band = next((f"{lo:g}-{hi:g}" for lo, hi in evaluate.RANGE_BINS if lo <= reach < hi), ">50")
                key = f"{navsim_data.CLASSES[int(agents['cls'][row][q])]}|{band}"
                by[key] = by.get(key, 0.0) + float(ep_lost[r])
                if a["source"] != "truth":
                    target = tg.build(navsim_data.agents_of(labels, row), projections[row])
                    m = target["mask"]
                    gxy, gcls = target["box"][m, :2].astype(np.float64), target["cls"][m]
                    one = {
                        "conf": np.array([1.0]),
                        "cls": agents["cls"][row][q : q + 1],
                        "xyz": agents["xyz"][row][q : q + 1],
                    }
                    _, tp, _ = evaluate.match(one["conf"], one["cls"], one["xyz"][:, :2], gxy, gcls)
                    real += int(tp.any())
                    false_alarm += int(not tp.any())
            entry["ep_lost_by_agent_points"] = {k: v / len(n) * 100 for k, v in sorted(by.items())}
            if a["source"] != "truth":
                entry["switches_by_a_real_agent"] = real
                entry["switches_by_a_false_alarm"] = false_alarm
        record["arms"][arm] = entry
    if official_dir is not None:
        record["official"] = str(official_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    return record


# ------------------------------------------------------------------------- B-1-4, the soft variant

#: B-1-4 (D27; PREDICTIONS.md, Part B · B-1-4): the side-car's agents at the detection threshold, each
#: proposal's score lowered by LAMBDA times its worst confidence-weighted shortfall under a margin that widens
#: with the agent's range; nothing is removed.
SOFT_LAMBDAS = (0.1, 0.3)
SOFT_MARGINS = (0.5, 1.0)
SOFT_WIDENING = (0.0, 0.05)
SOFT_KEEP = 0.5


def soft_settings() -> list:
    """(lambda, margin, widening per metre, horizon): 16 settings."""
    return [
        (lam, m, b, h) for lam in SOFT_LAMBDAS for m in SOFT_MARGINS for b in SOFT_WIDENING for h in HORIZONS
    ]


def soft_name(setting) -> str:
    lam, m, b, h = setting
    return f"l{lam:g}-m{m:g}-b{b:g}-h{h:g}"


def penalty_of(clearance, conf, reach, keep, margin: float, widening: float, horizon: float):
    """(B, K) the worst conf_a * max(0, m_a - clearance) over kept agents and plan times within `horizon`,
    m_a = margin + widening * the agent's range. clearance (B, K, A, T); conf, reach, keep (B, A)."""
    import torch

    m = margin + widening * reach  # B, A
    short = (m[:, None, :, None] - clearance).clamp(min=0) * (conf * keep)[:, None, :, None]
    return (
        short[..., : horizon_steps(horizon)].amax(dim=(2, 3))
        if short.shape[2]
        else torch.zeros(short.shape[:2])
    )


def pick_with_penalty(scores: np.ndarray, penalty: np.ndarray, lam: float) -> np.ndarray:
    """The release's highest score after the penalty, first index on ties."""
    return (np.asarray(scores, np.float64) - lam * np.asarray(penalty, np.float64)).argmax(1).astype(np.int64)


def penalty_arrays(proposals, agents: dict, rows: np.ndarray, chunk: int = 128) -> dict:
    """Per (margin, widening, horizon), (N, K) penalties for `rows`, computed once from the clearances."""
    import torch

    from gallops.drivejepa_perception import geometry

    shapes = [(m, b, h) for m in SOFT_MARGINS for b in SOFT_WIDENING for h in HORIZONS]
    out = {shape: np.zeros((len(rows), 32), np.float32) for shape in shapes}
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            r = rows[i : i + chunk]
            conf = torch.from_numpy(np.asarray(agents["conf"][r], dtype=np.float32))
            keep = conf >= SOFT_KEEP
            xy = torch.from_numpy(np.asarray(agents["xyz"][r][..., :2], dtype=np.float32))
            clearance = geometry.clearances(
                torch.from_numpy(np.asarray(proposals[r], dtype=np.float32)),
                xy,
                torch.from_numpy(np.asarray(agents["lwh"][r][..., :2], dtype=np.float32)),
                torch.from_numpy(np.asarray(agents["vel"][r], dtype=np.float32)),
                torch.from_numpy(np.asarray(agents["path"][r], dtype=np.float32)),
                keep,
            )
            reach = xy.norm(dim=-1)
            for m, b, h in shapes:
                out[(m, b, h)][i : i + len(r)] = penalty_of(clearance, conf, reach, keep, m, b, h).numpy()
    return out


def soft_picks(out_root: Path, split: str) -> tuple:
    """(data, seed -> setting name -> pick) for every B-1-4 setting on a split."""
    data = load_split(out_root, split)
    sources = agent_sources(out_root, split)
    picks = {}
    for k in SEEDS:
        penalties = penalty_arrays(data["proposals"], sources[f"s{k}"][0], data["rows"])
        picks[f"s{k}"] = {
            soft_name(st): pick_with_penalty(data["scores"], penalties[st[1:]], st[0])
            for st in soft_settings()
        }
    return data, picks


def run_soft_grid(out_root: Path, out: Path) -> dict:
    """B-1-4's grid on R-clean's known-command windows, by B-1's value and rule (mean over seeds)."""
    from gallops.drivejepa_b2d.provenance import git_sha

    problems = predictions_committed() + (
        [] if SOFT_SECTION in _committed_predictions() else [f"no '{SOFT_SECTION}'"]
    )
    if problems:
        raise SystemExit("B-1-4 may not read R-clean yet:\n- " + "\n- ".join(problems))
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    data, picks = soft_picks(out_root, "fresh")
    navtest = np.load(Path(out_root) / O_DIRS["navtest"][1] / "labels.npz")
    nmanifest = json.loads((Path(out_root) / O_DIRS["navtest"][1] / "manifest.json").read_text())
    weights, city_record = city_weights(data["city"], np.array(nmanifest["cities"])[navtest["log_id"]])
    failing = ceiling_vetoed(data["pdm"])
    results = {}
    for st in soft_settings():
        name = soft_name(st)
        entry = {f"s{k}": evaluate_pick(data, picks[f"s{k}"][name], None, failing, weights) for k in SEEDS}
        entry["mean_over_seeds"] = {
            key: float(np.mean([entry[f"s{k}"][key] for k in SEEDS]))
            for key in ("value", "minus_release", "switch_rate", "pick_ep")
        }
        results[name] = entry
    record = {
        "pass": "veto-soft-grid",
        "split": "fresh",
        "smoke": False,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": git_sha(),
        "rows": int(len(data["rows"])),
        "inputs": data["inputs"],
        "value": "vendored final, human-filtered, no EC (filtered_final)",
        "weights": city_record,
        "results": results,
    }
    out.mkdir(parents=True)
    (out / "grid.json").write_text(json.dumps(record, indent=1))
    return record


SOFT_FROZEN = "recipes/drivejepa_perception/part_b14_frozen.json"
SOFT_SECTION = "## Part B · B-1-4, the soft veto"


def _committed_predictions() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{PREDICTIONS}"], capture_output=True, text=True
    ).stdout


def soft_frozen_record(grid: dict, grid_sha256: str) -> dict:
    v = {name: e["mean_over_seeds"]["value"] for name, e in grid["results"].items()}
    rates = {name: e["mean_over_seeds"]["switch_rate"] for name, e in grid["results"].items()}
    return {
        "rule": "the setting with the highest R-clean selection value, mean over seeds 0-2 (as B-1); a "
        "tie goes to the fewer switches, then to the grid's order (D27; PREDICTIONS.md, Part B · B-1-4)",
        "source": {"grid_sha256": grid_sha256, "split": grid["split"], "rows": grid["rows"]},
        "weights": grid["weights"],
        "values": {"B-1-4": {name: {"value": v[name], "veto_rate": rates[name]} for name in v}},
        "chosen": {"B-1-4": choose(v, rates)},
    }


def soft_freeze(grid_dir: Path, write: Path) -> dict:
    grid = json.loads((Path(grid_dir) / "grid.json").read_text())
    write = Path(write)
    if write.exists():
        raise SystemExit(f"{write} exists: B-1-4's setting is frozen once")
    record = soft_frozen_record(grid, _sha256(Path(grid_dir) / "grid.json"))
    write.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["chosen"], indent=1))
    return record


def read_navtest_soft(cfg, out_root, grid_dir, verified, anchor_dir, r_dir, out, workers) -> dict:
    """B-1-4 on navtest, once, at its frozen setting: the three seeds scored officially, paired with R."""
    from gallops.drivejepa_perception import navtest

    problems = predictions_committed() + committed_clean(SOFT_FROZEN)
    frozen = load_frozen(REPO / SOFT_FROZEN, grid_dir)
    anchor = json.loads((Path(anchor_dir) / "anchor.json").read_text())
    if not all(anchor["gates"].values()):
        problems.append(f"B-0's anchor does not hold: {anchor['gates']}")
    out = Path(out)
    if (out / "navtest.json").exists():
        problems.append(f"{out} holds a finished read: navtest is read once")
    if problems:
        raise SystemExit("B-1-4 may not read navtest:\n- " + "\n- ".join(problems))
    out.mkdir(parents=True, exist_ok=True)
    name = frozen["chosen"]["B-1-4"]
    setting = {soft_name(st): st for st in soft_settings()}[name]
    data = load_split(out_root, "navtest")
    sources = agent_sources(out_root, "navtest")
    labels_dir = Path(out_root) / O_DIRS["navtest"][1]
    tokens = [str(t) for t in data["token"]]
    pairs, rates = {}, {}
    for k in SEEDS:
        arm = f"B-1-4-s{k}"
        penalties = penalty_arrays(data["proposals"], sources[f"s{k}"][0], data["rows"])
        pick = pick_with_penalty(data["scores"], penalties[setting[1:]], setting[0])
        plans = out / f"plans-{arm}"
        if not plans.exists():
            meta = {"arm": arm, "setting": name, "frozen_sha256": _sha256(REPO / SOFT_FROZEN)}
            navtest.write_plans(
                plans,
                tokens,
                np.asarray(data["proposals"][data["rows"], pick]),
                meta,
                pick=pick,
                row=data["rows"],
            )
        scored = out / f"score-{arm}"
        if not scored.exists():
            navtest.score_plans(cfg, plans, verified, scored, workers, arm)
        pairs[arm] = navtest.paired(scored, r_dir, labels_dir)
        rates[arm] = {"switch_rate": float((pick != data["pick"]).mean())}
    per_seed = {arm: (p["epdms_hf"]["a_minus_b"], p["epdms_hf"]["se"]) for arm, p in pairs.items()}
    record = {
        "pass": "veto-soft-navtest",
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen": frozen["chosen"],
        "pairs": pairs,
        "rates": rates,
        "V6": {
            "per_seed": {arm: {"b14_minus_r": d, "se": se} for arm, (d, se) in per_seed.items()},
            "supports": all(d > 2 * se for d, se in per_seed.values()),
            "refutes": any(d < -2 * se for d, se in per_seed.values()),
        },
    }
    (out / "navtest.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record["V6"], indent=1))
    return record


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.veto")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("clearances", "grid"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True, help="the passes' directory, {out}")
        p.add_argument("--split", choices=("select", "fresh", "navtest"), required=True)
        p.add_argument("--out", required=True)
        p.add_argument(
            "--smoke", action="store_true", help="navtrain's select rows: a smoke run, choosing nothing"
        )
        if name == "grid":
            p.add_argument("--clearances", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--grid", required=True)
    p.add_argument("--write", required=True)
    p = sub.add_parser("navtest", help="B-1 on navtest, once, at the frozen settings")
    p.add_argument("--root", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--verified", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--r", required=True, help="R's official run (navtest score)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=32)
    p = sub.add_parser("soft-grid", help="B-1-4's grid on R-clean")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("soft-freeze")
    p.add_argument("--grid", required=True)
    p.add_argument("--write", required=True)
    p = sub.add_parser("soft-navtest", help="B-1-4 on navtest, once")
    p.add_argument("--root", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--verified", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--r", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=32)
    p = sub.add_parser("analyse", help="the frozen settings taken apart on a split")
    p.add_argument("--root", required=True)
    p.add_argument("--split", choices=("fresh", "navtest"), required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "clearances":
        from gallops.config import SiteConfig

        write_clearances(SiteConfig.load(), Path(args.root), args.split, Path(args.out), args.smoke)
    elif args.cmd == "grid":
        record = run_grid(Path(args.root), args.split, Path(args.clearances), Path(args.out), args.smoke)
        print(json.dumps({k: record[k] for k in ("split", "smoke", "rows", "logs", "weights")}, indent=1))
    elif args.cmd == "freeze":
        freeze(Path(args.grid), Path(args.write))
    elif args.cmd == "soft-grid":
        record = run_soft_grid(Path(args.root), Path(args.out))
        print(json.dumps({k: record[k] for k in ("split", "rows", "weights")}, indent=1))
    elif args.cmd == "soft-freeze":
        soft_freeze(Path(args.grid), Path(args.write))
    elif args.cmd == "soft-navtest":
        from gallops.config import SiteConfig

        read_navtest_soft(
            SiteConfig.load(),
            Path(args.root),
            Path(args.grid),
            Path(args.verified),
            Path(args.anchor),
            Path(args.r),
            Path(args.out),
            args.workers,
        )
    elif args.cmd == "navtest":
        from gallops.config import SiteConfig

        read_navtest(
            SiteConfig.load(),
            Path(args.root),
            Path(args.grid),
            Path(args.verified),
            Path(args.anchor),
            Path(args.r),
            Path(args.out),
            args.workers,
        )
    else:
        record = analyse(Path(args.root), args.split, Path(args.grid), Path(args.out))
        print(
            json.dumps(
                {arm: {k: v for k, v in e.items() if k != "subsets"} for arm, e in record["arms"].items()},
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
