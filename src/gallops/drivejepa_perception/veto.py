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


def split_rows(split: str, labels_dir: Path) -> np.ndarray:
    """The rows B-1 reads: every row of fresh and navtest; navtrain's select rows for the smoke."""
    from gallops.drivejepa_perception import navsim_data

    labels = np.load(Path(labels_dir) / "labels.npz")
    manifest = json.loads((Path(labels_dir) / "manifest.json").read_text())
    if navsim_data.source_of(manifest) != ("navtrain" if split == "select" else split):
        raise SystemExit(f"{labels_dir} is not {split}'s labels pass")
    if split != "select":
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
    rows = split_rows(split, labels_dir)
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
    args = parser.parse_args(argv)
    if args.cmd == "clearances":
        from gallops.config import SiteConfig

        write_clearances(SiteConfig.load(), Path(args.root), args.split, Path(args.out), args.smoke)
    elif args.cmd == "grid":
        record = run_grid(Path(args.root), args.split, Path(args.clearances), Path(args.out), args.smoke)
        print(json.dumps({k: record[k] for k in ("split", "smoke", "rows", "logs", "weights")}, indent=1))
    else:
        freeze(Path(args.grid), Path(args.write))


if __name__ == "__main__":
    main()
