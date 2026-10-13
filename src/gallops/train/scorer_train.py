"""Train the sub-score scorer on cached features, and write weights paddock can load.

    python -m gallops.train.scorer_train --cache <dj_cache/v4> [--cache <dj_cache/v4_nusc>] \\
        --labelled round0_cam <labelled dir> --start round0_cam --drivable <drivable dir> \\
        --checkpoint <released Drive-JEPA .ckpt> --train-run sighted-v1 [--seeds 0 1 2 3 4] \\
        [--epochs 20] [--real mask]

Writes `{weights}/drivejepa_subscore/<train-run>/step_<epochs>/`: `seed_<k>.pt` per seed,
`scorer.json` (the seeds and the input layout) and the `checkpoint.json` sidecar. Paddock's
`drivejepa_subscore` adapter averages the seeds' sigmoids. The model is `subscorer.build`.

**Preflight runs first, every run, and has no skip flag.** A cache is a recording too, and the
camera-frame bug lived for six weeks because every check compared one recording with another.
So each group below compares what this code computes now with what was recorded, or with the
world:

    code      the tree is committed and the tests pass (`preflight.code_checks`)
    labels    3 sampled rendered cache files, relabelled by today's code, equal their .labels.npz
    fidelity  for each START source -- episodes the released weights drove, the weights the
              cache was built with -- the cached pdm_score picks the same plan the live model
              picked, as the labeller recorded it, on >= 95% of ticks
    sight     the cached features depend on the image (`subscorer.sight_from_cache`)
    data      no neuroNCAP benchmark scene, and no scene-0061, among the rendered ticks

The training recipe is the one the folds measured, unchanged: 20 epochs, Adam 1e-4, batch 64,
BCE on the six answers, real nuScenes ticks weighted per `--real`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from gallops import checkpoint as sidecar
from gallops.config import SiteConfig
from gallops.data.nuscenes_real import NEURONCAP_SCENES
from gallops.train import subscorer
from gallops.train.preflight import FIDELITY_VERDICT_TOLERANCE, PreflightFailed, code_checks
from gallops.train.scorer_folds import released_state
from gallops.train.scorer_labels import compute_labels
from gallops.train.swerve import git_sha

REPO = Path(__file__).resolve().parents[3]
LABEL_CHECK_FILES = 3


def labels_check(caches: list[Path], drivable: Path, seed: int = 0) -> tuple[dict, list[str]]:
    """Relabel a few rendered cache files with today's code; they must equal what is on disk."""
    rng = np.random.default_rng(seed)
    files = [
        f
        for c in caches
        for src in sorted(p for p in c.iterdir() if p.is_dir() and p.name not in subscorer.REAL_SOURCES)
        for f in sorted(src.glob("*.npz"))
        if not f.name.endswith(".labels.npz")
        and ".tmp" not in f.name
        and f.with_name(f.name.replace(".npz", ".labels.npz")).exists()
    ]
    if not files:
        return {}, ["labels: no labelled rendered cache file to re-check"]
    picked = [
        files[i] for i in rng.choice(len(files), size=min(LABEL_CHECK_FILES, len(files)), replace=False)
    ]
    problems = []
    for f in picked:
        fresh, _, _ = compute_labels(f, drivable, np.random.default_rng(seed))
        stored = np.load(f.with_name(f.name.replace(".npz", ".labels.npz")))
        for key in subscorer.LABEL_KEYS:
            a, b = np.asarray(fresh[key], dtype=np.float64), np.asarray(stored[key], dtype=np.float64)
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-5, equal_nan=True):
                problems.append(
                    f"labels: {f.parent.name}/{f.name} {key} differs from a relabel by today's code"
                )
    return {"relabelled_files": [f"{f.parent.name}/{f.name}" for f in picked]}, problems


def fidelity_check(d: dict, labelled: dict[str, Path], start: list[str]) -> tuple[dict, list[str]]:
    """Cached pdm_score against the live one the labeller recorded, on START sources."""
    if not start:
        return {}, [
            "fidelity: no --start source, so nothing checks that the cache rebuilt what the model saw; "
            "name a source the released weights drove (a round-0 collection always qualifies)"
        ]
    source = subscorer.source_of(d)
    report, problems = {}, []
    for name in start:
        if name not in labelled:
            problems.append(f"fidelity: --start {name} has no --labelled directory")
            continue
        rows = np.where(source == name)[0]
        if not len(rows):
            problems.append(f"fidelity: no cached ticks from START source {name}")
            continue
        recorded: dict[tuple[str, int], np.ndarray] = {}
        for stem in {e.split("/", 1)[1] for e in d["episode"][rows]}:
            jl = labelled[name] / f"{stem}.jsonl"
            if not jl.exists():
                continue
            for line in jl.read_text().splitlines():
                r = json.loads(line)
                if r.get("pdm_score") is not None:
                    recorded[(stem, int(r["timestamp"]))] = np.asarray(r["pdm_score"], dtype=np.float32)
        same, gaps = [], []
        for i in rows:
            key = (d["episode"][i].split("/", 1)[1], int(d["timestamp"][i]))
            if key not in recorded:
                continue
            live, cached = recorded[key], np.asarray(d["pdm"][i], dtype=np.float32)
            same.append(int(live.argmax()) == int(cached.argmax()))
            gaps.append(float(np.abs(live - cached).max()))
        if not same:
            problems.append(f"fidelity: no cached tick of {name} matches a labelled tick")
            continue
        agree = float(np.mean(same))
        report[name] = {
            "compared": len(same),
            "pick_agrees": round(agree, 4),
            "median_score_gap": round(float(np.median(gaps)), 5),
        }
        if agree < 1 - FIDELITY_VERDICT_TOLERANCE:
            problems.append(
                f"fidelity: on {name} the cached scores pick the live model's plan on only {agree:.1%} of "
                f"{len(same)} ticks -- the cache did not rebuild what the model saw, or {name} was not "
                "driven by these weights"
            )
    return report, problems


def data_check(d: dict) -> tuple[dict, list[str]]:
    source = subscorer.source_of(d)
    rendered = ~np.isin(source, subscorer.REAL_SOURCES)
    leaked = sorted(set(d["scene"][rendered]) & NEURONCAP_SCENES)
    report = {
        "ticks_by_source": dict(Counter(source.tolist())),
        "rendered_ticks_by_scenario": dict(Counter(d["shape"][rendered].tolist())),
        "rendered_scenes": sorted(set(d["scene"][rendered].tolist())),
        "usable": int(d["usable"].sum()),
    }
    problems = [f"data: benchmark scene(s) {leaked} among the training ticks"] if leaked else []
    return report, problems


def preflight(args, caches: list[Path], d: dict) -> dict:
    labelled = {name: Path(p) for name, p in args.labelled}
    report, problems = {}, []
    for name, (part, found) in (
        ("code", code_checks(REPO)),
        ("labels", labels_check(caches, args.drivable)),
        ("fidelity", fidelity_check(d, labelled, args.start)),
        ("sight", subscorer.sight_from_cache(d)),
        ("data", data_check(d)),
    ):
        report[name] = part
        problems += found
    print("  preflight: " + json.dumps(report), flush=True)
    if problems:
        raise PreflightFailed(problems)
    return report


def load(caches: list[Path]) -> dict:
    parts = [subscorer.reduce(c) for c in caches]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--labelled", nargs=2, action="append", default=[], metavar=("SOURCE", "DIR"))
    parser.add_argument("--start", action="append", default=[], help="a source the released weights drove")
    parser.add_argument("--drivable", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True, help="the released Drive-JEPA checkpoint (MLP init)")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--real", choices=("all", "mask", "weight"), default="mask")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.time()
    d = load(args.cache)
    print(
        f"  {len(d['scene'])} cached ticks from {len(args.cache)} cache(s) in {time.time() - started:.0f} s",
        flush=True,
    )
    checked = preflight(args, args.cache, d)

    import torch

    source = subscorer.source_of(d)
    train_idx = np.where(d["usable"] & (source != "nusc_holdout"))[0]
    arrays = (
        torch.from_numpy(d["own"]),
        torch.from_numpy(d["probe"]),
        torch.from_numpy(subscorer.geometry(d["plans"], d["speed"])),
        torch.from_numpy(subscorer.targets(d)),
        torch.from_numpy(subscorer.channel_weights(source, args.real)),
    )
    state = released_state(args.checkpoint, torch)
    out = (
        Path(SiteConfig.load().resolve("{weights}"))
        / "drivejepa_subscore"
        / args.train_run
        / f"step_{args.epochs:09d}"
    )
    out.mkdir(parents=True, exist_ok=True)
    params = 0
    for seed in args.seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        model = subscorer.build(torch, state).to(args.device)
        layout = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        if layout != subscorer.layout():
            raise SystemExit(f"model layout drifted from subscorer.layout(): {layout}")
        subscorer.fit(
            model,
            arrays,
            train_idx,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            seed=seed,
            device=args.device,
            torch=torch,
        )
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / f"seed_{seed}.pt")
        params = sum(p.numel() for p in model.parameters())
        print(
            f"  seed {seed}: {len(train_idx)} ticks x {args.epochs} epochs in {time.time() - t0:.0f} s",
            flush=True,
        )
    (out / "scorer.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "channels": list(subscorer.CHANNELS),
                "layout": {k: list(v) for k, v in subscorer.layout().items()},
                "inputs": {
                    "own": "per plan, max over its 8 last-stage refiner tokens; "
                    "32 native + 32 swerves (second pass)",
                    "probe": "32 probe paths x 8 waypoint tokens, probe-major, from the second pass",
                    "geo": "plan (x, y) / 20 m (16), headings (8), ego speed / 10 m/s (1)",
                },
                "vocab": "swerves = vocab_v1, probes = probe_v1 "
                "(gallops experiments/2026-10-blind-scorer/ops/dj_cache)",
                "pick": "nc^4 * on_road * (5 progress + 5 no_ttc + 2 in_lane) / 12 "
                "over the seeds' mean sigmoids",
            },
            indent=1,
        )
    )
    real = bool(np.isin(source[train_idx], subscorer.REAL_SOURCES).any())
    trained_on = [
        sidecar.TrainedOn(
            dataset="neuroncap-actor-insertion",
            split="5 non-benchmark nuScenes train scenes (0053 0058 0064 0070 0075), "
            "rendered with an inserted car",
        )
    ]
    if real:
        trained_on.append(
            sidecar.TrainedOn(
                dataset="nuscenes", split="train minus neuroNCAP and reconstructed scenes, minus 10% held out"
            )
        )
    sidecar.write(
        out,
        sidecar.Checkpoint(
            model="drivejepa_subscore",
            step=args.epochs,
            train_git_sha=git_sha(REPO),
            train_run=args.train_run,
            trained_on=tuple(trained_on),
            params_m=round(params / 1e6, 3),
            precision="fp32",
            notes={
                "preflight": checked,
                "caches": [str(c) for c in args.cache],
                "real": args.real,
                "recipe": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seeds": args.seeds},
                "init": "released Drive-JEPA scorer MLP for the own-token columns; "
                "probe and geometry columns at zero",
            },
        ),
    )
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
