"""B-0-3, the headroom check: how much room there is, before anything is spent trying to fill it.

    python -m gallops.drivejepa_perception.headroom count --root {out} --split navtest \
        --official {out}/part-b/b0/navtest-R --out {out}/part-b/b0/headroom-navtest.json
    python -m gallops.drivejepa_perception.headroom profile --root {out} --out {out}/part-b/b0/profile.json

docs/drivejepa-perception-sidecar/part-b/B-0-foundations.md, Q7 and B-0-3. For every scene where the
release's pick fails NC or TTC (on navtest by the official scorer's own per-token columns; on R-clean, which
has no official run, by the vendored labels):
- **room for a better pick:** is there a proposal among its 32 the vendored labels pass on both NC and TTC?
- **room for an agent input:** is the object behind the failure (the vendored scorer names it: the
  at-fault collision's track, else the TTC track) a labelled agent in CAM_F0's view within 50 m, the
  side-car's targets (`targets.target_index`)? It may instead be a static object (no agent input names
  it), outside the view or beyond 50 m, or not among the labels at all;
- **room for this side-car:** did seed k find that agent (`evaluate.match`, 2 m, confidence 0.5)?
And, beside them, the best of the 32 against the pick, overall. This reads the release's labels only and
chooses nothing (B-0, B-0-3).

`profile` puts R-clean beside navtest (B-0-1, the R-clean split's check): cities, agents and targets per
window by class, the share of labelled paths that last 1 s, the ego's speed, the commands, the three
subsets, and the release pick's failure rates and score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gallops.drivejepa_perception import evaluate, navsim_data
from gallops.drivejepa_perception import targets as tg

NC, TTC, FINAL = 0, 5, 8
NO_AGENT = -1
#: The kinds of object behind a failure.
KINDS = ("in_view_50m", "static", "outside_view_or_range", "not_in_labels", "unnamed")


def failure_kind(agents: dict, projection: np.ndarray, track: int) -> tuple:
    """(kind, its index among the side-car's targets or -1) of the labelled object with `track`."""
    if track == NO_AGENT:
        return "unnamed", -1
    where = np.flatnonzero(agents["track"] == track)
    if not len(where):
        return "not_in_labels", -1
    a = int(where[0])
    if agents["cls"][a] >= tg.NUM_CLASSES:
        return "static", -1
    index = tg.target_index(agents, projection)
    slot = np.flatnonzero(index == a)
    return ("in_view_50m", int(slot[0])) if len(slot) else ("outside_view_or_range", -1)


def found_by(agents_file: dict, row: int, target: dict, slot: int) -> bool:
    """Did the side-car's detections on `row` hit target `slot` (the readout's own match)?"""
    m = target["mask"]
    gxy, gcls = target["box"][m, :2].astype(np.float64), target["cls"][m]
    _, _, hit_of_gt = evaluate.match(
        agents_file["conf"][row], agents_file["cls"][row], agents_file["xyz"][row][:, :2], gxy, gcls
    )
    return bool(hit_of_gt[slot] >= 0)


def count(root: Path, split: str, official_dir, seeds=(0, 1, 2)) -> dict:
    """The headroom counts on a split (module docstring)."""
    from gallops.drivejepa_perception import navtest as nt
    from gallops.drivejepa_perception import veto

    root = Path(root)
    tok, lab, pdm_name, agents_name = veto.O_DIRS[split]
    labels = dict(np.load(root / lab / "labels.npz"))
    pdm = np.load(root / pdm_name / "pdm.npz")
    cand, human = pdm["candidates"], pdm["human"]
    pick = np.load(root / tok / "pick.npy").astype(np.int64)
    n = np.arange(len(pick))
    projections = tg.front_projection(np.load(root / tok / "lidar2img.npy"))
    tokens = [str(t) for t in labels["token"]]
    if official_dir is not None:
        m = json.loads((Path(official_dir) / "manifest.json").read_text())
        official = nt.per_token(str(Path(official_dir) / m["csv"]))
        nc_fail = np.array([official[t]["NC"] < 1 for t in tokens])
        ttc_fail = np.array([official[t]["TTC"] < 1 for t in tokens])
        judged_by = "official"
    else:
        nc_fail, ttc_fail = cand[n, pick, NC] < 1, cand[n, pick, TTC] < 1
        judged_by = "vendored"
    vend_nc, vend_ttc = cand[n, pick, NC] < 1, cand[n, pick, TTC] < 1
    failing = nc_fail | ttc_fail
    safe = (cand[..., NC] == 1) & (cand[..., TTC] == 1)
    has_safe = safe.any(1)
    agent_files = {k: np.load(root / agents_name.format(seed=k) / "agents.npz") for k in seeds}

    kinds = {k: 0 for k in KINDS}
    found = {f"s{k}": 0 for k in seeds}
    rows_with_room = np.flatnonzero(failing & has_safe)
    for r in rows_with_room:
        agents = navsim_data.agents_of(labels, int(r))
        track = int(pdm["collision_agent"][r, pick[r]])
        if track == NO_AGENT and "ttc_agent" in pdm.files:
            track = int(pdm["ttc_agent"][r, pick[r]])
        kind, slot = failure_kind(agents, projections[r], track)
        kinds[kind] += 1
        if kind == "in_view_50m":
            target = tg.build(agents, projections[r])
            for k in seeds:
                found[f"s{k}"] += int(found_by(agent_files[k], int(r), target, slot))
    best = cand[..., FINAL].max(1)
    filtered_pick = veto.filtered_final(cand[n, pick], human)
    filtered_best = veto.filtered_final(cand, human[:, None]).max(1)
    terms = ("nc", "dac", "tlc", "ddc", "ep", "ttc", "lk", "hc")
    return {
        "split": split,
        "rows": int(len(pick)),
        "failures_judged_by": judged_by,
        "pick_fails_by_term_vendored": {t: float((cand[n, pick, i] < 1).mean()) for i, t in enumerate(terms)},
        "pick_fails_nc": int(nc_fail.sum()),
        "pick_fails_ttc": int(ttc_fail.sum()),
        "pick_fails_nc_or_ttc": int(failing.sum()),
        "official_vendored_disagree": None
        if official_dir is None
        else {"nc": int((nc_fail != vend_nc).sum()), "ttc": int((ttc_fail != vend_ttc).sum())},
        "with_a_safe_proposal": int(len(rows_with_room)),
        "by_object_behind_the_failure": kinds,
        "in_view_found_by_seed": found,
        "best_of_32_minus_pick_vendored_final": float((best - cand[n, pick, FINAL]).mean() * 100),
        "best_of_32_minus_pick_filtered": float((filtered_best - filtered_pick).mean() * 100),
        "share_of_32_safe": float(safe.mean()),
    }


def profile(root: Path) -> dict:
    """R-clean beside navtest: is the split where settings are chosen like the one they are read on?"""
    from gallops.drivejepa_perception import veto

    root = Path(root)
    out = {}
    for split in ("fresh", "navtest"):
        tok, lab, pdm_name, _ = veto.O_DIRS[split]
        labels = dict(np.load(root / lab / "labels.npz"))
        manifest = json.loads((root / lab / "manifest.json").read_text())
        cities = np.array(manifest["cities"])[labels["log_id"]]
        projections = tg.front_projection(np.load(root / tok / "lidar2img.npy"))
        proposals = np.load(root / tok / "proposals.npy")
        pdm = np.load(root / pdm_name / "pdm.npz")
        pick = np.load(root / tok / "pick.npy").astype(np.int64)
        n = np.arange(len(pick))
        per_class = np.bincount(labels["cls"].astype(np.int64), minlength=4) / len(n)
        in_view = [len(tg.target_index(navsim_data.agents_of(labels, r), projections[r])) for r in n]
        ego = np.load(root / tok / "ego_status.npy")[:, -1]
        speed = labels["ego_speed"][:, 0]
        subsets = navsim_data.subsets(labels, proposals)
        chosen = pdm["candidates"][n, pick]
        out[split] = {
            "rows": int(len(n)),
            "logs": len(manifest["logs"]),
            "city_share": {c: float((cities == c).mean()) for c in sorted(set(cities.tolist()))},
            "agents_per_window": dict(zip(navsim_data.CLASSES, per_class.round(3).tolist())),  # noqa: B905
            "targets_in_view_per_window": float(np.mean(in_view)),
            "path_ok_share": manifest["path_ok_share"],
            "ego_speed_quantiles": np.quantile(speed, [0.1, 0.5, 0.9]).round(2).tolist(),
            "stopped_share": float((speed < 0.5).mean()),
            "command_share": (ego[:, 7:11].mean(0)).round(3).tolist(),
            "subsets": {k: float(v.mean()) for k, v in subsets.items()},
            "release_pick_vendored": {
                "nc_fail": float((chosen[:, NC] < 1).mean()),
                "ttc_fail": float((chosen[:, TTC] < 1).mean()),
                "final": float(chosen[:, FINAL].mean()),
                "filtered_final": float(veto.filtered_final(chosen, pdm["human"]).mean()),
            },
        }
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.headroom")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("count")
    p.add_argument("--root", required=True)
    p.add_argument("--split", choices=("navtest", "fresh"), required=True)
    p.add_argument("--official", default="", help="navtest: the release's official run (`navtest score`)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    if args.cmd == "count":
        record = count(Path(args.root), args.split, Path(args.official) if args.official else None)
    else:
        record = profile(Path(args.root))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
