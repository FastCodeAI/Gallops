"""Part A: train / selection split over the KIT Multimodal train scenes on disk.

    ./run.sh src/splits.py            # writes out/splits.json + out/split_locations.json

The selection set is 50 scenes drawn with a fixed seed from the sorted list of
train scenes present on disk; the rest is "train".  Neither touches validation,
overlap_train_val, test or test-e2e (asserted below and again by load_splits()).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import common as C

SEED = 20261016
N_SELECTION = 50
SPLITS_JSON = C.OUT / "splits.json"


def split_ids(name: str) -> list[str]:
    path = C.SPLIT_DIR / C.SPLIT_FILES[name]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def scenes_on_disk(split_dir_name: str) -> dict[str, Path]:
    """scene id -> folder, across every root; the first root wins if both hold it."""
    found: dict[str, Path] = {}
    for root in C.KIT_ROOTS:
        folder = root / split_dir_name
        if folder.is_dir():
            for scene in sorted(folder.iterdir()):
                if scene.is_dir():
                    found.setdefault(scene.name, scene)
    return found


def forbidden_ids() -> set[str]:
    return set().union(*(split_ids(n) for n in ("val", "overlap_train_val", "test", "test-e2e")))


def scene_location(scene: Path) -> tuple[float, float, float, float]:
    """(origin lat, origin lon, first-pose x, first-pose y).  poses.txt is TUM in a
    local metric frame anchored at maps/origin.json."""
    origin = json.loads((scene / "maps/origin.json").read_text())
    with open(scene / "poses.txt") as handle:
        first = handle.readline().split()
    return origin["latitude"], origin["longitude"], float(first[1]), float(first[2])


def build() -> dict:
    listed_train = split_ids("train")
    on_disk = scenes_on_disk("train")
    available = sorted(s for s in listed_train if s in on_disk)
    missing = sorted(set(listed_train) - set(on_disk))
    extra = sorted(set(on_disk) - set(listed_train))
    assert not extra, f"scenes under data/train not in train.txt: {extra}"

    rng = random.Random(SEED)
    selection = sorted(rng.sample(available, N_SELECTION))
    train = sorted(set(available) - set(selection))

    bad = forbidden_ids()
    assert not set(train) & bad and not set(selection) & bad, "train/selection overlap a held-out split"
    assert not set(train) & set(selection)

    payload = {
        "seed": SEED,
        "rule": f"random.Random({SEED}).sample(sorted(train scenes on disk), {N_SELECTION})",
        "source_split_file": str(C.SPLIT_DIR / "train.txt"),
        "source_split_file_sha256": C.sha256_file(C.SPLIT_DIR / "train.txt"),
        "roots": [str(r) for r in C.KIT_ROOTS],
        "listed_in_train_txt": len(listed_train),
        "on_disk": len(available),
        "missing_from_disk": missing,
        "train": train,
        "selection": selection,
        "scene_paths": {s: str(on_disk[s]) for s in available},
        "overlap_checked_against": {n: len(split_ids(n)) for n in ("val", "overlap_train_val", "test", "test-e2e")},
    }
    return payload


def load_splits() -> dict:
    """Read splits.json, check its hash line and the held-out guard again."""
    data = json.loads(SPLITS_JSON.read_text())
    recorded = data.pop("sha256_of_content")
    assert recorded == _content_hash(data), "splits.json edited after it was written"
    bad = forbidden_ids()
    assert not (set(data["train"]) | set(data["selection"])) & bad
    data["sha256_of_content"] = recorded
    return data


def _content_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def location_report(data: dict) -> dict:
    """Do train and selection cover the same places?  The geo split's own
    grouping is not shipped, so places are read from each scene's map origin and
    first pose, binned into 1 km cells."""
    rows = {}
    for subset in ("train", "selection"):
        for s in data[subset]:
            lat, lon, x, y = scene_location(Path(data["scene_paths"][s]))
            rows[s] = {"subset": subset, "origin": f"{lat:.5f},{lon:.5f}", "x": x, "y": y,
                       "cell_1km": f"{int(np.floor(x / 1000))},{int(np.floor(y / 1000))}"}
    origins = {sub: Counter(r["origin"] for r in rows.values() if r["subset"] == sub) for sub in ("train", "selection")}
    cells = {sub: Counter(r["cell_1km"] for r in rows.values() if r["subset"] == sub) for sub in ("train", "selection")}
    sel_cells_in_train = sum(n for c, n in cells["selection"].items() if c in cells["train"])
    return {
        "origins": {k: dict(v) for k, v in origins.items()},
        "cells_1km": {k: dict(v) for k, v in cells.items()},
        "selection_scenes_in_a_train_cell": sel_cells_in_train,
        "train_cells_not_in_selection": len(set(cells["train"]) - set(cells["selection"])),
        "per_scene": rows,
    }


def main() -> None:
    C.OUT.mkdir(exist_ok=True)
    data = build()
    data["sha256_of_content"] = _content_hash(data)
    SPLITS_JSON.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    data = load_splits()
    print(f"train.txt {data['listed_in_train_txt']}, on disk {data['on_disk']}, missing {data['missing_from_disk']}")
    print(f"train {len(data['train'])}  selection {len(data['selection'])}  sha256(content) {data['sha256_of_content']}")
    print(f"splits.json file sha256 {C.sha256_file(SPLITS_JSON)}")
    loc = location_report(data)
    (C.OUT / "split_locations.json").write_text(json.dumps(loc, indent=1, sort_keys=True) + "\n")
    print("origins:", json.dumps(loc["origins"]))
    print(f"1 km cells: train {len(loc['cells_1km']['train'])}, selection {len(loc['cells_1km']['selection'])}; "
          f"selection scenes inside a train cell {loc['selection_scenes_in_a_train_cell']}/50")


if __name__ == "__main__":
    main()
