"""Clip-clustered intervals for the two 930-step runs' carried-against-reset pick grade, written after review.

Not part of any rule written beforehand: the reports of the collapse fix and the recall lesson gave the
two means only. This script was added on 12 Nov 2026 in answer to the review of gallops #36, and
computes the interval the first training run's report gave for its own gap, by the same function
(`s7_pilot_final.bootstrap`, copied below byte for byte apart from its two constants, because that
module imports the training code).

Inputs, read only: each run's per-moment dumps (the scorer's chosen path per moment, `<name>.npz`, and
the 32 paths' grades, `<name>_grades.npy`) and the picking set's index. The graded moments are the
picking set's in_picking rows with a label, in index order, as the runs' own checks take them.

    python interval_after_review.py {experiment_store} out.json

The first training run's two dumps go through the same code first, and the script stops unless they
give that run's recorded interval exactly (`04_carried_vs_wiped_training_run/tables/pilot_final.json`):
the check that this is the same computation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B_DRAWS, B_SEED = 2000, 0

#: (name, carried dump, reset or wiped dump), relative to the experiment's store
PAIRS = [
    ("first training run: carried against wiped (check)", "runs/pilot_seed0/carry/dumps/p01.2", "runs/pilot_seed0/wiped/dumps/p01.2"),
    ("collapse fix, step 930: carried against reset", "runs/build4_short/dumps/step0930", "runs/build4_short/dumps/step0930_wiped"),
    ("recall lesson, step 930: carried against reset", "runs/build5_short/dumps/step0930_carried", "runs/build5_short/dumps/step0930_reset"),
]
#: the first training run's recorded interval (pilot_final.json: comparison.bootstrap.overall.ci95)
RECORDED = [-0.007629073085026701, 0.03150194492695941]


def bootstrap(a: dict, b: dict, clip_of: dict, subset: set | None):
    """clip-clustered paired bootstrap of mean(a) - mean(b) over the moments (keys of a and b), clusters = clips; 95 % percentile interval."""
    keys = [k for k in a if k in b and (subset is None or k in subset)]
    if not keys:
        return None
    clips = sorted({clip_of[k] for k in keys}); by = {c: [] for c in clips}
    for k in keys:
        by[clip_of[k]].append(a[k] - b[k])
    arr = [np.array(by[c]) for c in clips]; rng = np.random.RandomState(B_SEED); diffs = []
    for _ in range(B_DRAWS):
        pick = rng.randint(0, len(arr), size=len(arr)); v = np.concatenate([arr[i] for i in pick]); diffs.append(v.mean())
    d = np.array(diffs); point = float(np.mean(np.concatenate(arr)))
    return {"moments": len(keys), "clips": len(clips), "point": point, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "draws": B_DRAWS}


def pick_grades(store: Path, dump: str, graded: pd.DataFrame) -> dict:
    """{sample_id: final grade of the chosen path} at the graded picking moments, in index order."""
    z = np.load(store / f"{dump}.npz", allow_pickle=True)
    grades = np.load(store / f"{dump}_grades.npy")
    chosen = dict(zip(list(z["token"]), z["chosen"]))
    assert len(graded) == grades.shape[0], (dump, len(graded), grades.shape)
    picks = np.array([chosen[t] for t in graded.sample_id])
    values = grades[np.arange(len(graded)), picks, -1].astype(np.float64)
    return dict(zip(graded.sample_id, values))


def main() -> None:
    store, out = Path(sys.argv[1]), Path(sys.argv[2])
    index = pd.read_parquet(store / "cache/picking/index_picking.parquet")
    graded = index[index.in_picking & index.has_label2]
    clip_of = dict(zip(graded.sample_id, graded["clip"]))
    result = {"note": __doc__.split("\n\n")[0], "pairs": []}
    for name, carried, other in PAIRS:
        a, b = pick_grades(store, carried, graded), pick_grades(store, other, graded)
        r = bootstrap(a, b, clip_of, None)
        r.update(name=name, carried_mean=float(np.mean(list(a.values()))), other_mean=float(np.mean(list(b.values()))))
        result["pairs"].append(r)
        print(json.dumps(r))
    check = result["pairs"][0]["ci95"]
    if check != RECORDED:
        raise SystemExit(f"the first run's interval is {check}, not the recorded {RECORDED}: not the same computation")
    result["check"] = "the first training run's recorded interval reproduced exactly"
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
