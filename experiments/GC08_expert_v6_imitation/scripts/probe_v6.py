"""Step 4 probe: can Drive-JEPA's own image-conditioned features tell what expert v6 does?

Features per tick, from cache v4 `round0_cam` (built from the lossless frames the camera-fixed model saw):
the 32 candidates' last-stage tokens and the 32 far-ahead probe paths' tokens, each mean- and max-pooled,
plus speed and acceleration. Targets: v6's labels on the same ticks (labelled_r0cam_v6), as
  kind  -- go (continue/resume) / brake (brake/stop) / swerve (brake+swerve); least-impact dropped
  side  -- on swerve ticks, left or right
  onset -- per episode, the first tick predicted to act vs the label's
Linear (logistic) classifiers, leave-one-scene-out; a linear read-out is a lower bound on what a trained
head can get from the same features. Speed/accel alone are the baseline: if the features add nothing over
them, the image is not carrying the decision.

    <drive-jepa python> probe_v6.py <cache dir> <labelled dir> <out json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

GROUP = {"continue": 0, "resume": 0, "brake": 1, "stop": 1, "brake+swerve": 2}
SCENES = ("0053", "0058", "0064", "0070", "0075")


def load(cache: Path, labelled: Path):
    rows = []
    for jl in sorted(labelled.glob("*.jsonl")):
        z = np.load(cache / "round0_cam" / f"{jl.stem}.npz")
        index = {int(t): i for i, t in enumerate(z["timestamp"])}
        native = z["native_tokens"].astype(np.float32)
        probe = z["vocab_tokens"][:, 32:64].astype(np.float32)
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            i = index.get(int(r["timestamp"]))
            if i is None or r["expert_kind"] not in GROUP:
                continue
            feats = np.concatenate(
                [native[i].mean((0, 1)), native[i].max((0, 1)), probe[i].mean((0, 1)), probe[i].max((0, 1))]
            )
            rows.append(
                {
                    "episode": jl.stem,
                    "scene": jl.stem.split("-")[1].split("__")[0],
                    "t": int(r["timestamp"]),
                    "x": feats,
                    "ego": np.array([r["speed"], r["accel"]], dtype=np.float32),
                    "kind": GROUP[r["expert_kind"]],
                    "side": int(r.get("expert_side") or 0),
                }
            )
    return rows


def fit_eval(train, test, use_image: bool, target: str):
    def matrix(rows):
        if use_image:
            return np.stack([np.concatenate([r["x"], r["ego"]]) for r in rows])
        return np.stack([r["ego"] for r in rows])

    y_train = np.array([r[target] for r in train])
    y_test = np.array([r[target] for r in test])
    if len(set(y_train)) < 2 or not len(test):
        return None, None
    scaler = StandardScaler().fit(matrix(train))
    model = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    model.fit(scaler.transform(matrix(train)), y_train)
    predicted = model.predict(scaler.transform(matrix(test)))
    return balanced_accuracy_score(y_test, predicted), predicted


def main() -> None:
    cache, labelled, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = load(cache, labelled)
    report = {"ticks": len(rows), "folds": {}}
    for held in SCENES:
        train = [r for r in rows if r["scene"] != held]
        test = [r for r in rows if r["scene"] == held]
        fold = {"test_ticks": len(test)}
        for name, image in (("ego_only", False), ("image", True)):
            acc, predicted = fit_eval(train, test, image, "kind")
            fold[f"kind_bal_acc_{name}"] = acc
            if image and predicted is not None:
                # onset: first tick of each episode predicted not-go vs the label's first not-go tick
                delays = []
                for episode in sorted({r["episode"] for r in test}):
                    idx = [i for i, r in enumerate(test) if r["episode"] == episode]
                    lab = next((test[i]["t"] for i in idx if test[i]["kind"] != 0), None)
                    pred = next((test[i]["t"] for i in idx if predicted[i] != 0), None)
                    if lab is not None and pred is not None:
                        delays.append((pred - lab) / 1e6)
                fold["onset_delay_s_median"] = float(np.median(delays)) if delays else None
                fold["onset_delay_s_abs_median"] = float(np.median(np.abs(delays))) if delays else None
            swerve_train = [r for r in train if r["kind"] == 2 and r["side"]]
            swerve_test = [r for r in test if r["kind"] == 2 and r["side"]]
            acc, _ = fit_eval(swerve_train, swerve_test, image, "side")
            fold[f"side_bal_acc_{name}"] = acc
            fold["side_test_ticks"] = len(swerve_test)
        report["folds"][held] = fold
        print(held, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fold.items()}), flush=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
