"""Scene lengths for train, selection, val and overlap_train_val (metadata only:
timestamp.reference.txt and the front-camera file listing; no images).

    ./run.sh src/scene_lengths.py     # writes out/scene_lengths.json + out/scene_lengths_hist.png

Moments ("keyframes") use the baseline's own rule, called on the devkit class:
KITScenesSceneAdapter.valid_keyframes(num_future_frames=10, stride=90, first_frame=40)
(run_kitscenes_inference.py:236-238, kitscenes_adapter.py:261-273).  That method
reads only self.num_frames, which the adapter sets to len(timestamp.reference.txt)
(:200-202), so it is called on a stand-in carrying that one field.
"""

from __future__ import annotations

import json
import pickle
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import common as C
import splits as S

ADP = C.kit_adapter_module()
FRONT = ADP.CAMERA_MAP["cam_f0"]


def scene_meta(path: str) -> dict:
    p = Path(path)
    ts = np.loadtxt(p / "timestamp.reference.txt", ndmin=1)
    n_jpg = sum(1 for _ in (p / FRONT).glob("*.jpg"))
    return {"frames": int(len(ts)), "front_jpgs": n_jpg,
            "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
            "median_dt_s": float(np.median(np.diff(ts))) if len(ts) > 1 else None}


def moments(n_frames: int) -> list[int]:
    stand_in = types.SimpleNamespace(num_frames=n_frames)
    return ADP.KITScenesSceneAdapter.valid_keyframes(stand_in, num_future_frames=10, stride=90, first_frame=40)


def baseline_scene_paths() -> dict[str, dict[str, str]]:
    """split -> {scene id: folder} for val / overlap_train_val across Naren's roots."""
    out = {}
    for split, folder in (("val", "val"), ("overlap_train_val", "overlap_train_val")):
        listed = S.split_ids(split)
        found = S.scenes_on_disk(folder)
        out[split] = {s: str(found[s]) for s in listed if s in found}
        missing = sorted(set(listed) - set(found))
        assert not missing, f"{split}: {len(missing)} scenes not on disk: {missing[:3]}"
    return out


def stats(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"n": int(len(v)), "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()),
            "mean": round(float(v.mean()), 2), "sum": float(v.sum())}


def main() -> None:
    data = S.load_splits()
    sets = {
        "train": {s: data["scene_paths"][s] for s in data["train"]},
        "selection": {s: data["scene_paths"][s] for s in data["selection"]},
        **baseline_scene_paths(),
    }
    per_scene = {}
    with ThreadPoolExecutor(16) as pool:
        for name, scenes in sets.items():
            metas = list(pool.map(scene_meta, scenes.values()))
            per_scene[name] = dict(zip(scenes.keys(), metas))

    result = {"rule": "valid_keyframes(num_future_frames=10, stride=90, first_frame=40)", "sets": {}}
    for name, scenes in per_scene.items():
        frames = [m["frames"] for m in scenes.values()]
        dur = [m["duration_s"] for m in scenes.values()]
        mom = {s: moments(m["frames"]) for s, m in scenes.items()}
        jpg_mismatch = {s: (m["frames"], m["front_jpgs"]) for s, m in scenes.items() if m["frames"] != m["front_jpgs"]}
        result["sets"][name] = {
            "scenes": len(scenes),
            "frames": stats(frames),
            "duration_s": stats(dur),
            "median_frame_dt_s": float(np.median([m["median_dt_s"] for m in scenes.values()])),
            "moments_total": sum(len(v) for v in mom.values()),
            "moments_per_scene_hist": dict(sorted(Counter(len(v) for v in mom.values()).items())),
            "pairs_from_lengths": sum(min(m["frames"], m["front_jpgs"]) - 5 for m in scenes.values()),
            "frames_vs_front_jpg_mismatch": jpg_mismatch,
        }
        if name in ("val", "overlap_train_val"):
            result["sets"][name]["moments_per_scene"] = {s: len(v) for s, v in mom.items()}
        result["sets"][name]["_moments"] = mom

    # Cross-check against the baseline pickle (read-only).
    assert C.sha256_file(C.BASELINE_PKL) == C.BASELINE_PKL_SHA256
    with open(C.BASELINE_PKL, "rb") as fh:
        rows = pickle.load(fh)
    pkl = Counter((r["split"], r["scene_id"]) for r in rows)
    pkl_kf = {(r["split"], r["scene_id"], int(r["keyframe_index"])) for r in rows}
    ours_kf = {(sp, s, k) for sp in ("val", "overlap_train_val") for s, ks in result["sets"][sp]["_moments"].items() for k in ks}
    result["baseline_check"] = {
        "pkl_rows": len(rows),
        "pkl_by_split": dict(Counter(r["split"] for r in rows)),
        "rule_by_split": {sp: result["sets"][sp]["moments_total"] for sp in ("val", "overlap_train_val")},
        "same_keyframes_as_pkl": ours_kf == pkl_kf,
        "in_rule_not_pkl": sorted(ours_kf - pkl_kf)[:10],
        "in_pkl_not_rule": sorted(pkl_kf - ours_kf)[:10],
        "scenes_with_zero_moments": {sp: sorted(s for s, ks in result["sets"][sp]["_moments"].items() if not ks)
                                     for sp in ("val", "overlap_train_val")},
        "pkl_scenes": len({k[1] for k in pkl}),
    }
    for v in result["sets"].values():
        v.pop("_moments")
    (C.OUT / "scene_lengths.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("moments_per_scene",)} for k, v in result["sets"].items()}, indent=1))
    print(json.dumps(result["baseline_check"], indent=1))
    histogram(per_scene)


def histogram(per_scene) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharex=True)
    bins = np.arange(0, 75, 2.5)
    for ax, (name, scenes) in zip(axes, per_scene.items()):
        d = [m["duration_s"] for m in scenes.values()]
        ax.hist(d, bins=bins, color="#4a6fa5")
        ax.set_title(f"{name} (n={len(d)}, median {np.median(d):.1f} s)", fontsize=9)
        ax.set_xlabel("scene duration (s)")
    axes[0].set_ylabel("scenes")
    fig.tight_layout()
    fig.savefig(C.OUT / "scene_lengths_hist.png", dpi=110)


if __name__ == "__main__":
    main()
