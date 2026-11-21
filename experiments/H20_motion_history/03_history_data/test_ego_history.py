"""Tests for ego_history.py. Run: taskset -c 0-1 nice -n 19 <alpasim-venv>/bin/python -I test_ego_history.py [clip_uuid chunk]

T1 leak: poses after t are never read (NaN-poisoned future and a truncated pose list both give bit-identical output).
T2 relative pose: the copies equal drivejepa.planner._yaw_of / _relative_pose of adapters/drivejepa014 AND drivejepa018bp06 (source text
   identical, and numerically identical on random poses).
T3 interpolation: _Poses.at equals the converter's own Ego.T (class source taken from s1_convert.py by ast, not re-typed), on a real clip.
T4 shape / validity on a real clip: (8, 6) float32, rows before the first pose are all zero with valid 0, every value finite.
"""

from __future__ import annotations

import ast
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ego_history as H  # noqa: E402

AS = Path("{paddock_sid}/Alpasim")
RD = AS / "analysis_3.0/Retrain-data"
RAW0 = Path("{dataset_store}/s0_census/raw")   # s0_census/common.py:47 RAW_NAS = s1_common.RAW0
res = {}


def converter_ego_class():
    src = (RD / "s1_converter/s1_convert.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Ego")
    ns = {"np": np, "Rotation": Rotation, "Slerp": Slerp}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "s1_convert.py:Ego", "exec"), ns)
    return ns["Ego"]


def planner_funcs(pkg: str):
    src = (AS / "adapters" / pkg / "planner.py").read_text()
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_yaw_of", "_relative_pose")]
    ns = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{pkg}/planner.py", "exec"), ns)
    return ns, {n.name: ast.get_source_segment(src, n) for n in nodes}


def load(uuid, chunk):
    z = zipfile.ZipFile(RAW0 / f"labels/egomotion.offline/egomotion.offline.chunk_{chunk:04d}.zip")
    return pd.read_parquet(io.BytesIO(z.read(f"{uuid}.egomotion.offline.parquet")))


def main():
    uuid, chunk = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("00040136-e651-4abd-991d-0655ccda9430", 712)
    df = load(uuid, chunk)
    ts, T = H.poses_from_egomotion(df)

    # T2
    import inspect
    mine = {n: inspect.getsource(getattr(H, n)) for n in ("_yaw_of", "_relative_pose")}
    rng = np.random.default_rng(1)
    t2 = {}
    for pkg in ("drivejepa014", "drivejepa018bp06"):
        ns, src = planner_funcs(pkg)
        same_text = all(src[n].strip() == mine[n].strip() for n in src)
        worst = 0.0
        for _ in range(200):
            A, B = np.eye(4), np.eye(4)
            A[:3, :3], B[:3, :3] = Rotation.random(2, random_state=int(rng.integers(1e9))).as_matrix()
            A[:3, 3], B[:3, 3] = rng.normal(0, 50, 3), rng.normal(0, 50, 3)
            worst = max(worst, float(np.abs(ns["_relative_pose"](A, B) - H._relative_pose(A, B)).max()))
        t2[pkg] = dict(source_identical=same_text, max_abs_diff=worst)
    res["T2"] = dict(t2, ok=all(v["source_identical"] and v["max_abs_diff"] == 0.0 for v in t2.values()))

    # T3
    Ego = converter_ego_class()
    ego = Ego(df)
    P = H._Poses(ts, T)
    times = np.linspace(ts[0], ts[-1], 997).astype(np.int64)
    d = max(float(np.abs(ego.T(t) - P.at(t)).max()) for t in times)
    res["T3"] = dict(times=len(times), max_abs_diff_4x4=d, ok=d < 1e-9)

    # T1 + T4
    anchors = [t for t in range(H.STEP_US, int(ts[-1]) + 1, H.STEP_US)]
    worst_leak, bad_shape, bad_finite, bad_invalid = 0.0, 0, 0, 0
    for t in anchors:
        ref = H.converter_reference(ts, T, t)
        full = H.build_history(ts, T, t, T_ref=ref)
        keep = ts <= t
        trunc = H.build_history(ts[keep], T[keep], t, T_ref=ref)
        Tp = T.copy(); Tp[~keep] = np.nan
        poison = H.build_history(ts, Tp, t, T_ref=ref)
        drive_full = H.build_history(ts, T, t)
        drive_poison = H.build_history(ts, Tp, t)
        worst_leak = max(worst_leak, float(np.abs(full - trunc).max()), float(np.abs(full - poison).max()),
                         float(np.abs(drive_full - drive_poison).max()))
        bad_shape += full.shape != (8, 6) or full.dtype != np.float32
        bad_finite += not np.isfinite(full).all() or not np.isfinite(poison).all()
        for i in range(8):
            k = 8 - i
            expect_valid = (t - (k + 1) * H.STEP_US) >= ts[0]
            if expect_valid != bool(full[i, 5]) or (not expect_valid and np.any(full[i] != 0)):
                bad_invalid += 1
    res["T1"] = dict(anchors=len(anchors), max_abs_diff_all_vs_truncated_vs_nan_future=worst_leak, ok=worst_leak == 0.0)
    res["T4"] = dict(bad_shape=int(bad_shape), non_finite=int(bad_finite), bad_valid_rows=int(bad_invalid),
                     ok=bad_shape == 0 and bad_finite == 0 and bad_invalid == 0)
    res["clip"] = uuid
    res["all_ok"] = all(res[k]["ok"] for k in ("T1", "T2", "T3", "T4"))
    print(json.dumps(res, indent=1))
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
