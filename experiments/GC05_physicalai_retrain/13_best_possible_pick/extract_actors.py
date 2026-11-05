#!/usr/bin/env python
"""Step 0: the other road users' boxes the oracle grader reads, one file per scene, from an EXISTING run's rollout.asl (read-only).
Method of "14_after_retraining_diagnosis/part4_why_it_stays_stopped/t04_extract.py" (:26-33): actor boxes from rollout_metadata.actor_definitions, actor poses
(local frame, box centre) per simulator step from actor_poses. The simulator replays these from the scene's log, so they are the same
in every run of the scene; `--check RUN_DIR` compares a second run's log against the stored file and prints the largest difference.

  alpasim/.venv/bin/python extract_actors.py <short8> <run_dir>            -> NAS actors/<short8>.npz (+ .json with its sha256)
  alpasim/.venv/bin/python extract_actors.py <short8> --check <run_dir>    -> largest pose difference vs the stored file
"""
from __future__ import annotations
import asyncio, glob, hashlib, json, math, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ALP = HERE.parents[2]; OUT = pathlib.Path("{runs}/drivejepa_retrain/13_best_possible_pick/actors")
sys.path[:0] = [str(ALP / "alpasim/src/utils/src"), str(ALP / "alpasim/src/grpc")]
from alpasim_utils.logs import async_read_pb_log  # noqa: E402


def yaw_q(q): return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
def se2(pose): return np.array([pose.vec.x, pose.vec.y, yaw_q(pose.quat)])


async def _read(asl):
    o = dict(defs={}, ap_t=[], ap=[], ego=[])
    async for e in async_read_pb_log(asl):
        k = e.WhichOneof("log_entry")
        if k == "rollout_metadata":
            m = e.rollout_metadata; o["render_start_us"] = int(m.session_metadata.render_start_timestamp_us); o["force_gt_us"] = int(m.force_gt_duration)
            o["rig_to_aabb_x"] = float(m.transform_ego_coords_rig_to_aabb.vec.x); o["egodriver"] = m.version_ids.egodriver_version.version_id
            o["defs"] = {a.actor_id: (float(a.aabb.size_x), float(a.aabb.size_y), a.actor_label) for a in m.actor_definitions.actor_aabb}
        elif k == "actor_poses":
            ap = e.actor_poses; o["ap_t"].append(int(ap.timestamp_us)); o["ap"].append({p.actor_id: se2(p.actor_pose) for p in ap.actor_poses})
        elif k == "controller_request":
            r = e.controller_request; st = r.state.state; o["ego"].append([int(r.state.timestamp_us), *se2(r.state.pose), st.linear_velocity.x, st.linear_velocity.y])
    return o


def load(run):
    (asl,) = glob.glob(str(pathlib.Path(run) / "rollouts" / "clipgt-*" / "*" / "rollout.asl")); o = asyncio.run(_read(asl)); o["asl"] = asl
    ids = sorted(a for a in o["defs"] if a != "EGO"); n = len(o["ap_t"]); A = np.full((n, len(ids), 3), np.nan)
    for i, d in enumerate(o["ap"]):
        for j, a in enumerate(ids):
            if a in d: A[i, j] = d[a]
    o["ids"] = ids; o["A"] = A; return o


def main():
    short = sys.argv[1]
    if sys.argv[2] == "--check":
        o = load(sys.argv[3]); z = np.load(OUT / f"{short}.npz"); same_ids = list(z["ids"]) == o["ids"]; same_t = np.array_equal(z["ap_t"], np.array(o["ap_t"]))
        both = np.isfinite(z["A"]) & np.isfinite(o["A"]) if same_ids and same_t else None
        res = dict(scene=short, checked_run=sys.argv[3], egodriver=o["egodriver"], same_actor_ids=same_ids, same_timestamps=same_t,
                   same_presence=bool(np.array_equal(np.isfinite(z["A"]), np.isfinite(o["A"]))) if both is not None else False,
                   max_abs_pose_diff=float(np.abs(z["A"][both] - o["A"][both]).max()) if both is not None else None, poses_compared=int(both.sum() // 3) if both is not None else 0,
                   same_ego_box=bool(np.allclose(z["ego_dims"], o["defs"]["EGO"][:2]) and abs(float(z["rig_to_aabb_x"]) - o["rig_to_aabb_x"]) < 1e-9))
        print(json.dumps(res)); return
    run = sys.argv[2]; o = load(run); OUT.mkdir(parents=True, exist_ok=True); f = OUT / f"{short}.npz"
    np.savez(f, ap_t=np.array(o["ap_t"], dtype=np.int64), ids=np.array(o["ids"]), A=o["A"], dims=np.array([o["defs"][a][:2] for a in o["ids"]], dtype=np.float64), labels=np.array([o["defs"][a][2] for a in o["ids"]]),
             ego_dims=np.array(o["defs"]["EGO"][:2], dtype=np.float64), rig_to_aabb_x=np.float64(o["rig_to_aabb_x"]), render_start_us=np.int64(o["render_start_us"]), force_gt_us=np.int64(o["force_gt_us"]), ego=np.array(o["ego"], dtype=np.float64))
    meta = dict(scene=short, source_run=str(run), source_asl=o["asl"], source_egodriver=o["egodriver"], actors=len(o["ids"]), steps=len(o["ap_t"]), step_us=sorted(set(np.diff(o["ap_t"]).tolist()))[:4],
                t_first_rel_handover_s=(o["ap_t"][0] - o["render_start_us"] - o["force_gt_us"]) / 1e6, t_last_rel_handover_s=(o["ap_t"][-1] - o["render_start_us"] - o["force_gt_us"]) / 1e6,
                ego_box_m=list(o["defs"]["EGO"][:2]), rig_to_aabb_x=o["rig_to_aabb_x"], labels={l: int(c) for l, c in zip(*np.unique([o["defs"][a][2] for a in o["ids"]], return_counts=True))}, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    json.dump(meta, open(OUT / f"{short}.json", "w"), indent=1); print(json.dumps(meta))


if __name__ == "__main__":
    main()
