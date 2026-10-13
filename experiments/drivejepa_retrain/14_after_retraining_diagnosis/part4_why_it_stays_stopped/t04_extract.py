#!/usr/bin/env python
"""T04 extraction: one pass over each rollout.asl -> cache/<ver>_<scene>.pkl with the actor boxes (rollout_metadata.actor_definitions),
the actor poses per simulator step (actor_poses, local->aabb centre), the ego state per controller tick, what the driver was sent
(driver_ego_trajectory newest pose + dynamic state, camera frame timestamps + md5), and the front frames at handover and +5 s.

  CUDA_VISIBLE_DEVICES="" {runs}/alpasim_bulk/nurec-venv/bin/python t04_extract.py
"""
import asyncio, glob, hashlib, multiprocessing as mp, pathlib, pickle, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part2_why_it_swerves"))
import t02_common as C  # noqa: E402
from alpasim_utils.logs import async_read_pb_log  # noqa: E402

STUCK = ["d2715023", "b44a4a44"]; MOVE = ["5f064b02", "e3ed554f"]; FOUR = STUCK + MOVE


def scenes(): return sorted(p.name[5:] for p in (C.BASE / "analysis_3.0/Retrain-data/07_closed_loop_test/menus").glob("d016_*") if p.is_dir())


async def _read(asl, keep_frames, keep_reqs):
    o = dict(actors={}, ap_t=[], ap=[], ego=[], drv=[], frames=[], img={}, reqs={}, dbg={}, kinds={}); rs = None; last_now = None
    async for e in async_read_pb_log(asl):
        k = e.WhichOneof("log_entry"); o["kinds"][k] = o["kinds"].get(k, 0) + 1
        if k == "rollout_metadata":
            m = e.rollout_metadata; rs = int(m.session_metadata.render_start_timestamp_us); o["render_start_us"] = rs; o["force_gt_us"] = int(m.force_gt_duration)
            o["rig_to_aabb_x"] = float(m.transform_ego_coords_rig_to_aabb.vec.x); o["egodriver"] = m.version_ids.egodriver_version.version_id
            o["gt"] = np.array([[int(p.timestamp_us), *C.se2(p.pose)] for p in m.ego_rig_recorded_ground_truth_trajectory.poses], dtype=np.float64)
            o["actors"] = {a.actor_id: (float(a.aabb.size_x), float(a.aabb.size_y), a.actor_label) for a in m.actor_definitions.actor_aabb}
        elif k == "actor_poses":
            ap = e.actor_poses; o["ap_t"].append(int(ap.timestamp_us)); o["ap"].append({p.actor_id: C.se2(p.actor_pose).astype(np.float32) for p in ap.actor_poses})
        elif k == "controller_request":
            r = e.controller_request; st = r.state.state; o["ego"].append([int(r.state.timestamp_us), *C.se2(r.state.pose), st.linear_velocity.x, st.linear_velocity.y, len(r.planned_trajectory_in_rig.poses)])
            if keep_reqs: o["reqs"][int(r.state.timestamp_us)] = r.SerializeToString()
        elif k == "driver_ego_trajectory":
            t = e.driver_ego_trajectory; p = t.trajectory.poses[-1]; s = t.dynamic_states[-1]
            o["drv"].append([int(p.timestamp_us), *C.se2(p.pose), s.linear_velocity.x, s.linear_velocity.y, s.linear_acceleration.x, s.linear_acceleration.y])
        elif k == "driver_request": last_now = int(e.driver_request.time_now_us)
        elif k == "driver_return":
            try: d = pickle.loads(e.driver_return.debug_info.unstructured_debug_info); o["dbg"][last_now] = {x: d.get(x) for x in ("replan", "chosen_index", "chosen_score", "source_t0_us", "cache_age_us", "pick_without_momentum", "momentum_code")}
            except Exception: pass
        elif k == "driver_camera_image":
            ci = e.driver_camera_image.camera_image
            if "front_wide" not in ci.logical_id: continue
            o["frames"].append((int(ci.frame_start_us), int(ci.frame_end_us), hashlib.md5(ci.image_bytes).hexdigest(), len(ci.image_bytes)))
            if keep_frames and rs is not None:
                for rel in (0.0, 5.0):
                    if abs((int(ci.frame_end_us) - rs - o["force_gt_us"]) / 1e6 - rel) < 0.026: o["img"][rel] = (int(ci.frame_start_us), int(ci.frame_end_us), bytes(ci.image_bytes))
    o["ego"] = np.array(o["ego"], dtype=np.float64); o["drv"] = np.array(o["drv"], dtype=np.float64); o["ap_t"] = np.array(o["ap_t"]); o["ho"] = o["render_start_us"] + o["force_gt_us"]
    return o


def one(job):
    ver, s = job; f = HERE / "cache" / f"{ver}_{s}.pkl"
    if f.is_file(): return f"{ver} {s} cached"
    t0 = time.time(); run = C.run_dir(ver, s); (asl,) = glob.glob(str(run / "rollouts" / f"clipgt-{s}*" / "*" / "rollout.asl"))
    o = asyncio.run(_read(asl, s in STUCK, s in STUCK and ver == "0.1.6")); o["run"] = str(run); o["ver"] = ver; o["scene"] = s
    pickle.dump(o, open(f, "wb")); return f"{ver} {s} {len(o['ap_t'])} actor steps {len(o['actors'])} actors {time.time() - t0:.0f}s kinds={o['kinds']}"


if __name__ == "__main__":
    jobs = [("0.1.6", s) for s in scenes()] + [("0.1.4b", s) for s in FOUR]
    with mp.Pool(12) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1): print(r, flush=True)
