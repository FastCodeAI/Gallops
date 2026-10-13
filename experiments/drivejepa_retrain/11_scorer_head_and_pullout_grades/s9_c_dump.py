#!/usr/bin/env python
"""S9 part C: pass-9 menus and MAP-FREE partial grades on the raw pull-out clips (training manifest type B_raw; 283 clips, 9,053 examples).
CPU only. Our own wrapper; nothing vendored, and nothing under s3_*/06_first_training_run/08_scorer_diagnosis, is edited. Method of s6_dump.py:
  model   the frozen encoder + the pass-9 trunk (generator + scorer), split forward checked once against the model's own forward;
  scene   the car from raw egomotion on S0's 10 Hz grid (r3_core.build_ego), the logged boxes moved from the rig frame to the clip frame
          and built into tracks (r3_raw_actors.obstacles_world -> r3_core.build_tracks), the real vehicle box (D2, calibration/vehicle_dimensions),
          car state and 41 frames of detections exactly as the S2 grader builds them (s2_grade.ego_state_at, s2_grade.detections); NO MAP;
  checks  the vendored calculator's own methods in its own order (train_pdm_scorer.py:185-207): _reset, then _calculate_no_at_fault_collision,
          _calculate_ttc, _calculate_history_comfort. _calculate_ego_area (the only step that reads map polygons, :333-402) is not called.
          The three map facts those methods read are set by hand, two settings (PREDICTIONS.md):
            strict  : "in more than one lane" true at every step  -> a side contact is the car's fault (:435-462), time buffer counts every
                      actor that is not behind (:665-673)
            lenient : none of the three                           -> side contacts with moving actors never at fault, time buffer counts
                      only actors ahead; "in an intersection" (:668-670) answered False by a stand-in object
Per example this stores: menu_9 (32, 8, 3), slot_9 (32, 256), logit_9 (32, 6), nc_strict / ttc_strict / nc_lenient / ttc_lenient / hc (32,),
the same five numbers for the human path (the calculator's reference row), the stored human path and the one rebuilt from egomotion
(alignment check), car speed, number of boxes at the anchor.
Usage (run_s9_dump.sh): s9_c_dump.py [--clips N --tag _timing] [--threads T] [--pool K] [--loaders L]"""
import os, sys, io, time, json, zipfile, pathlib, argparse
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA_VISIBLE_DEVICES=\"\""
HERE = pathlib.Path(__file__).resolve().parent; RETRAIN = HERE.parent; sys.path.insert(0, str(RETRAIN / "05_small_overfit_check/train_patch"))
import s35_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--clips", type=int, default=0); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--pool", type=int, default=2); ap.add_argument("--loaders", type=int, default=2)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--shard", type=int, default=256); ap.add_argument("--tag", default="")
a = ap.parse_args(); MAIN = 9; SEED = 20261008
NAS = pathlib.Path("{runs}/drivejepa_retrain/11_scorer_head_and_pullout_grades/pullout"); OUT = NAS / ("dump" + a.tag); LOG = HERE / "state_log.csv"
C.setup_env()
import numpy as np, pandas as pd, torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
assert not torch.cuda.is_available(), "S9 is CPU only"
torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
import s3_common as C3
from dataset_s35 import CacheOnlyDataset

def meminfo(k):
    for l in open("/proc/meminfo"):
        if l.startswith(k + ":"): return int(l.split()[1]) / 1048576
def state(): l1, l5, l15 = open("/proc/loadavg").read().split()[:3]; return float(l1), float(l5), float(l15), meminfo("SwapFree"), meminfo("MemAvailable")
_last = [0.0]; PAUSED = dict(load_s=0.0, mem_s=0.0, n=0)
def log_state(note, force=False):
    if force or time.time() - _last[0] >= 600:
        _last[0] = time.time(); l1, l5, l15, sf, ma = state(); dk = os.statvfs("/"); nk = os.statvfs("{nas}")
        with open(LOG, "a") as f: f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S+05:30')},{l1},{l5},{l15},{sf:.1f},{ma:.1f},{dk.f_bavail * dk.f_frsize / 2**30:.0f},{nk.f_bavail * nk.f_frsize / 2**40:.2f},{note}\n")
def my_pss_gb():
    par = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                for l in open(f"/proc/{d}/status"):
                    if l.startswith("PPid:"): par[int(d)] = int(l.split()[1]); break
            except OSError: pass
    mine = {os.getpid()}; grew = True
    while grew:
        grew = False
        for c, pp in par.items():
            if pp in mine and c not in mine: mine.add(c); grew = True
    pss = 0.0
    for q in mine:
        try:
            for l in open(f"/proc/{q}/smaps_rollup"):
                if l.startswith("Pss:"): pss += int(l.split()[1]) / 1048576; break
        except OSError: pass
    return pss
PEAK = [0.0]
def gate():
    """Same limits as S6: pause at 5-min load >= 32 or available RAM < 20 GB, resume at load < 32 and RAM > 40 GB; exit above 12 GB own memory."""
    l1, l5, l15, sf, ma = state(); r = my_pss_gb(); PEAK[0] = max(PEAK[0], r)
    if r > 12.0: log_state(f"s9_c_dump ABORT own memory {r:.1f} GB", True); raise SystemExit(f"own memory {r:.1f} GB exceeds the 12 GB cap")
    if l5 >= 32.0 or ma < 20.0:
        t = time.time(); PAUSED["n"] += 1; log_state(f"s9_c_dump PAUSED load {l5} avail {ma:.1f}", True)
        while True:
            time.sleep(120); l1, l5, l15, sf, ma = state()
            if l5 < 32.0 and ma > 40.0: break
        PAUSED["load_s"] += time.time() - t; log_state("s9_c_dump resumed", True)
    log_state(f"s9_c_dump{a.tag} running own {r:.1f} GB")

# ---------------------------------------------------------------- map-free grading worker
_SCENES = {}
class NoMap:
    """Stand-in for the drivable-area map in the lenient setting: the one question the time-buffer code asks it (train_pdm_scorer.py:668-670)."""
    def is_in_layer(self, point, layer): return False
def load_raw_scene(u, ch):
    """car + boxes of one raw clip in the clip frame (origin and clock of the first egomotion sample). Functions copied from S0:
    r3_raw_scan.ego_from_bytes (:5-7), r3_raw_actors.raw_ego / obstacles_world (:7-18); read-only on the NAS label files."""
    if u in _SCENES: return _SCENES[u]
    import pyarrow.parquet as pq, r3_core as K, s3_scorer as SS
    R = C3.S0C.RAW_NAS; z = zipfile.ZipFile(R / f"labels/egomotion.offline/egomotion.offline.chunk_{ch:04d}.zip"); d = pq.read_table(io.BytesIO(z.read(f"{u}.egomotion.offline.parquet"))).to_pandas().sort_values("timestamp")
    ts = d["timestamp"].to_numpy() / 1e6; xy = d[["x", "y"]].to_numpy(); yaw = np.unwrap(K.yaw_q(d.qx.to_numpy(), d.qy.to_numpy(), d.qz.to_numpy(), d.qw.to_numpy())); S = K.build_ego(ts, xy, yaw); S["uuid"] = u
    o = pq.read_table(R / f"labels/obstacle.offline.members/chunk_{ch:04d}/{u}.obstacle.offline.parquet").to_pandas(); assert (o.reference_frame == "rig").all()
    t = o.reference_frame_timestamp_us.to_numpy().astype(np.float64) / 1e6; ex, ey, eyaw = np.interp(t, ts, xy[:, 0]), np.interp(t, ts, xy[:, 1]), np.interp(t, ts, yaw); c, s = np.cos(eyaw), np.sin(eyaw); x, y = o.center_x.to_numpy(), o.center_y.to_numpy()
    ob = pd.DataFrame({"tid": o.track_id.astype(str), "ts": o.timestamp_us.to_numpy().astype(np.float64) / 1e6 - ts[0], "x": ex + c * x - s * y, "y": ey + s * x + c * y,
                       "yaw": K.yaw_q(o.orientation_x.to_numpy(), o.orientation_y.to_numpy(), o.orientation_z.to_numpy(), o.orientation_w.to_numpy()) + eyaw, "L": o.size_x.to_numpy(), "W": o.size_y.to_numpy(), "cat": o.label_class})
    S = K.build_tracks(S, ob); vp_src = "vehicle_dimensions"
    try:
        vd = pd.read_parquet(R / f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{ch:04d}.parquet").loc[u]; S["vp"] = SS.vehicle_params(vd["length"], vd["width"], vd["rear_axle_to_bbox_center"], vd["wheelbase"], vd["height"])
    except Exception: S["vp"] = None; vp_src = "pacifica_fallback"
    S["vp_src"] = vp_src
    if len(_SCENES) >= 3: _SCENES.pop(next(iter(_SCENES)))
    _SCENES[u] = S; return S
def grade(task):
    """-> dict of arrays for one example; index 0 of the calculator's batch is the reference row (here: the human path)."""
    import types, s2_grade as G, s3_scorer as SS
    from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as CNS
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex, EgoAreaIndex
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    S = load_raw_scene(task["clip"], task["chunk"]); K = G.K; n = len(S["t"]); i = int(round(task["t_us"] / 1e6 / K.DT)); assert 0 <= i < n, (i, n)
    vp = S["vp"]; ego = G.ego_state_at(S, i, vp); det = G.detections(S, i)
    obs = PDMObservation(CNS.proposal_sampling, CNS.proposal_sampling, 100, observation_sample_res=1, extend_observation_for_ttc=False); obs.update_detections_tracks(det, [[] for _ in det], {}, compute_traffic_light_data=True)   # as s2_grade.build_cache
    def ego_k(k):
        """car state k frames after the anchor; past the end of the clip the last state is held, with the clock still advancing"""
        j = i + k
        if j <= n - 1: return G.ego_state_at(S, j, vp)
        e = G.ego_state_at(S, n - 1, vp); from nuplan.common.actor_state.ego_state import EgoState; from nuplan.common.actor_state.state_representation import TimePoint
        return EgoState.build_from_rear_axle(rear_axle_pose=e.rear_axle, rear_axle_velocity_2d=e.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=e.dynamic_car_state.rear_axle_acceleration_2d, tire_steering_angle=0.0,
                                             time_point=TimePoint(ego.time_point.time_us + 100_000 * k), vehicle_parameters=e.car_footprint.vehicle_parameters)
    ref = InterpolatedTrajectory([ego_k(k) for k in range(41)])                                        # reference row = the human path (as s2_grade.build_cache)
    j0 = max(0, i - 15); past = InterpolatedTrajectory([G.ego_state_at(S, j, vp) for j in range(j0, i + 1, 5)]) if i - j0 >= 5 else None                                                  # 1.5 s at 2 Hz, as s2_grade.build_cache
    rel = np.array([np.r_[K.ego_frame(S["xy"][min(i + 5 * k, n - 1)], S["xy"][i], S["yaw"][i]), S["yaw"][min(i + 5 * k, n - 1)] - S["yaw"][i]] for k in range(1, 9)])
    sc = SS.scorer_for(ego.car_footprint.vehicle_parameters); sim = CNS.before_score_proposals(types.SimpleNamespace(ego_state=ego, trajectory=ref), np.asarray(task["menu"], dtype=np.float32), ego); out = {}
    for name in ("strict", "lenient"):
        sc._reset(sim, obs, None, [], NoMap(), past)
        if name == "strict": sc._ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES] = True
        sc._calculate_no_at_fault_collision(); sc._calculate_ttc(); nc = sc._multi_metrics[MultiMetricIndex.NO_COLLISION].copy(); tt = sc._weighted_metrics[WeightedMetricIndex.TTC].copy()
        out[f"nc_{name}"] = nc[1:].astype(np.float32); out[f"ttc_{name}"] = tt[1:].astype(np.float32); out[f"human_nc_{name}"] = np.float32(nc[0]); out[f"human_ttc_{name}"] = np.float32(tt[0])
        if name == "strict": out["nc_time_idx"] = np.where(np.isfinite(sc._collision_time_idcs[1:]), sc._collision_time_idcs[1:], -1).astype(np.int16)
    sc._calculate_history_comfort(); hc = sc._weighted_metrics[WeightedMetricIndex.HISTORY_COMFORT].astype(np.float64)
    out["hc"] = hc[1:].astype(np.float32); out["human_hc"] = np.float32(hc[0]); out["human_from_ego"] = rel.astype(np.float32); out["car_speed"] = np.float32(S["speed"][i]); out["boxes_at_anchor"] = np.int16(len(det[0].tracked_objects.tracked_objects))
    out["has_past"] = np.bool_(past is not None); out["future_frames"] = np.int16(min(n - 1 - i, 40)); out["real_vehicle_box"] = np.bool_(S["vp_src"] == "vehicle_dimensions"); out["max_sim_speed"] = np.hypot(sim[1:, :, 3], sim[1:, :, 4]).max(1).astype(np.float32)
    return out

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); man = pd.read_parquet(RETRAIN / "04_training_set/manifest.parquet"); man = man[man.type == "B_raw"].sort_values(["clip_uuid", "anchor_t"]).reset_index(drop=True); assert not man.has_label2.any() and (man.source == "raw").all()
    clips = pd.read_csv(RETRAIN / "04_training_set/clips.csv").set_index("uuid"); lab = pd.read_parquet(RETRAIN / "04_training_set/labels_index.parquet").set_index("sample_id"); man["chunk"] = clips.loc[man.clip_uuid, "chunk"].to_numpy(); man["t_us"] = lab.loc[man.sample_id, "t_us"].to_numpy()
    if a.clips:
        u = np.sort(man.clip_uuid.unique()); keep = set(u[np.sort(np.random.default_rng(SEED).choice(len(u), a.clips, replace=False))]); man = man[man.clip_uuid.isin(keep)].reset_index(drop=True)
    C3._assert_train(set(man.clip_uuid)); print(f"pull-out set: {len(man)} examples, {man.clip_uuid.nunique()} clips, {man.chunk.nunique()} chunks", flush=True)
    agent = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=0); agent.eval(); m = agent._pad_model
    ds = CacheOnlyDataset(cache_path=str(C.NAS / "04_training_set/s1/cache"), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True, has_label2=dict(zip(man.sample_id, man.has_label2)),
                          token_clips=dict(zip(man.sample_id, man.clip_uuid)), slim_root=None, split_guard=lambda u: C3._assert_train(list(u)))
    idx = {t: i for i, t in enumerate(ds.tokens)}; order = [idx[t] for t in man.sample_id]; info = man.set_index("sample_id")
    sd = torch.load(C.NAS / f"06_first_training_run/run/ckpt/pass_{MAIN:02d}.pt", map_location="cpu"); assert sd["passno"] == MAIN; w = {k[len("_pad_model."):]: v for k, v in sd["trainable"].items()}
    missing, unexpected = m.load_state_dict(w, strict=False); assert not unexpected and all(k.startswith("_backbone") for k in missing), (missing[:3], unexpected[:3])
    @torch.no_grad()
    def trunk(t, ego, image_feature):                                                               # s6_dump.py trunk(), unchanged
        bev = t.hist_encoding(ego)[:, None] + t.init_feature.weight[None]
        for refine in t._trajectory_head: bev, props = refine(bev, image_feature)
        B = len(props); slot = bev.reshape(B, props.shape[1], props.shape[2], -1).amax(-2); logit = t.scorer.pred_score.mlp[-1](t.scorer.pred_score.mlp[:-1](slot)); return props, slot, logit
    pool = ProcessPoolExecutor(max_workers=a.pool, mp_context=mp.get_context("spawn")); shards = [order[i:i + a.shard] for i in range(0, len(order), a.shard)]; t0 = time.time(); n_done = 0; tim = dict(encoder=0.0, trunk=0.0, grade=0.0, load=0.0); parity = None
    log_state(f"s9_c_dump{a.tag} start {len(man)} examples threads {a.threads} pool {a.pool}", True)
    for si, sh in enumerate(shards):
        fn = OUT / f"shard_{si:04d}.npz"
        if fn.exists(): continue
        gate(); R = {}; put = lambda k, v: R.setdefault(k, []).append(v); tl = time.time()
        dl = DataLoader(ds, batch_sampler=[sh[i:i + a.bs] for i in range(0, len(sh), a.bs)], num_workers=a.loaders, collate_fn=default_collate)
        for f, tg in dl:
            tim["load"] += time.time() - tl; gate(); B = len(tg["token"])
            with torch.no_grad():
                if parity is None: ref = agent.forward({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
                te = time.time(); f["lidar2img"] = f["lidar2img"][:, 1:2]; ego = f["ego_status"][:, -1]; cam = torch.cat([m.transform(f["camera_feature_2"])[:, None], m.transform(f["camera_feature_1"])[:, None]], dim=1)
                imf = m._backbone(cam, img_metas=f); tim["encoder"] += time.time() - te; tt = time.time(); props, slot, logit = trunk(m, ego, imf); tim["trunk"] += time.time() - tt
                if parity is None:
                    parity = dict(menu_max_abs=float((props - ref["proposals"]).abs().max()), logit_max_abs=float((logit - ref["pred_logit"]).abs().max()), same_pick=bool((torch.sigmoid(logit)[:, :, -1].argmax(1) == ref["pdm_score"].argmax(1)).all()))
                    print("parity of the split forward vs the model's own forward (pass 9):", parity, flush=True); assert parity["menu_max_abs"] < 1e-4 and parity["logit_max_abs"] < 1e-3 and parity["same_pick"], parity
            tgd = time.time(); toks = list(tg["token"]); tasks = [dict(clip=info.loc[t, "clip_uuid"], chunk=int(info.loc[t, "chunk"]), t_us=int(info.loc[t, "t_us"]), menu=props[i].float().numpy()) for i, t in enumerate(toks)]; G_ = list(pool.map(grade, tasks)); tim["grade"] += time.time() - tgd
            for i, t in enumerate(toks):
                put("token", t); put("clip", info.loc[t, "clip_uuid"]); put("anchor_t", float(info.loc[t, "anchor_t"])); put("human", tg["trajectory"][i].numpy().astype(np.float32)); put("ego_status", ego[i].numpy().astype(np.float32))
                put("menu_9", props[i].numpy().astype(np.float32)); put("slot_9", slot[i].numpy().astype(np.float16)); put("logit_9", logit[i].numpy().astype(np.float32))
                for k, v in G_[i].items(): put(k, v)
            n_done += B; tl = time.time()
        assert R["token"] == [ds.tokens[j] for j in sh], "shard order"
        tmp = OUT / f"shard_{si:04d}.tmp.npz"; np.savez(tmp, **{k: np.stack(v) for k, v in R.items()}); os.replace(tmp, fn); el = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] shard {si + 1}/{len(shards)}: {n_done} examples this launch, {el / max(n_done, 1):.2f} s/example wall | {json.dumps({k: round(v / max(n_done, 1), 3) for k, v in tim.items()})} | own memory peak {PEAK[0]:.1f} GB | paused {PAUSED}", flush=True)
    pool.shutdown(); el = time.time() - t0; ch = os.times(); cpu_s = ch.user + ch.system + ch.children_user + ch.children_system
    res = dict(tag=a.tag, examples_this_launch=n_done, clips=int(man.clip_uuid.nunique()), wall_s=el, cpu_s=cpu_s, cpu_h=cpu_s / 3600, s_per_example_wall=el / max(n_done, 1), cpu_s_per_example=cpu_s / max(n_done, 1), per_example_s={k: v / max(n_done, 1) for k, v in tim.items()},
               threads=a.threads, pool=a.pool, loaders=a.loaders, paused=PAUSED, peak_own_pss_gb=PEAK[0], parity=parity, torch=torch.__version__, cuda=torch.cuda.is_available())
    with open(HERE / "c_dump_launches.jsonl", "a") as fh: fh.write(json.dumps(res) + "\n")
    log_state(f"s9_c_dump{a.tag} end {n_done} examples", True); print("DONE", json.dumps(res), flush=True)
