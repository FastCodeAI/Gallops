#!/usr/bin/env python
"""S1 label-1 converter: raw PhysicalAI clip -> Drive-JEPA training samples (pad_feature.gz / pad_target.gz) + shipped-model menus.

Run:  cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python analysis_3.0/Retrain-data/02_build_samples/s1_convert.py
Read-only on adapters/, drafts/, runs/, alpasim/, navsim_v2 (imported, never edited). Writes only under 02_build_samples/ and the NAS.
The geometry (rotation-only f-theta -> declared opencv pinhole) follows Re-ranker/s2b_parity/parity.py:74-96 and the per-clip
repointing of Re-ranker/s2c_practice/gen_menus.py:126-131; the Ego class follows gen_menus.py:68-87. Both are re-stated here
because importing those modules opens log files under Re-ranker/ (parity.py:36, gen_menus.py:37).
"""
from __future__ import annotations
import argparse, gzip, hashlib, io, json, os, pathlib, pickle, sys, time, zipfile
import cv2, numpy as np, pandas as pd, yaml
from types import SimpleNamespace
from scipy.spatial.transform import Rotation, Slerp
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import s1_common as C  # noqa: E402
sys.path.insert(0, "{runs}/paddock_sid/analysis/drivejepa/mode1")
import mode1_common as M1  # noqa: E402  (offline harness: draft-first sys.path -> drafts/drivejepa-0.1.4 -> adapters/drivejepa014)
import step2_r1_proposals as R1  # noqa: E402  (proposal_metrics, step2_r1_proposals.py:127-135)
from drivejepa import constants as K  # noqa: E402
from drivejepa import planner as P  # noqa: E402
from alpasim_runtime.route_generator import RouteGenerator, RouteGeneratorRecorded  # noqa: E402
from alpasim_utils.geometry import Pose  # noqa: E402

STEP_US, HORIZON_US, HIST = 500_000, 4_000_000, 4
#: S2.1 D2: rows blacked out at the bottom of both 512x256 model images (camera_feature_1/2), where the raw camera sees the ego
#: bonnet and AlpaSim renders do not. N = highest measured bonnet edge over 29 pilot clips (35 rows) + 2 (03_grade_candidates/s21_bonnet.csv).
#: Default ON; --bonnet-rows 0 reproduces the S1 samples. The BEV `camera_feature` is never read by forward and is left untouched.
BONNET_MASK_ROWS = 37
ALP = pathlib.Path("{runs}/paddock_sid/Alpasim")
CACHE = C.NAS / "cache"; MENUS = C.NAS / "menus"; RECT = C.NAS / "rectified"

# ---- declared camera: the one AlpaSim renders (adapters/alpasim_configs/driver/drivejepa.yaml:49-63, extra_cameras[0]) ----
def declared_camera():
    y = yaml.safe_load(open(ALP / "adapters/alpasim_configs/driver/drivejepa.yaml")); found = []
    def walk(o):
        if isinstance(o, dict):
            if "extra_cameras" in o: found.append(o["extra_cameras"])
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(y); cam = found[0][0]; assert cam["logical_id"] == C.FRONT and cam["intrinsics"]["model"] == "opencv_pinhole"
    op = cam["intrinsics"]["opencv_pinhole"]; h, w = cam["resolution_hw"]
    return dict(width=w, height=h, fx=op["focal_length"][0], fy=op["focal_length"][1], cx=op["principal_point"][0], cy=op["principal_point"][1],
                radial=op["radial"], tangential=op["tangential"], t_rig=cam["rig_to_camera"]["translation_m"], q_xyzw=cam["rig_to_camera"]["rotation_xyzw"])
VIRT = declared_camera()
K_V = np.array([[VIRT["fx"], 0, VIRT["cx"]], [0, VIRT["fy"], VIRT["cy"]], [0, 0, 1]], dtype=np.float64)
DIST_V = np.array([VIRT["radial"][0], VIRT["radial"][1], VIRT["tangential"][0], VIRT["tangential"][1], VIRT["radial"][2]], dtype=np.float64)
R_V = Rotation.from_quat(VIRT["q_xyzw"]).as_matrix(); t_V = np.array(VIRT["t_rig"], dtype=np.float64)   # camera pose in the rig

def build_maps(ft, R_N):
    """For every declared-pinhole pixel (with its opencv distortion), the native f-theta pixel seeing the same direction (rotation only)."""
    u, v = np.meshgrid(np.arange(VIRT["width"], dtype=np.float64), np.arange(VIRT["height"], dtype=np.float64))
    n = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1)[:, None, :], K_V, DIST_V).reshape(-1, 2)
    d = (R_N.T @ (R_V @ np.concatenate([n, np.ones((len(n), 1))], 1).T)).T; d /= np.linalg.norm(d, axis=1, keepdims=True)
    theta = np.arccos(np.clip(d[:, 2], -1, 1)); r = np.polyval(np.array(ft["angle_to_pixeldist_poly"])[::-1], theta)
    rho = np.hypot(d[:, 0], d[:, 1]); rho[rho == 0] = 1.0; x = r * d[:, 0] / rho; y = r * d[:, 1] / rho
    c_, d_, e_ = ft["linear_cde"]; xs = c_ * x + d_ * y + ft["principal_point"][0]; ys = e_ * x + y + ft["principal_point"][1]
    bad = (theta > ft["max_angle"]) | (d[:, 2] <= 0); xs[bad] = -1; ys[bad] = -1
    sh = (VIRT["height"], VIRT["width"]); return xs.reshape(sh).astype(np.float32), ys.reshape(sh).astype(np.float32), float(bad.mean())

def rectify(frame_bgr, maps):
    return cv2.remap(frame_bgr, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

class Ego:
    """egomotion.offline -> rig pose T(t) (clip frame <- rig) and backward-difference dynamics in the rig frame (gen_menus.py:68-87)."""
    def __init__(self, df):
        df = df.sort_values("timestamp"); self.ts = df.timestamp.values.astype(np.int64); self.p = df[["x", "y", "z"]].values.astype(np.float64)
        self.R = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].values); self.slerp = Slerp(self.ts.astype(np.float64), self.R)
        dp = np.diff(self.p, axis=0); dt = np.diff(self.ts) / 1e6; v_local = np.vstack([dp[:1] / dt[:1, None], dp / dt[:, None]])
        self.v_rig = np.einsum("nij,nj->ni", self.R.inv().as_matrix(), v_local)
        dv = np.diff(self.v_rig, axis=0); self.a_rig = np.vstack([dv[:1] / dt[:1, None], dv / dt[:, None]])
    def T(self, t):
        t = float(np.clip(t, self.ts[0], self.ts[-1])); M = np.eye(4); M[:3, :3] = self.slerp([t]).as_matrix()[0]
        M[:3, 3] = [np.interp(t, self.ts, self.p[:, i]) for i in range(3)]; return M
    def dyn(self, t):
        i = max(int(np.searchsorted(self.ts, t, side="right")) - 1, 1)
        return float(self.v_rig[i, 0]), float(self.v_rig[i, 1]), float(self.a_rig[i, 0]), float(self.a_rig[i, 1]), int(t - self.ts[i])

def route_obj(route_rig):
    return SimpleNamespace(waypoints=[SimpleNamespace(x=float(w[0]), y=float(w[1]), z=0.0) for w in route_rig])

def load_clip(uuid, chunk):
    d = C.NAS / "clips" / uuid
    ts = pd.read_parquet(d / f"{uuid}.{C.FRONT}.timestamps.parquet").timestamp.values.astype(np.int64)
    ze = zipfile.ZipFile(C.RAW0 / f"labels/egomotion.offline/egomotion.offline.chunk_{chunk:04d}.zip")
    ego = Ego(pd.read_parquet(io.BytesIO(ze.read(f"{uuid}.egomotion.offline.parquet"))))
    ir = pd.read_parquet(C.NAS / f"calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_{chunk:04d}.parquet").loc[(uuid, C.FRONT)]
    er = pd.read_parquet(C.NAS / f"calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_{chunk:04d}.parquet").loc[(uuid, C.FRONT)]
    assert ir.model_type == "ftheta", ir.model_type
    ft = json.loads(ir.model_parameters); R_N = Rotation.from_quat([er.qx, er.qy, er.qz, er.qw]).as_matrix(); t_N = np.array([er.x, er.y, er.z])
    return d, ts, ego, ft, R_N, t_N

def decode(mp4, need):
    cap = cv2.VideoCapture(str(mp4)); frames = {}; i = 0; want = set(need)
    while want and i <= max(need):
        ok, fr = cap.read()
        if not ok: break
        if i in want: frames[i] = fr; want.discard(i)
        i += 1
    cap.release(); assert not want, f"frames missing: {sorted(want)[:5]}"
    return frames

def anchors_for(ego):
    return [t for t in range(STEP_US, int(ego.ts[-1]) + 1, STEP_US) if t + HORIZON_US <= ego.ts[-1] and t - STEP_US >= ego.ts[0]]

def sample(ego, t, rg, older_rgb, newer_rgb):
    """One training sample. Features composed exactly as planner.build_features (planner.py:624-678) with the adapter's own functions."""
    Tn = ego.T(t)
    # Route stand-in = the human's own future path (every egomotion sample after t, rig frame at t). KNOWN DIFFERENCE vs inference,
    # where the route comes from AlpaSim's RouteGenerator; that generator is run too, for comparison only (it raises on some clips).
    fut = ego.ts > t; Tinv = np.linalg.inv(Tn); route = (Tinv[:3, :3] @ ego.p[fut].T).T + Tinv[:3, 3]
    command = P._command_one_hot(route_obj(route), min_lookahead_m=K.MIN_LOOKAHEAD_M, lateral_m=K.COMMAND_LATERAL_M)   # planner.py:401-424
    try:
        if rg is None: raise RuntimeError("no generator")
        r2 = RouteGenerator.prepare_for_policy(rg.generate_route(t, Pose.from_se3(Tn.astype(np.float32)))).waypoints.astype(np.float64)
        cmd_rg = "LSRU"[int(np.argmax(P._command_one_hot(route_obj(r2), min_lookahead_m=K.MIN_LOOKAHEAD_M, lateral_m=K.COMMAND_LATERAL_M)))]
    except Exception as e:  # noqa: BLE001
        cmd_rg = "ERR:" + type(e).__name__
    rows = []; lag = 0
    for j in range(HIST):
        tj = t - (HIST - 1 - j) * STEP_US
        if tj < ego.ts[0]: continue                                     # rows before the clip start do not exist; forward reads only [-1]
        vx, vy, ax, ay, lg = ego.dyn(tj); lag = max(lag, lg)
        rows.append(np.concatenate([P._relative_pose(ego.T(tj), Tn), [vx, vy], [ax, ay], command]))   # planner.py:749-787
    ego_status = np.array(rows, dtype=np.float32)
    cf1 = P._pipeline_a(newer_rgb); cf2 = P._pipeline_a(older_rgb)        # camera_feature_1 = NEWEST, _2 = older (planner.py:637-639)
    if BONNET_MASK_ROWS > 0:
        cf1 = cf1.copy(); cf2 = cf2.copy(); cf1[:, -BONNET_MASK_ROWS:, :] = 0.0; cf2[:, -BONNET_MASK_ROWS:, :] = 0.0
    h, w = newer_rgb.shape[:2]; scale = P._scale_matrix(K.BEV_SCALE); bev, shp, l2i = [], [], []
    for im in [P._dummy_image(h, w), newer_rgb]:
        tsr, s_ = P._pipeline_b(im); bev.append(tsr); shp.append(s_); l2i.append(scale @ P._lidar2img(K_V, R_V, t_V))
    feats = {"camera_feature_1": cf1.astype(np.float32), "camera_feature_2": cf2.astype(np.float32), "camera_feature": np.stack(bev).astype(np.float32),
             "lidar2img": np.stack(l2i).astype(np.float32), "img_shape": np.array(shp, dtype=np.float32), "ego_status": ego_status}
    label = np.array([P._relative_pose(ego.T(t + k * STEP_US), Tn) for k in range(1, K.NUM_POSES + 1)], dtype=np.float64)   # (8,3) x,y,heading, rig frame at t
    return feats, label, command, route, lag, cmd_rg

def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no-model", action="store_true"); ap.add_argument("--bonnet-rows", type=int, default=None); a = ap.parse_args()
    global BONNET_MASK_ROWS
    if a.bonnet_rows is not None: BONNET_MASK_ROWS = a.bonnet_rows
    print("bonnet mask rows:", BONNET_MASK_ROWS, flush=True)
    clips = pd.read_csv(HERE / "pilot_clips.csv"); C.assert_train(clips.uuid)
    if a.limit: clips = clips.head(a.limit)
    model = None
    if not a.no_model:
        from drivejepa.bridge import BridgedDriveJEPA
        t0 = time.time(); model = BridgedDriveJEPA(device="cpu"); print(f"model up in {time.time()-t0:.0f} s; planner {os.path.realpath(P.__file__)} v{K.ADAPTER_VERSION}", flush=True)
    rows = []; cliprows = []
    for r in clips.itertuples():
        tc = time.time(); d, ts, ego, ft, R_N, t_N = load_clip(r.uuid, r.chunk)
        tA = anchors_for(ego); near = lambda t: int(np.argmin(np.abs(ts - t)))
        pairs = {t: (near(t - STEP_US), near(t)) for t in tA}; need = sorted({j for p in pairs.values() for j in p})
        maps = build_maps(ft, R_N); frames = decode(d / f"{r.uuid}.{C.FRONT}.mp4", need)
        rect = {j: cv2.cvtColor(rectify(f, maps), cv2.COLOR_BGR2RGB) for j, f in frames.items()}; frames = None
        try: rg = RouteGeneratorRecorded(ego.p.copy())
        except Exception as e: rg = None; print("RouteGeneratorRecorded failed for", r.uuid, repr(e)[:120], flush=True)   # noqa: BLE001
        (RECT / r.uuid).mkdir(parents=True, exist_ok=True)
        for k, t in enumerate(tA):
            jo, jn = pairs[t]; feats, label, command, route, lag, cmd_rg = sample(ego, t, rg, rect[jo], rect[jn])
            tok = f"{r.uuid[:8]}_{t // 1000:05d}"; cd = CACHE / r.uuid / tok; cd.mkdir(parents=True, exist_ok=True)
            with gzip.open(cd / "pad_feature.gz", "wb", compresslevel=1) as f: pickle.dump({k_: torch.tensor(v) for k_, v in feats.items()}, f)
            with gzip.open(cd / "pad_target.gz", "wb", compresslevel=1) as f: pickle.dump({"trajectory": torch.tensor(label), "token": tok}, f)
            row = dict(type=r.type, uuid=r.uuid, token=tok, t_us=t, frame_older=jo, frame_newer=jn, dt_older_us=int(ts[jo] - (t - STEP_US)), dt_newer_us=int(ts[jn] - t), pair_gap_us=int(ts[jn] - ts[jo]),
                       n_status_rows=len(feats["ego_status"]), vx=float(feats["ego_status"][-1, 3]), vy=float(feats["ego_status"][-1, 4]), ax=float(feats["ego_status"][-1, 5]), ay=float(feats["ego_status"][-1, 6]),
                       command="LSRU"[int(np.argmax(command))], command_routegen=cmd_rg, speed=float(np.hypot(feats["ego_status"][-1, 3], feats["ego_status"][-1, 4])), dyn_lag_us=lag, label_x8=float(label[-1, 0]), label_y8=float(label[-1, 1]), label_len=float(np.hypot(*np.diff(np.vstack([[0, 0], label[:, :2]]), axis=0).T).sum()))
            if model is not None:
                tf = time.time(); out = model({k_: v[None] for k_, v in feats.items()}); row["secs_forward"] = time.time() - tf
                props = np.asarray(out["proposals"])[0].astype(np.float32); S = np.asarray(out["pdm_score"])[0].astype(np.float32)
                assert props.shape == (32, 8, 3) and S.shape == (32,)
                (MENUS / r.uuid).mkdir(parents=True, exist_ok=True); np.savez(MENUS / r.uuid / f"{tok}.npz", proposals=props, pdm_score=S, label=label, ego_status=feats["ego_status"], route_rig=route.astype(np.float32), t_us=t)
                mets = [R1.proposal_metrics(p) for p in props]; ch = int(S.argmax()); ends = np.array([m["end_speed"] for m in mets])
                row.update(fastest_end=float(ends.max()), median_end=float(np.median(ends)), chosen_idx=ch, chosen_step1=float(mets[ch]["speeds"][0]), chosen_end=float(ends[ch]), chosen_len=float(mets[ch]["total_len"]),
                           chosen_max_abs_y=float(mets[ch]["max_abs_y"]), n_lat1=int(sum(m["max_abs_y"] >= 1.0 for m in mets)), n_lat1_moving=int(sum(m["max_abs_y"] >= 1.0 and m["total_len"] >= 2.0 for m in mets)),
                           max_prop_abs_y=float(max(m["max_abs_y"] for m in mets)), max_prop_len=float(max(m["total_len"] for m in mets)))
            rows.append(row)
        # keep rectified frames for C1/C2 for a few anchors (PNG, lossless), not all
        keep = sorted({pairs[t][1] for t in tA[::8]} | {pairs[t][0] for t in tA[::8]})
        for j in keep: cv2.imwrite(str(RECT / r.uuid / f"frame_{j:04d}.png"), cv2.cvtColor(rect[j], cv2.COLOR_RGB2BGR))
        cliprows.append(dict(type=r.type, uuid=r.uuid, chunk=r.chunk, n_anchors=len(tA), n_frames=len(ts), invalid_px_frac=maps[2], native_t=t_N.round(3).tolist(), mount_offset_m=float(np.linalg.norm(t_V - t_N)),
                             rot_diff_deg=float(np.degrees(np.arccos(np.clip((np.trace(R_V.T @ R_N) - 1) / 2, -1, 1)))), native_pp=ft["principal_point"], max_angle=ft["max_angle"], secs=round(time.time() - tc, 1)))
        print(cliprows[-1], flush=True)
        sfx = "_test" if a.limit else ""
        pd.DataFrame(rows).to_csv(HERE / f"samples{sfx}.csv", index=False); pd.DataFrame(cliprows).to_csv(HERE / f"clips_converted{sfx}.csv", index=False)
    if model is not None: model.close()
    print("samples", len(rows))

if __name__ == "__main__":
    main()
