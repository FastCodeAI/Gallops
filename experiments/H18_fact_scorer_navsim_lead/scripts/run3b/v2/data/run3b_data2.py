"""run 3B v2 data (run3b/CLAUDE.md v2, sections 1 and 4): one sample = what run 3A epoch 13's model reads, with IMAGES for the
live WA-JEPA encoder in place of the cached tokens, + the detection targets of run 3A's agent decoder D + the planner / scorer
targets exactly as r3b/data.py Run3BData builds them (its split, draw, teachers, weights, StepSampler and label_meta are
inherited / imported, never edited).

Images  uint8 (4 cams, 2 frames, 3, 256, 512), RGB; cameras b0 f0 l0 r0 (run 3A's / lidar2img's order), frames t-0.5 s then t.
        NAVSIM: Phase 1's 256 x 512 copies (phase1-images/navsim: crop round(H*28/1080) rows top and bottom, cv2.resize INTER_LINEAR,
        JPEG q95) of the frames the WA-JEPA cache used (agent_input.cameras[-2], [-1] = Phase 1 frames 2 and 3). The cache builder
        (wajepa-runs/dump_wajepa.py -> bevformer/wajepa.py clips_from_agent_input) applied the same crop / resize to the original
        1920 x 1080 JPEGs, which are only on the NAS; the q95 re-encode is the one difference (check A gives the token difference).
        LEAD: lead_inputs_v2 views (L0 F0 R0 B0 side by side, 2048 x 256 JPEG q95), members f-2 and f (images.members[2:4]),
        reordered to b0 f0 l0 r0 = views (3, 1, 0, 2), as r3b/eval_run3a.py fed run 3A.
        The encoder input = normalise(images) (torchvision ToTensor + ImageNet Normalize arithmetic), clips (B*4, 3, 2, 256, 512).
lidar2img  float32 (4, 4, 4), run 3A's convention (the 0.4-scale 768 x 448 matrix; the model and D apply to_512 themselves).
        NAVSIM: drivejepa4v_feature.gz of run 3A's cache, as is. LEAD: inv(TO_512) @ lead_inputs_v2's 512 x 256 lidar2img.
img_shape  float32 (4, 3) = (448, 768, 3) per camera (NAVSIM: the cached value; LEAD: the same constant, as eval_run3a.py).
ego_status float32 (4, 11). NAVSIM: drivejepa4v_feature.gz. LEAD: lead_inputs_v2/ego_status sidecar (as r3b/data.py).
K, ego2cam float32 (4, 3, 3) / (4, 4, 4) per camera for the 512 x 256 image, with inv(TO_512) @ K4 @ ego2cam = lidar2img:
        NAVSIM: RQ decomposition of TO_512 @ cached lidar2img (float64; K upper triangular with K[2, 2] = 1, ego2cam rigid; 'ego'
        is the frame run 3A's lidar2img maps from). LEAD: the record's K and inv(cam2ego). (augment2.py uses them.)
det     D's target format (DriveJEPAAgentTargetBuilder.agent_targets, run 3A's AgentLoss input): agent_class int64 (P,) 0 vehicle
        1 pedestrian 2 bicycle, agent_box float32 (P, 7) x y z l w h yaw (ego frame), agent_vel float32 (P, 2) vx vy (absolute,
        ego frame), agent_mask bool (P,), agent_count int64 () = agents in range before the cut; P = DET_PAD = 64 (D's), nearest
        first. NAVSIM: run 3A's cached agent_targets.gz (the targets D and run 3A were trained on). LEAD: lead_inputs_v2 boxes with
        in_range (|x|, |y| <= 32 m) & not underground & cls3 >= 0 (cls3: vehicle = cls5 vehicle / emergency, parked cars
        included; pedestrian; bicycle = cls5 cyclist); statics (cls5 3) and unclassed (cls5 -1) have no D class and are left out.
Planner / scorer (copied from r3b/data.py Run3BData.nav_item / lead_item, same RNG streams, same functions draw / scorer_rows):
        expert (8, 3), ds (0 NAVSIM / 2 LEAD), paths (129, 8, 3), valid (129,), facts {k: (129, 8)}, a, b (129,), R (8,), ntg,
        teachers (n, 8, 3) (NAVSIM run 3A's anchor teachers, LEAD pack teachers), n_drawn, meta; label_meta(g) inherited."""
import gzip
import importlib.util
import os
import pickle
import sys

import cv2
import numpy as np
import torch
import yaml

cv2.setNumThreads(1)
R3B_DIR = "{runs}/run3b/r3b"
P1_ROOT = "{runs}/run3a-adapted"
DEVKIT = "{data.drivejepa_tree}/navsim_v2"


def _load_r3b():
    """r3b/data.py as module 'r3b_data' (imported, never edited; no bytecode written next to it)."""
    if "r3b_data" in sys.modules:
        return sys.modules["r3b_data"]
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("r3b_data", f"{R3B_DIR}/data.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["r3b_data"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


R3B = _load_r3b()
P1_CFG = yaml.safe_load(open(f"{P1_ROOT}/phase1/config.yaml"))   # = phase1.epoch_eval.CFG (what r3b's train2.py passes)
StepSampler = R3B.StepSampler                                     # r3b's: 32 NAVSIM + 32 LEAD by weight, deterministic in (seed, b)
NAV, LEAD = R3B.NAV, R3B.LEAD
NAV_CACHE = R3B.NAV_CACHE
PACK_V1 = R3B.PACK                                    # {data.paddock_data}/run3b/pack (r3b's)
PACK_V2 = "{data.paddock_data}/run3b/pack_v2"  # default: LEAD hits with dirtdebris as debris (coordinator, 19 Nov)
TO_512 = np.array([[2 / 3, 0, 0, 0], [0, 0.625, -7, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)  # drive_jepa_model.to_512
TO_512_INV = np.linalg.inv(TO_512)
B0F0L0R0 = (3, 1, 0, 2)          # run 3A camera order, as indices into Phase 1's / lead_inputs' L0 F0 R0 B0
FRAMES = (2, 3)                  # of the 4-frame history (t-1.5, t-1, t-0.5, t): run 3A's encoder reads t-0.5 and t
IMG_SHAPE = (448.0, 768.0, 3.0)  # DriveJEPA4VFeatureBuilder.IMG_SHAPE
DET_PAD = 64                     # DriveJEPAAgentTargetBuilder.MAX_AGENTS (D's training targets)
DET_RANGE_M = 32.0
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # wajepa._TO_TENSOR_NORM


def load_gz(p):
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def rq_calib(l2i_3a):
    """run 3A lidar2img (4, 4) -> K (3, 3) and ego2cam (4, 4) of the 512 x 256 image, float64, K4 @ ego2cam = TO_512 @ l2i_3a."""
    M = TO_512 @ np.asarray(l2i_3a, np.float64)
    A = M[:3, :3]
    P = np.eye(3)[::-1]
    Q, U = np.linalg.qr((P @ A).T)
    Kc = P @ U.T @ P
    Rc = P @ Q.T
    D = np.diag(np.sign(np.diag(Kc)))
    Kc, Rc = Kc @ D, D @ Rc
    s = Kc[2, 2]
    Kc = Kc / s
    t = np.linalg.solve(Kc, M[:3, 3] / s)
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = Rc, t
    return Kc, E


def normalise(images):
    """uint8 / float [0, 255] (..., 3, H, W) -> float32, bit-equal to wajepa's CPU ToTensor + Normalize (x / 255, - mean, / std).
    The divisor is a device tensor: a Python-scalar divisor makes CUDA multiply by 1/255 (differs from CPU by up to 7e-7)."""
    x = images.float() / torch.tensor(255.0, device=images.device)
    shape = [1] * (x.dim() - 3) + [3, 1, 1]
    return (x - torch.tensor(MEAN, device=x.device).view(shape)) / torch.tensor(STD, device=x.device).view(shape)


def to_clips(images):
    """(B, 4, 2, 3, H, W) -> normalised clips (B * 4, 3, 2, H, W), the layout encode_clips / the encoder take."""
    B = images.shape[0]
    return normalise(images).permute(0, 1, 3, 2, 4, 5).reshape(B * 4, 3, 2, *images.shape[-2:])


def _pad_det(cls, box, vel, count, pad):
    n = min(len(cls), pad)
    out = dict(agent_class=torch.zeros(pad, dtype=torch.long), agent_box=torch.zeros(pad, 7), agent_vel=torch.zeros(pad, 2),
               agent_mask=torch.zeros(pad, dtype=torch.bool), agent_count=torch.tensor(int(count)))
    out["agent_class"][:n] = torch.as_tensor(np.asarray(cls[:n], np.int64))
    out["agent_box"][:n] = torch.as_tensor(np.asarray(box[:n], np.float32))
    out["agent_vel"][:n] = torch.as_tensor(np.asarray(vel[:n], np.float32))
    out["agent_mask"][:n] = True
    return out


def lead_det(B, pad=DET_PAD):
    """lead_inputs_v2 boxes -> D's targets (nearest `pad` of the in-range, above-ground boxes with a D class)."""
    keep = np.where(B["in_range"] & ~B["underground"] & (B["cls3"] >= 0))[0]
    d = np.hypot(B["box"][keep, 0], B["box"][keep, 1])
    keep_sorted = keep[np.argsort(d, kind="stable")]
    return _pad_det(B["cls3"][keep_sorted].astype(np.int64), B["box"][keep_sorted], B["vel"][keep_sorted], len(keep), pad)


class Run3BData2(R3B.Run3BData):
    """global index: [0, n_nav) NAVSIM, [n_nav, n_nav + n_lead) LEAD (r3b's index, split, weights, label_meta).
    split 'train' (navtrain minus the 51 held-out logs; LEAD minus the 416 held-out routes) or 'heldout'."""

    def __init__(self, split, seed=0, det_pad=DET_PAD, pack=PACK_V2):
        """pack: the pool / facts pack directory (r3b/data.py's format: navsim.bin, navsim_meta.pkl, lead.bin, lead_meta.pkl).
        Default pack_v2 (LEAD hit labels with static.prop.dirtdebris* as debris; NAVSIM files are links to pack/)."""
        if not os.path.isfile(f"{pack}/lead_meta.pkl"):
            raise FileNotFoundError(f"{pack}/lead_meta.pkl (pass pack=PACK_V1 for r3b's pack)")
        old = R3B.PACK
        R3B.PACK = pack            # r3b's Pack reads the module constant when built; the split / index are built from this pack
        try:
            super().__init__(split, P1_CFG, seed)
        finally:
            R3B.PACK = old
        self.pack = pack
        self.det_pad = det_pad
        self.nav_img_root = self.p1.img_root

    # ------------------------------------------------------------------------------- NAVSIM
    def nav_images(self, paths):
        """Phase 1 row 'images' (L0 F0 R0 B0 x 4 frames) -> uint8 (4, 2, 3, 256, 512), b0 f0 l0 r0, frames t-0.5, t."""
        out = np.empty((4, 2, 3, 256, 512), np.uint8)
        for c, pc in enumerate(B0F0L0R0):
            for t, ft in enumerate(FRAMES):
                p = f"{self.nav_img_root}/{paths[pc][ft]}"
                im = cv2.imread(p, cv2.IMREAD_COLOR)
                if im is None:
                    raise FileNotFoundError(p)
                out[c, t] = im[:, :, ::-1].transpose(2, 0, 1)
        return torch.from_numpy(out)

    def nav_item(self, i):
        k, j = self.nav[i]
        r = self.p1.row(k)
        m = self.nav_pack.meta[j]
        assert r["token"] == m["token"], (r["token"], m["token"])
        base = f"{NAV_CACHE}/{m['log']}/{m['token']}"
        feat = load_gz(f"{base}/drivejepa4v_feature.gz")
        l2i = feat["lidar2img"].float()
        Ks, Es = zip(*[rq_calib(l2i[c].double().numpy()) for c in range(4)])
        det = load_gz(f"{base}/agent_targets.gz")
        if self.det_pad != DET_PAD:
            n = int(det["agent_mask"].sum())
            det = _pad_det(det["agent_class"][:n].numpy(), det["agent_box"][:n].numpy(), det["agent_vel"][:n].numpy(),
                           int(det["agent_count"]), self.det_pad)
        # ---- r3b/data.py Run3BData.nav_item, planner / scorer part, unchanged
        o = self.nav_pack.blob(j)
        rows = R3B.draw(o, self.a8, self._rng(i), True)
        paths, valid, f, a, b = R3B.scorer_rows(o, rows, self.a8, True)
        tidx = np.load(f"{R3B.TEACHER_ROOT}/{m['log']}/unknown/{m['token']}/metric_cache.pkl.npy")
        if len(tidx) > R3B.N_TEACH:
            tidx = self._rng(i + 2 * 10 ** 7).choice(tidx, R3B.N_TEACH, replace=False)
        teachers = self.a8[np.asarray(tidx, np.int64)] if len(tidx) else np.zeros((0, 8, 3), np.float32)
        return dict(images=self.nav_images(r["images"]), lidar2img=l2i, img_shape=feat["img_shape"].float(),
                    ego_status=torch.as_tensor(np.asarray(feat["ego_status"], np.float32)),
                    K=torch.as_tensor(np.stack(Ks), dtype=torch.float32), ego2cam=torch.as_tensor(np.stack(Es), dtype=torch.float32),
                    det=det, expert=torch.as_tensor(o["human"]), ds=NAV, paths=torch.as_tensor(paths), valid=torch.as_tensor(valid),
                    facts={k_: torch.as_tensor(v) for k_, v in f.items()}, a=torch.as_tensor(a), b=torch.as_tensor(b),
                    R=torch.as_tensor(o["expert_metres"]), ntg=bool(o["nothing_to_gain"]),
                    teachers=torch.as_tensor(np.asarray(teachers, np.float32)), n_drawn=len(rows),
                    meta=dict(ds=NAV, token=m["token"], log=m["log"], mc=f"{R3B.MC_ROOT}/{m['log']}/unknown/{m['token']}/metric_cache.pkl",
                              pack_i=j, frames=[[r["images"][pc][ft] for ft in FRAMES] for pc in B0F0L0R0]))

    # ------------------------------------------------------------------------------- LEAD
    def lead_images(self, r):
        """lead_inputs_v2 views of members f-2, f -> uint8 (4, 2, 3, 256, 512), b0 f0 l0 r0. Returns (images, n read from archive)."""
        im = r["images"]
        out = np.empty((4, 2, 3, 256, 512), np.uint8)
        from_archive = 0
        for t, mem in enumerate(im["members"][2:4]):
            pre = "shift_" if "/rgb_perturbated/" in mem else ""  # unpack_views.view_path
            p = f"{R3B.V2}/views/{r['route']}/{pre}{os.path.splitext(os.path.basename(mem))[0]}.jpg"
            x = cv2.imread(p, cv2.IMREAD_COLOR)
            if x is None:  # not unpacked: the archive, same crop and resize (r3b/data.py's fallback; counted)
                import zipfile
                with zipfile.ZipFile(im["archive"]) as zf:
                    st = cv2.imdecode(np.frombuffer(zf.read(mem), np.uint8), cv2.IMREAD_COLOR)
                x = np.concatenate([cv2.resize(st[v["crop_rows"][0]:v["crop_rows"][1], v["tile_x"][0]:v["tile_x"][1]], (512, 256),
                                               interpolation=cv2.INTER_LINEAR) for v in im["views"]], 1)
                from_archive += 1
            for c, vc in enumerate(B0F0L0R0):
                out[c, t] = x[:, 512 * vc:512 * (vc + 1), ::-1].transpose(2, 0, 1)
        return torch.from_numpy(out), from_archive

    def lead_item(self, i):
        j = self.lead[i]
        m = self.lead_pack.meta[j]
        r = self._route(m["v2_file"])[m["v2_i"]]
        assert r["token"] == m["token"] and bool(r["shifted"]) == m["shifted"]
        images, from_archive = self.lead_images(r)
        l2i512 = np.asarray(r["lidar2img"], np.float64)[list(B0F0L0R0)]
        l2i = np.stack([TO_512_INV @ l2i512[c] for c in range(4)])
        Ks = np.asarray(r["K"], np.float64)[list(B0F0L0R0)]
        Es = np.stack([np.linalg.inv(np.asarray(r["cam2ego"][c], np.float64)) for c in B0F0L0R0])
        det = lead_det(r["boxes"], self.det_pad)
        ego = self._ego(r["route"])[(r["token"], bool(r["shifted"]))]
        # ---- r3b/data.py Run3BData.lead_item, planner / scorer part, unchanged
        o = self.lead_pack.blob(j)
        rows = R3B.draw(o, self.a9, self._rng(i + 10 ** 7), False)
        paths, valid, f, a, b = R3B.scorer_rows(o, rows, self.a9, False)
        te = o["teachers"].astype(np.int64)
        teachers = self.a9[o["anchor_pool"][te - 1].astype(np.int64)] if len(te) else np.zeros((0, 8, 3), np.float32)
        arch = r["images"]["archive"]
        zrel = arch[len(R3B.LEAD360):] if arch.startswith(R3B.LEAD360) else arch[len(R3B.LEAD360_SHIFT):]
        return dict(images=images, lidar2img=torch.as_tensor(l2i, dtype=torch.float32), img_shape=torch.tensor([IMG_SHAPE] * 4),
                    ego_status=torch.as_tensor(np.asarray(ego, np.float32)),
                    K=torch.as_tensor(Ks, dtype=torch.float32), ego2cam=torch.as_tensor(Es, dtype=torch.float32), det=det,
                    expert=torch.as_tensor(np.asarray(r["ego_future"], np.float32)), ds=LEAD, paths=torch.as_tensor(paths),
                    valid=torch.as_tensor(valid), facts={k_: torch.as_tensor(v) for k_, v in f.items()}, a=torch.as_tensor(a),
                    b=torch.as_tensor(b), R=torch.as_tensor(o["expert_metres"]), ntg=bool(o["nothing_to_gain"]),
                    teachers=torch.as_tensor(np.asarray(teachers, np.float32)), n_drawn=len(rows),
                    meta=dict(views_from_archive=from_archive, ds=LEAD, token=r["token"], shifted=bool(r["shifted"]), route=r["route"],
                              zrel=zrel, frame=int(r["frame"]), set=r["set"], scenario=r["scenario"], pack_i=j,
                              frames=[int(x) for x in r["images"]["frames"][2:4]], members=list(r["images"]["members"][2:4])))


def collate2(samples):
    """list of samples -> batch. Tensors stacked on dim 0; det = dict of stacked D targets; teachers and meta stay lists."""
    st = lambda k: torch.stack([s[k] for s in samples])  # noqa: E731
    out = {k: st(k) for k in ("images", "lidar2img", "img_shape", "ego_status", "K", "ego2cam", "expert", "paths", "valid", "a", "b", "R")}
    out["det"] = {k: torch.stack([s["det"][k] for s in samples]) for k in samples[0]["det"]}
    out["facts"] = {k: torch.stack([s["facts"][k] for s in samples]) for k in samples[0]["facts"]}
    out["ds"] = torch.tensor([s["ds"] for s in samples])
    out["ntg"] = torch.tensor([s["ntg"] for s in samples])
    out["n_drawn"] = torch.tensor([s["n_drawn"] for s in samples])
    out["teachers"] = [s["teachers"] for s in samples]
    out["meta"] = [s["meta"] for s in samples]
    return out


def run3a_features(batch, tokens):
    """the dict run 3A's DriveJEPAModel.forward reads, from a (device) batch and the encoder's tokens (B * 4, 512, 1024)."""
    B = batch["images"].shape[0]
    return dict(wajepa_tokens=tokens.view(B, 4, 512, 1024), lidar2img=batch["lidar2img"], img_shape=batch["img_shape"],
                ego_status=batch["ego_status"])
