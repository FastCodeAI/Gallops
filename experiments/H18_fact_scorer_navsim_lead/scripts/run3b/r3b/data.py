"""run 3B data: one sample = what the encoder, the decoders, the planner and the scorer read, NAVSIM or LEAD.

NAVSIM  Phase 1's dataset (phase1/data/dataset.py, imported unchanged; targets-v2 local copy): images (4 cams L0 F0 R0 B0 x 4
        frames, 256 x 512), lidar2img, agent targets. run 3A's ego_status (4, 11) from its cache (drivejepa4v_feature.gz). The
        planner's target = the human path (the fact labels' row 1 'human'). Facts: pack/navsim.
LEAD    lead_inputs_v2 records; images from the local views (lead_inputs_v2/views, 2048 x 256 = L0 F0 R0 B0); agent targets
        from the v2 boxes with the Bench2Drive rule (|x|, |y| <= 32 m, not underground, a Phase 1 class; nearest 300);
        ego_status from the v1 sidecar; target = LEAD's expert path (= the labels' 'human'). Facts: pack/lead.
Scorer paths per sample: the expert (row 0 of the pack rows) + the balanced draw from the pool (per yes/no fact: up to 16
failing rows and, for each, its nearest passing row, union capped at 128 = KNOB DRAW_CAP), padded to 1 + DRAW_CAP.
The planner's 32 are added on the GPU and labelled live (labeller.py)."""
import gzip
import os
import pickle
import sys
import zlib
from collections import OrderedDict

import cv2
import numpy as np
import torch

sys.path.insert(0, "{runs}/run3a-adapted")
from phase1.data.dataset import Phase1Dataset  # noqa: E402

cv2.setNumThreads(1)
DEVKIT = "{data.drivejepa_tree}/navsim_v2"
PACK = "{data.paddock_data}/run3b/pack"
NAV_CACHE = "{data.p3_renesas}/wajepa_tokens_v2"
MC_ROOT = "{data.p3_renesas}/train_metric_cache_v2_pruned"
TEACHER_ROOT = "{data.p3_renesas}/anchors_scores_index_v2_rearok_lane"  # run 3A's teacher lists (compute_navsim_score.TEACHER_DIR)
N_TEACH = 4                                                            # run 3A's num_pseudo_targets
V2 = "{data.paddock_data}/lead_inputs_v2"
LEAD360 = "{data.paddock_dump}/lead360/"
LEAD360_SHIFT = "{data.paddock_dump}/lead360_perturbated/"
TARGETS_LOCAL = "{data.paddock_data}/run3b/phase1_local/targets-v2"
DRAW_PER, DRAW_CAP = 16, 128        # KNOBS
FACTS_YN = ("hit", "offroad", "left_route", "red", "stop", "wrong_way", "comfort")
RANGE_M, MAX_AGENTS = 32.0, 300
NAV, LEAD = 0, 2


def anchors():
    a8 = np.load(f"{DEVKIT}/data/8192.npy")[:, 4::5].astype(np.float32)
    add = np.load("{data.paddock_data}/run3a-facts/carla_v2/anchors_lead_K1024.npy")[:, 4::5].astype(np.float32)
    return a8, np.concatenate([a8, add])


class Pack:
    def __init__(self, name):
        d = pickle.load(open(f"{PACK}/{name}_meta.pkl", "rb"))
        self.meta, self.path, self._fh, self._pid = d["meta"], f"{PACK}/{name}.bin", None, None

    def blob(self, i):
        if self._fh is None or self._pid != os.getpid():
            self._fh, self._pid = open(self.path, "rb"), os.getpid()
        m = self.meta[i]
        self._fh.seek(m["offset"])
        return pickle.loads(zlib.decompress(self._fh.read(m["nbytes"])))


def yes_no(o, k):
    if k == "red":
        return o["red"] == 1
    if k == "comfort":
        return o["comfort_step"]
    return o[k].astype(bool)


def draw(o, anch, rng, navsim):
    """balanced draw: pool rows (>= 1) -> chosen row indices."""
    n = len(o["a"])
    pool = np.arange(1, n)
    xy = anch[o["anchor_pool"].astype(np.int64)][:, :, :2]  # rows 1..n-1
    chosen = []
    for k in FACTS_YN:
        if k == "comfort" and not navsim:
            continue
        fail = yes_no(o, k)[1:].any(1)
        F, Pp = pool[fail], pool[~fail]
        if not len(F):
            continue
        f = rng.choice(F, min(DRAW_PER, len(F)), replace=False)
        chosen += f.tolist()
        if len(Pp):
            d = np.linalg.norm(xy[f - 1][:, None] - xy[Pp - 1][None], axis=-1).mean(-1)  # (f, passing)
            taken = set()
            for i in range(len(f)):
                for j in np.argsort(d[i], kind="stable"):
                    if Pp[j] not in taken:
                        taken.add(int(Pp[j]))
                        break
            chosen += sorted(taken)
    u = np.unique(np.asarray(chosen, np.int64))
    if len(u) > DRAW_CAP:
        u = np.sort(rng.choice(u, DRAW_CAP, replace=False))
    return u


def scorer_rows(o, rows, anch, navsim):
    """the expert + the drawn rows, padded to 1 + DRAW_CAP: paths and their true facts / score parts."""
    idx = np.r_[0, rows].astype(np.int64)
    n = 1 + DRAW_CAP
    paths = np.zeros((n, 8, 3), np.float32)
    paths[0] = o["human"]
    paths[1:len(idx)] = anch[o["anchor_pool"][rows - 1].astype(np.int64)]
    valid = np.zeros(n, bool)
    valid[:len(idx)] = True
    f = {}
    for k in ("hit", "offroad", "left_route", "stop", "wrong_way"):
        f[k] = np.zeros((n, 8), np.float32)
        f[k][:len(idx)] = o[k][idx]
    f["red"] = np.zeros((n, 8), np.float32)
    f["red"][:len(idx)] = o["red"][idx] == 1
    f["red_known"] = np.zeros((n, 8), bool)
    f["red_known"][:len(idx)] = o["red"][idx] >= 0
    f["comfort"] = np.zeros((n, 8), np.float32)
    f["comfort"][:len(idx)] = o["comfort_step"][idx]
    f["metres_sofar"] = np.zeros((n, 8), np.float32)
    f["metres_sofar"][:len(idx)] = np.cumsum(o["metres"][idx].astype(np.float64), -1)
    f["ttc"] = np.zeros((n, 8), np.float32)
    f["ttc"][:len(idx)] = np.minimum(o["ttc"][idx].astype(np.float32), 4.0)
    f["cutin_brake"] = np.zeros((n, 8), np.float32)
    f["cutin_brake"][:len(idx)] = np.minimum(o["cutin_brake"][idx].astype(np.float32), 8.0)
    f["lane_off"] = np.zeros((n, 8), np.float32)
    f["lane_off"][:len(idx)] = o["lane_off"][idx]
    a = np.zeros(n, np.float32)
    b = np.zeros(n, np.float32)
    a[:len(idx)], b[:len(idx)] = o["a"][idx], o["b"][idx]
    return paths, valid, f, a, b


class Run3BData(torch.utils.data.Dataset):
    """global index: [0, n_nav) NAVSIM, [n_nav, n_nav + n_lead) LEAD. split 'train' or 'heldout'."""

    def __init__(self, split, p1_cfg, seed=0):
        self.split = split
        cfg = dict(p1_cfg)
        cfg["paths"] = dict(cfg["paths"], targets=TARGETS_LOCAL)
        self.p1 = Phase1Dataset("navsim", cfg, split=split)
        import json
        index = json.load(open(f"{TARGETS_LOCAL}/index.json"))
        keep = (lambda e: not e["heldout"]) if split == "train" else (lambda e: e["heldout"])
        p1_tokens = [e["token"] for e in index if e["source"] == "navsim" and keep(e)]
        self.nav_pack, self.lead_pack = Pack("navsim"), Pack("lead")
        nav_pos = {m["token"]: i for i, m in enumerate(self.nav_pack.meta)}
        self.nav = [(k, nav_pos[t]) for k, t in enumerate(p1_tokens) if t in nav_pos]
        self.nav_missing = len(p1_tokens) - len(self.nav)
        held = split == "heldout"
        self.lead = [i for i, m in enumerate(self.lead_pack.meta) if m["heldout"] == held]
        self.a8, self.a9 = anchors()
        self.seed = seed
        self._v2, self._egos = OrderedDict(), OrderedDict()
        ov = json.load(open("{runs}/run3a/facts/lead/carla/lead360_overlap.json"))["routes"]
        self.zrel = {k.split("/")[-1][:-4]: k for k in ov}

    def __len__(self):
        return len(self.nav) + len(self.lead)

    # ----------------------------------------------------------------------------------- sampling weights
    def weights(self):
        """(nav weights, lead weights): stop-sign frames (both kinds in the pool) x2 on both; LEAD ped / cyclist in path x4."""
        wn = np.array([2.0 if self.nav_pack.meta[j]["stop_both"] else 1.0 for _, j in self.nav])
        wl = np.array([(2.0 if self.lead_pack.meta[j]["stop_both"] else 1.0) * (4.0 if self.lead_pack.meta[j]["vru_path"] else 1.0)
                       for j in self.lead])
        return wn, wl

    # ----------------------------------------------------------------------------------- items
    def _rng(self, i):
        if getattr(self, "deterministic", False):  # evaluation: the same draw every time
            return np.random.default_rng([self.seed, i])
        return np.random.default_rng([self.seed, i, int.from_bytes(os.urandom(4), "little")])

    def __getitem__(self, i):
        if i < len(self.nav):
            return self.nav_item(i)
        return self.lead_item(i - len(self.nav))

    def nav_item(self, i):
        k, j = self.nav[i]
        s = self.p1[k]
        m = self.nav_pack.meta[j]
        assert s["meta"]["token"] == m["token"]
        o = self.nav_pack.blob(j)
        ego = pickle.load(gzip.open(f"{NAV_CACHE}/{m['log']}/{m['token']}/drivejepa4v_feature.gz"))["ego_status"]
        rows = draw(o, self.a8, self._rng(i), True)
        paths, valid, f, a, b = scorer_rows(o, rows, self.a8, True)
        tidx = np.load(f"{TEACHER_ROOT}/{m['log']}/unknown/{m['token']}/metric_cache.pkl.npy")
        if len(tidx) > N_TEACH:  # run 3A: np.random.choice(idx_arr, num_pseudo_targets, replace=False)
            tidx = self._rng(i + 2 * 10 ** 7).choice(tidx, N_TEACH, replace=False)
        teachers = self.a8[np.asarray(tidx, np.int64)] if len(tidx) else np.zeros((0, 8, 3), np.float32)
        t = s["targets"]
        tgt = dict(dataset=0, agent_cls=t["agent_cls"], agent_box=t["agent_box"], agent_vel=t["agent_vel"])
        return dict(images=s["images"], lidar2img=s["lidar2img"], ego_status=torch.as_tensor(np.asarray(ego, np.float32)),
                    expert=torch.as_tensor(o["human"]), ds=NAV, targets=tgt, paths=torch.as_tensor(paths), valid=torch.as_tensor(valid),
                    facts={k_: torch.as_tensor(v) for k_, v in f.items()}, a=torch.as_tensor(a), b=torch.as_tensor(b),
                    R=torch.as_tensor(o["expert_metres"]), ntg=bool(o["nothing_to_gain"]),
                    teachers=torch.as_tensor(np.asarray(teachers, np.float32)), n_drawn=len(rows),
                    meta=dict(ds=NAV, token=m["token"], log=m["log"], mc=f"{MC_ROOT}/{m['log']}/unknown/{m['token']}/metric_cache.pkl",
                              pack_i=j))

    def _route(self, name):
        if name not in self._v2:
            self._v2[name] = pickle.load(gzip.open(f"{V2}/routes/{name}"))["records"]
            if len(self._v2) > 8:
                self._v2.popitem(last=False)
        return self._v2[name]

    def _ego(self, route):
        if route not in self._egos:
            self._egos[route] = pickle.load(gzip.open(f"{V2}/ego_status/{route}.pkl.gz"))
            if len(self._egos) > 8:
                self._egos.popitem(last=False)
        return self._egos[route]

    def lead_item(self, i):
        j = self.lead[i]
        m = self.lead_pack.meta[j]
        r = self._route(m["v2_file"])[m["v2_i"]]
        assert r["token"] == m["token"] and bool(r["shifted"]) == m["shifted"]
        o = self.lead_pack.blob(j)
        im = r["images"]
        out = np.zeros((4, 4, 3, 256, 512), np.uint8)
        from_archive = 0
        for t, mem in enumerate(im["members"]):
            pre = "shift_" if "/rgb_perturbated/" in mem else ""  # unpack_views.view_path
            p = f"{V2}/views/{r['route']}/{pre}{os.path.splitext(os.path.basename(mem))[0]}.jpg"
            x = cv2.imread(p, cv2.IMREAD_COLOR)
            if x is None:  # not unpacked (yet): the archive, same crop and resize (counted: 'views_from_archive')
                import zipfile
                with zipfile.ZipFile(im["archive"]) as zf:
                    st = cv2.imdecode(np.frombuffer(zf.read(mem), np.uint8), cv2.IMREAD_COLOR)
                x = np.concatenate([cv2.resize(st[v["crop_rows"][0]:v["crop_rows"][1], v["tile_x"][0]:v["tile_x"][1]], (512, 256),
                                               interpolation=cv2.INTER_LINEAR) for v in im["views"]], 1)
                from_archive += 1
            for c in range(4):
                out[c, t] = x[:, 512 * c:512 * (c + 1), ::-1].transpose(2, 0, 1)
        B = r["boxes"]
        keep = np.where(B["in_range"] & ~B["underground"] & (B["cls5"] >= 0))[0]
        d = np.hypot(B["box"][keep, 0], B["box"][keep, 1])
        keep = keep[np.argsort(d, kind="stable")[:MAX_AGENTS]]
        tgt = dict(dataset=1,  # the criterion's class-merge rule: CARLA (Bench2Drive's), emergency told apart
                   agent_cls=torch.as_tensor(B["cls5"][keep].astype(np.int64)), agent_box=torch.as_tensor(B["box"][keep].astype(np.float32)),
                   agent_vel=torch.as_tensor(B["vel"][keep].astype(np.float32)))
        ego = self._ego(r["route"])[(r["token"], bool(r["shifted"]))]
        rows = draw(o, self.a9, self._rng(i + 10 ** 7), False)
        paths, valid, f, a, b = scorer_rows(o, rows, self.a9, False)
        te = o["teachers"].astype(np.int64)
        teachers = self.a9[o["anchor_pool"][te - 1].astype(np.int64)] if len(te) else np.zeros((0, 8, 3), np.float32)
        arch = im["archive"]
        zrel = arch[len(LEAD360):] if arch.startswith(LEAD360) else arch[len(LEAD360_SHIFT):]
        return dict(images=torch.from_numpy(out), lidar2img=torch.as_tensor(r["lidar2img"], dtype=torch.float32),
                    ego_status=torch.as_tensor(np.asarray(ego, np.float32)), expert=torch.as_tensor(np.asarray(r["ego_future"], np.float32)),
                    ds=LEAD, targets=tgt, paths=torch.as_tensor(paths), valid=torch.as_tensor(valid),
                    facts={k_: torch.as_tensor(v) for k_, v in f.items()}, a=torch.as_tensor(a), b=torch.as_tensor(b),
                    R=torch.as_tensor(o["expert_metres"]), ntg=bool(o["nothing_to_gain"]),
                    teachers=torch.as_tensor(np.asarray(teachers, np.float32)), n_drawn=len(rows),
                    meta=dict(views_from_archive=from_archive, ds=LEAD, token=r["token"], shifted=bool(r["shifted"]), route=r["route"], zrel=zrel,
                              frame=int(r["frame"]), set=r["set"], scenario=r["scenario"], pack_i=j))

    def label_meta(self, g):
        """what a labeller needs to load the scene, from the global index alone (no images read)."""
        if g < len(self.nav):
            m = self.nav_pack.meta[self.nav[g][1]]
            return dict(ds=NAV, token=m["token"], mc=f"{MC_ROOT}/{m['log']}/unknown/{m['token']}/metric_cache.pkl")
        m = self.lead_pack.meta[self.lead[g - len(self.nav)]]
        return dict(ds=LEAD, token=m["token"], zrel=self.zrel[m["route"]], frame=int(m["token"].split("#")[1]), shifted=m["shifted"])


def collate(samples):
    out = dict(images=torch.stack([s["images"] for s in samples]), lidar2img=torch.stack([s["lidar2img"] for s in samples]),
               ego_status=torch.stack([s["ego_status"] for s in samples]), expert=torch.stack([s["expert"] for s in samples]),
               ds=torch.tensor([s["ds"] for s in samples]), targets=[s["targets"] for s in samples],
               paths=torch.stack([s["paths"] for s in samples]), valid=torch.stack([s["valid"] for s in samples]),
               a=torch.stack([s["a"] for s in samples]), b=torch.stack([s["b"] for s in samples]),
               R=torch.stack([s["R"] for s in samples]), ntg=torch.tensor([s["ntg"] for s in samples]),
               teachers=[s["teachers"] for s in samples], n_drawn=torch.tensor([s["n_drawn"] for s in samples]),
               meta=[s["meta"] for s in samples])
    out["facts"] = {k: torch.stack([s["facts"][k] for s in samples]) for k in samples[0]["facts"]}
    return out


class StepSampler(torch.utils.data.Sampler):
    """batch b = 32 NAVSIM + 32 LEAD drawn with replacement by weight; deterministic in (seed, b) so the labellers can be told
    the next batch ahead. Yields lists of global indices."""

    def __init__(self, data, steps, seed=0, n_nav=32, n_lead=32, start=0):
        self.data, self.steps, self.seed, self.n_nav, self.n_lead, self.start = data, steps, seed, n_nav, n_lead, start
        wn, wl = data.weights()
        self.pn, self.pl = wn / wn.sum(), wl / wl.sum()

    def batch(self, b):
        rng = np.random.default_rng([self.seed, 7, b])
        nav = rng.choice(len(self.pn), self.n_nav, replace=True, p=self.pn)
        lead = rng.choice(len(self.pl), self.n_lead, replace=True, p=self.pl) + len(self.data.nav)
        return [int(x) for x in nav] + [int(x) for x in lead]

    def __iter__(self):
        for b in range(self.start, self.steps):
            yield self.batch(b)

    def __len__(self):
        return self.steps - self.start
