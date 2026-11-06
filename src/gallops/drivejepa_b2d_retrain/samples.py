"""Phase 2 samples. One sample = one graded frame: two cached `native` frames (Phase 1's cache, t - 0.5 s and
t; a clip's first frame is repeated, as the runner's history does on its first tick), ego status as the
runner sends it, the expert's 8 rear-axle poses, the frame's teachers, the expert's distance (field dref)
and the route length, and the scorer context for the grade in the loop."""

import json
import os

import numpy as np
import torch

from gallops.drivejepa_b2d_retrain import grade as G2
from gallops.drivejepa_b2d_retrain import wiring as W

D = str(W.DATA)
ALIAS = {
    "VehicleTurningRoutePedestrian_Town15_Route523_Weathe": "VehicleTurningRoutePedestrian_Town15_Route523_Weather2"  # noqa: E501
}
CK = tuple(k for k in G2.SC.REQUIRED + G2.SC.OPTIONAL) + G2.XK


class Tables:
    """Everything indexed by frame, loaded once in the parent so the loader workers share it."""

    def __init__(self, split, limit_clips=None):
        sp = json.load(open(W.SPLITS))
        self.clips = sorted(sp[split])[:limit_clips] if limit_clips else sorted(sp[split])
        z = np.load(W.VOCAB / "paths_cache.npz", allow_pickle=True)
        names = [str(c) for c in z["clip_names"]]
        rear, cid, fr = z["rear"][:, 4::5, :].astype(np.float32), z["clip_id"], z["frame"]
        path = {(names[c], int(f)): i for i, (c, f) in enumerate(zip(cid, fr))}  # noqa: B905
        e = np.load(f"{D}/inputs/ego_inputs.npz")
        ego = {(str(c), int(f)): i for i, (c, f) in enumerate(zip(e["clip"], e["frame"]))}  # noqa: B905
        t = np.load(os.environ.get("P2_TEACHERS", f"{D}/teachers_2048_v2.npz"))
        route = json.load(open(f"{D}/inputs/route_len.json"))
        tpos = {str(f): i for i, f in enumerate(t["frame_id"])}
        self.t_off, self.t_idx, t_dref = t["offsets"], t["idx"], t["d_ref"]
        self.fc_frames = json.load(open(W.CACHE / "index.json"))["frames"]
        self.ctx, self.items = {}, []
        for clip in self.clips:
            name = ALIAS.get(clip, clip)
            c = G2.load_clip(f"{D}/context", name)
            self.ctx[clip] = c
            names_f = {n: i for i, n in enumerate(self.fc_frames[clip])}
            for j, fid in enumerate(c["frame_id"]):
                n = int(str(fid).rsplit("|", 1)[1]) // 100
                ti = tpos[str(fid)]
                self.items.append(
                    (
                        clip,
                        j,
                        names_f[f"{n:05d}.jpg"],
                        path[(clip, n)],
                        ego[(clip, n)],
                        ti,
                        float(t_dref[ti]),
                        route[clip],
                    )
                )
        self.rear = rear
        self.speed, self.accel, self.one_hot = e["speed"], e["accel"], e["one_hot"]
        m = torch.load(W.RUNS / "ade_set.pt", weights_only=False)["moments"][0]
        self.lidar2img = torch.as_tensor(np.asarray(m["lidar2img"]))
        self.img_shape = m["img_shape"]

    def __len__(self):
        return len(self.items)


class Samples(torch.utils.data.Dataset):
    def __init__(self, T):
        self.T = T
        self.fc = None

    def __len__(self):
        return len(self.T)

    def __getitem__(self, k):
        from gallops.drivejepa_b2d_retrain.frames import FrameCache, _to_tensor

        if self.fc is None:
            self.fc = FrameCache()
        T = self.T
        clip, j, i, pi, ei, ti, dref, rl = T.items[k]
        new, old = self.fc.frame(clip, i), self.fc.frame(clip, max(i - 1, 0))
        ego = np.concatenate(
            [np.zeros(3, np.float32), [T.speed[ei], 0.0], [T.accel[ei], 0.0], T.one_hot[ei]]
        ).astype(np.float32)
        return dict(
            k=k,
            new=_to_tensor(new),
            old=_to_tensor(old),
            ego=torch.from_numpy(ego)[None],
            traj=torch.from_numpy(T.rear[pi]),
            dref=dref,
            route=rl,
            teachers=T.t_idx[T.t_off[ti] : T.t_off[ti + 1]].astype(np.int64),
            ctx={q: v[j : j + 1] for q, v in T.ctx[clip].items()},
        )


def collate(rows):
    return dict(
        k=torch.tensor([r["k"] for r in rows]),
        new=torch.stack([r["new"] for r in rows]),
        old=torch.stack([r["old"] for r in rows]),
        ego=torch.stack([r["ego"] for r in rows]),
        traj=torch.stack([r["traj"] for r in rows]),
        dref=torch.tensor([r["dref"] for r in rows], dtype=torch.float32),
        route=torch.tensor([r["route"] for r in rows], dtype=torch.float32),
        teachers=[r["teachers"] for r in rows],
        ctx=G2.collate2([r["ctx"] for r in rows], "cpu"),
    )
