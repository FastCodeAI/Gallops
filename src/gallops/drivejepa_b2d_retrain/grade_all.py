"""Step A run, spec v2: every feasible path of the 2,048 on every frame. One npz per clip in
$B2D_DATA/grade_v2: frame table + every graded path's index and each term.
    python -m gallops.drivejepa_b2d_retrain.grade_all [--shard i n] [--clips SUBSTR ...] [--batch 8]
        [--out DIR]"""

import argparse
import json
import os
import time

import numpy as np
import torch

from gallops.drivejepa_b2d_retrain import grade as G2
from gallops.drivejepa_b2d_retrain import wiring as W

D = str(W.DATA)
CTX = f"{D}/context"
V = str(W.VOCAB / "b2d_2048.npy")
ALIAS = {
    "VehicleTurningRoutePedestrian_Town15_Route523_Weathe": "VehicleTurningRoutePedestrian_Town15_Route523_Weather2"  # noqa: E501
}
T = ("n_ped", "n_veh", "n_static", "n_red", "n_stop")
ROUTE = json.load(open(f"{D}/inputs/route_len.json"))


def vocab(dev="cuda"):
    p = torch.from_numpy(np.load(V)[:, 4::5, :].astype(np.float64)).to(dev)
    return G2.to_centre_t(p).float()


def feasible(Vc, v0):
    vi = Vc[:, 0, :2].norm(dim=-1) / 0.5
    dy = torch.rad2deg(Vc[:, 0, 2].abs())
    return ((vi[None] - v0[:, None]).abs() <= 2.0) & (dy[None] < 15.0)


def grade_batch(c, Vc, route, dev="cuda"):
    """c: context dict of a few frames -> per-frame results."""
    ctx = G2.collate2([c], dev)
    g = G2.prepare2(ctx, dev)
    F = g["F"]
    g["route_len"] = torch.full((F,), float(route), device=dev)
    m = feasible(Vc, g["v0"])
    C = max(1, int(m.sum(1).max()))
    order = torch.argsort(m.int(), dim=1, descending=True, stable=True)[:, :C]
    valid = torch.gather(m, 1, order)
    t = G2.terms(g, Vc[order])
    te = G2.terms(g, g["expert"][:, None])
    dref = te["d"][:, 0]
    S, P = G2.grade(t, dref)
    Se, Pe = G2.grade(te, dref)  # dref = the expert's own distance
    return order, valid, t, dref, S, P, te, Se, Pe


def run_clip(clip, Vc, batch, out):
    name = ALIAS.get(clip, clip)
    c = G2.load_clip(CTX, name)
    F = len(c["frame_id"])
    route = ROUTE[clip]
    R = {k: [] for k in ("idx", "S", "P", "d", "outside", "off_share", "bad_m") + T}
    n_per = np.zeros(F, np.int32)
    fr = {
        k: np.zeros(F, np.float32)
        for k in ("d_ref", "exp_S", "exp_P", "exp_d", "exp_outside", "exp_off_share")
    }
    fi = {k: np.zeros(F, np.int16) for k in ("exp_" + x for x in T)}
    for b in range(0, F, batch):
        sl = {k: v[b : b + batch] for k, v in c.items()}
        order, valid, t, dref, S, P, te, Se, Pe = grade_batch(sl, Vc, route)
        v = valid.cpu().numpy()
        n_per[b : b + len(v)] = v.sum(1)
        for k, x in (
            ("idx", order),
            ("S", S),
            ("P", P),
            ("d", t["d"]),
            ("outside", t["outside"]),
            ("off_share", t["off_share"]),
            ("bad_m", t["bad_m"]),
        ) + tuple((k, t[k]) for k in T):
            R[k].append(x.cpu().numpy()[v])
        for k, x in (
            ("d_ref", dref),
            ("exp_S", Se[:, 0]),
            ("exp_P", Pe[:, 0]),
            ("exp_d", te["d"][:, 0]),
            ("exp_outside", te["outside"][:, 0]),
            ("exp_off_share", te["off_share"][:, 0]),
        ):
            fr[k][b : b + len(v)] = x.cpu().numpy()
        for k in T:
            fi["exp_" + k][b : b + len(v)] = te[k][:, 0].cpu().numpy()
    o = dict(
        frame_id=c["frame_id"],
        v0=c["v0"],
        n_paths=n_per,
        idx=np.concatenate(R["idx"]).astype(np.int16),
        S=np.concatenate(R["S"]).astype(np.float32),
        P=np.concatenate(R["P"]).astype(np.float16),
        d=np.concatenate(R["d"]).astype(np.float32),
        outside=np.concatenate(R["outside"]).astype(np.float32),
        off_share=np.concatenate(R["off_share"]).astype(np.float32),
        bad_m=np.concatenate(R["bad_m"]).astype(np.float32),
        route_len=np.float32(route),
        **{k: np.concatenate(R[k]).astype(np.int8) for k in T},
        **fr,
        **fi,
    )
    p = f"{out}/{clip}.npz"
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, **o)
    os.replace(tmp, p)
    return F, int(n_per.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=f"{D}/grade_v2")
    a = ap.parse_args()
    if os.environ.get("P2_MEM_FRAC"):
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ["P2_MEM_FRAC"])
        )  # torch's cache grew to 27 GB on 3.4 GiB of real use and starved M2 (22:04)
    sp = json.load(open(W.SPLITS))
    clips = sorted(sp["train"] + sp["heldout"])[a.shard[0] :: a.shard[1]]
    if a.clips:
        clips = [c for c in clips if any(s in c for s in a.clips)]
    clips = [c for c in clips if not os.path.exists(f"{a.out}/{c}.npz")]
    os.makedirs(a.out, exist_ok=True)
    Vc = vocab()
    t0 = time.time()
    nf = npth = 0
    print(len(clips), "clips", flush=True)
    for i, c in enumerate(clips):
        f, n = run_clip(c, Vc, a.batch, a.out)
        nf += f
        npth += n
        if i % 10 == 0 or i == len(clips) - 1:
            print(
                f"{i + 1}/{len(clips)} frames {nf} paths {npth} {time.time() - t0:.0f} s {npth / max(time.time() - t0, 1e-3):.0f} paths/s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",  # noqa: E501
                flush=True,
            )
    print("GRADE_DONE", nf, npth, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
