"""Teacher index + the Step A report numbers, from the per-clip grade files.
python -m gallops.drivejepa_b2d_retrain.teachers -> $B2D_DATA/teachers_2048_v2.npz,
    step_a_v2_numbers.json"""

import json
import os
import re

import numpy as np

from gallops.drivejepa_b2d_retrain import wiring as W

D = str(W.DATA)
G = f"{D}/grade_v2"
OUT = f"{D}/teachers_2048_v2.npz"
RES = f"{D}/step_a_v2_numbers.json"
ALIAS = {
    "VehicleTurningRoutePedestrian_Town15_Route523_Weathe": "VehicleTurningRoutePedestrian_Town15_Route523_Weather2"  # noqa: E501
}
T = ("n_ped", "n_veh", "n_static", "n_red", "n_stop")
sp = json.load(open(W.SPLITS))
rng = np.random.default_rng(20261106)
fid, off, tix, split, kind, nT, dref, nfeas = [], [0], [], [], [], [], [], []
exp = {k: [] for k in ("S", "P", "outside") + T}
fire = {
    s: {k: 0 for k in T + ("outside", "P_lt_1", "clean", "paths", "teacher")} for s in ("train", "heldout")
}
for s in ("train", "heldout"):
    for clip in sp[s]:
        z = np.load(f"{G}/{clip}.npz")
        c = np.load(f"{D}/context/{ALIAS.get(clip, clip)}.npz")
        turn = np.abs(np.degrees(np.arctan2(np.sin(c["expert"][:, -1, 2]), np.cos(c["expert"][:, -1, 2]))))
        n = z["n_paths"]
        o = np.r_[0, np.cumsum(n)]
        idx, S = z["idx"], z["S"]
        tot = sum(z[k].astype(int) for k in T)
        P = z["P"].astype(np.float32)
        out = z["outside"]
        f = fire[s]
        f["paths"] += len(S)
        f["teacher"] += int((S >= 0.95).sum())
        f["clean"] += int(((tot == 0) & (out == 0)).sum())
        f["outside"] += int((out > 0).sum())
        f["P_lt_1"] += int((P < 1).sum())
        for k in T:
            f[k] += int((z[k] > 0).sum())
        for i in range(len(n)):
            t = idx[o[i] : o[i + 1]][S[o[i] : o[i + 1]] >= 0.95]
            nT.append(len(t))
            if len(t) > 256:
                t = rng.choice(t, 256, replace=False)
            tix.append(t.astype(np.int16))
            off.append(off[-1] + len(t))
            fid.append(str(z["frame_id"][i]))
            split.append(s)
            dref.append(float(z["d_ref"][i]))
            nfeas.append(int(n[i]))
            kind.append("stopped" if z["v0"][i] < 0.5 else "turn" if turn[i] >= 20 else "straight")
            exp["S"].append(float(z["exp_S"][i]))
            exp["P"].append(float(z["exp_P"][i]))
            exp["outside"].append(float(z["exp_outside"][i]))
            for k in T:
                exp[k].append(int(z["exp_" + k][i]))
split, kind, nT, dref, nfeas = np.array(split), np.array(kind), np.array(nT), np.array(dref), np.array(nfeas)
E = {k: np.array(v) for k, v in exp.items()}
if not os.path.exists(OUT):
    np.savez(
        OUT,
        frame_id=np.array(fid),
        offsets=np.array(off, np.int64),
        idx=np.concatenate(tix),
        n_teachers_uncapped=nT.astype(np.int32),
        d_ref=dref.astype(np.float32),
        split=split,
        kind=kind,
    )
R = dict(
    frames={s: int((split == s).sum()) for s in ("train", "heldout")},
    feasible_median={s: float(np.median(nfeas[split == s])) for s in ("train", "heldout")},
    no_feasible=int((nfeas == 0).sum()),
    expert_under_half_m={s: float((dref[split == s] < 0.5).mean()) for s in ("train", "heldout")},
)
R["expert"] = {}
for s in ("train", "heldout"):
    m = split == s
    low = m & (E["S"] < 0.95)
    r = dict(reach_095=float((E["S"][m] >= 0.95).mean()), mean=float(E["S"][m].mean()), below=int(low.sum()))
    # which factor holds it back: the smallest factor of the frame's grade
    fac = np.stack(
        [
            E["P"],
            0.5 ** E["n_ped"],
            0.6 ** E["n_veh"],
            0.65 ** E["n_static"],
            0.7 ** E["n_red"],
            0.8 ** E["n_stop"],
            1 - E["outside"],
        ],
        1,
    )
    names = ["progress", "pedestrian", "vehicle", "static", "red", "stop", "outside"]
    a = fac[low].argmin(1)
    r["smallest_factor"] = {names[i]: int((a == i).sum()) for i in range(7)}
    r["fires"] = {names[i]: int((fac[low, i] < 1).sum()) for i in range(7)}
    R["expert"][s] = r
R["supply"] = {
    s: {
        k: dict(
            n=int(((split == s) & (kind == k if k != "all" else True)).sum()),
            ge1=float((nT[(split == s) & (kind == k if k != "all" else True)] >= 1).mean()),
            ge4=float((nT[(split == s) & (kind == k if k != "all" else True)] >= 4).mean()),
            median=float(np.median(nT[(split == s) & (kind == k if k != "all" else True)])),
        )
        for k in ("all", "straight", "turn", "stopped")
    }
    for s in ("train", "heldout")
}
R["terms"] = {s: {k: fire[s][k] / fire[s]["paths"] for k in fire[s] if k != "paths"} for s in fire}
R["paths"] = {s: fire[s]["paths"] for s in fire}
sp_ = 0
for f in ("grade_v2_s0.log", "grade_v2_s1.log"):
    p = f"{D}/logs/{f}"
    m = re.findall(r"GRADE_DONE (\d+) (\d+) (\d+)", open(p).read()) if os.path.exists(p) else []
    if m:
        a, b, c = map(int, m[-1])
        sp_ += b / c
R["paths_per_s_two_shards"] = sp_
json.dump(R, open(RES, "w"), indent=1)
print(json.dumps(R, indent=1))
