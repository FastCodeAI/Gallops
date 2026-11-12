"""Tables, metric plot and menu pictures from out/passes.json and the saved dumps.  CPU only.

    ./run.sh src/report.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "out"
DATA = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s1")
DUMPS = DATA / "dumps"

import phase1 as P1  # noqa: E402  (per_moment / aggregate; importing builds nothing)


def row(name, a, c=None):
    f = lambda k: f"{a[k]['mean']:.3f} / {a[k]['median']:.3f}"
    cont = f"{c['m1a']['mean']:.3f} / {c['m1b']['mean']:.3f}" if c else "—"
    drift = f(("m5_drift")) if "m5_drift" in a else "—"
    return (f"| {name} | {f('m1a')} | **{f('m1b')}** | {f('min_fde3')} | {a['m2_lt1']['mean']:.2f} / {a['m2_lt2']['mean']:.2f} "
            f"| {f('m3_variety')} | {a['m4_worst20_m1a']['mean']:.3f} / {a['m4_worst20_m1b']['mean']:.3f} | {drift} "
            f"| {f('m6_pick_ade')} | {cont} |")


def main() -> None:
    st = json.loads((OUT / "passes.json").read_text())
    g1 = st["G1"]
    lines = ["| Model | 1a closest ADE | 1b its FDE@3s | lowest FDE@3s | 2 count <1 m / <2 m (mean) | 3 variety | "
             "4 worst-20 % 1a / 1b (mean) | 5 drift | 6 pick ADE | continuity 1a / 1b (mean) |",
             "|---|---|---|---|---|---|---|---|---|---|",
             row("original", g1["original_main"], g1["original_cont"]),
             row("start (pass 3)", g1["start_main"], g1["start_cont"])]
    for p in st["passes"]:
        star = " ★" if p["pass"] == st.get("best_pass") else ""
        lines.append(row(f"pass {p['pass']}{star}", p["main"], p["cont"]))
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # plot
    ps = [p["pass"] for p in st["passes"]]
    s = g1["start_main"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for a, (k, t) in zip(ax.flat, [("m1b", "1b: closest candidate's 3 s endpoint error (m)"),
                                   ("m2_lt2", "2: candidates within 2 m (count of 32)"),
                                   ("m3_variety", "3: variety, mean pairwise distance (m)"),
                                   ("m5_drift", "5: drift from the starting menu (m)")]):
        a.plot(ps, [p["main"][k]["mean"] for p in st["passes"]], "o-", label="trained generator")
        a.axhline(s[k]["mean"], ls="--", c="k", label="starting model")
        a.set_title(t, fontsize=10); a.set_xlabel("pass"); a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Generator on KIT, picking set (1,763 moments): mean per pass")
    fig.tight_layout(); fig.savefig(OUT / "metrics_by_pass.png", dpi=120); plt.close(fig)

    # menu pictures: start vs best pass
    best = st["best_pass"]
    mom = torch.load(DATA / "moments.pt", weights_only=False)["selection"]
    human = {(m["scene_id"], m["t"]): m["future_poses"] for m in mom}
    with open(DUMPS / "start.pkl", "rb") as fh:
        start = pickle.load(fh)
    with open(DUMPS / f"pass_{best:02d}.pkl", "rb") as fh:
        bestd = pickle.load(fh)
    ps_ = P1.per_moment(start, human, None)
    pb_ = P1.per_moment(bestd, human, None)
    delta = {k: pb_[k]["m1b"] - ps_[k]["m1b"] for k in ps_}
    order = sorted(delta, key=lambda k: (delta[k], k))
    worst = [tuple(k) for k in json.loads((OUT / "worst20_main.json").read_text())["moments"]]

    def distinct(keys, n=3):   # one moment per scene, so three pictures show three places
        out, seen = [], set()
        for k in keys:
            if k[0] not in seen:
                out.append(k); seen.add(k[0])
            if len(out) == n:
                return out
        return out
    groups = {"most_improved": distinct(order), "most_worsened": distinct(order[::-1]), "worst20_examples": distinct(worst)}
    sel = {}
    for g, keys in groups.items():
        fig, ax = plt.subplots(len(keys), 2, figsize=(9, 4.6 * len(keys)))
        for r, k in enumerate(keys):
            for c, (name, d, pm) in enumerate([("start (pass 3)", start, ps_), (f"pass {best}", bestd, pb_)]):
                a = ax[r, c]
                p = np.asarray(d[k]["proposals"], dtype=np.float64)
                h = np.asarray(human[k], dtype=np.float64)
                ade = np.linalg.norm(p[:, :6, :2] - h[None, :6, :2], axis=-1).mean(1)
                i = int(ade.argmin())
                for q in p:
                    a.plot(-np.r_[0, q[:, 1]], np.r_[0, q[:, 0]], c="0.6", lw=0.7)
                a.plot(-np.r_[0, p[i, :, 1]], np.r_[0, p[i, :, 0]], c="tab:red", lw=2, label="closest candidate")
                a.plot(-np.r_[0, h[:, 1]], np.r_[0, h[:, 0]], "k--", lw=1.8, label="human path")
                a.plot(-h[5, 1], h[5, 0], "ko", ms=4)
                half = max(6.0, 1.1 * float(np.abs(np.r_[p[..., 1].ravel(), h[:, 1]]).max()))
                a.set_xlim(-half, half); a.grid(alpha=0.3)
                a.set_title(f"{name}\n{k[0][:8]} t={k[1]}  1a {pm[k]['m1a']:.2f}  1b {pm[k]['m1b']:.2f} m", fontsize=9)
                a.set_xlabel("← left   sideways (m)   right →"); a.set_ylabel("forward (m)")
        ax[0, 0].legend(fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(OUT / f"menus_{g}.png", dpi=110); plt.close(fig)
        sel[g] = [{"moment": list(k), "start_1b": ps_[k]["m1b"], "best_1b": pb_[k]["m1b"]} for k in keys]

    # where did 1b change (best vs start), by moment group
    w = set(worst)
    d = np.array([delta[k] for k in sorted(delta)])
    dw = np.array([delta[k] for k in sorted(delta) if k in w])
    dn = np.array([delta[k] for k in sorted(delta) if k not in w])
    speed = {}
    for m in mom:
        v = float(np.linalg.norm(m["ego_status"][-1, 3:5]))
        speed[(m["scene_id"], m["t"])] = v
    bins = [(0, 0.5, "stopped <0.5"), (0.5, 5, "0.5–5"), (5, 10, "5–10"), (10, 15, "10–15"), (15, 99, ">=15")]
    by_speed = {lab: {"n": int(sum(lo <= speed[k] < hi for k in delta)),
                      "mean_delta_1b": float(np.mean([delta[k] for k in delta if lo <= speed[k] < hi] or [np.nan])),
                      "start_1b": float(np.mean([ps_[k]["m1b"] for k in delta if lo <= speed[k] < hi] or [np.nan]))}
                for lo, hi, lab in bins}
    cmd = {}
    for m in mom:
        c = ["left", "straight", "right", "unknown"][int(np.argmax(m["command"]))]
        cmd.setdefault(c, []).append(delta[(m["scene_id"], m["t"])])
    summ = {"best_pass": best, "delta_1b_all": {"mean": float(d.mean()), "median": float(np.median(d)),
                                                "better_gt_10cm": int((d < -0.1).sum()), "worse_gt_10cm": int((d > 0.1).sum())},
            "delta_1b_worst20": {"n": len(dw), "mean": float(dw.mean()), "median": float(np.median(dw)),
                                 "better_gt_10cm": int((dw < -0.1).sum()), "worse_gt_10cm": int((dw > 0.1).sum())},
            "delta_1b_rest": {"n": len(dn), "mean": float(dn.mean()), "median": float(np.median(dn))},
            "by_speed_mps": by_speed,
            "by_command": {c: {"n": len(v), "mean_delta_1b": float(np.mean(v))} for c, v in cmd.items()},
            "pictures": sel}
    # turning vs straight human paths (heading change at 4 s), and the high-speed share
    turn = {k: abs(float(np.asarray(human[k])[-1, 2])) > 0.3 for k in delta}
    pv = {k: (ps_[k]["m3_variety"], pb_[k]["m3_variety"]) for k in delta}
    def grp(f):
        ks = [k for k in delta if f(k)]
        return {"n": len(ks), "start_1b_mean": float(np.mean([ps_[k]["m1b"] for k in ks])),
                "best_1b_mean": float(np.mean([pb_[k]["m1b"] for k in ks])),
                "delta_1b_median": float(np.median([delta[k] for k in ks])),
                "better_gt_10cm": int(sum(delta[k] < -0.1 for k in ks)), "worse_gt_10cm": int(sum(delta[k] > 0.1 for k in ks)),
                "variety_start_best": [float(np.mean([pv[k][0] for k in ks])), float(np.mean([pv[k][1] for k in ks]))]}
    summ["by_path_shape"] = {"turning (|heading| > 0.3 rad at 4 s)": grp(lambda k: turn[k]),
                             "straight-ish": grp(lambda k: not turn[k]),
                             "below 15 m/s": grp(lambda k: speed[k] < 15), "15 m/s and above": grp(lambda k: speed[k] >= 15)}
    (OUT / "best_vs_start.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summ.items() if k != "pictures"}, indent=1))


if __name__ == "__main__":
    main()
