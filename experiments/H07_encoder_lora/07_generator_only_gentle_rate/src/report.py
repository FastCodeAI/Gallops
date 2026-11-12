"""Tables, two plots and menu pictures from out/state.json, out/run_*.json and the saved dumps.  CPU only.

    ./run.sh src/report.py
"""

from __future__ import annotations

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import phase1 as P1
import rerun as R

OUT = R.OUT
RUNS = [f"w{w:g}" for w in R.WEIGHTS]
COL = {"w0": "tab:blue", "w0.3": "tab:orange", "w1": "tab:green"}


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def fmt(s, g):
    return f"{s[g]['m1b']['mean']:.3f} / {s[g]['m1b']['median']:.3f}"


def main_row(name, s):
    return (f"| {name} | {fmt(s, 'ALL')} | {fmt(s, 'SLOW')} | {fmt(s, 'FAST')} | {fmt(s, 'SLOW_TURNS')} "
            f"| {s['SLOW']['m2_lt2']['mean']:.2f} | {s['SLOW']['m3_variety']['mean']:.3f} | {s['ALL']['m5_drift']['mean']:.3f} "
            f"| {s['SLOW']['better_gt_10cm']} / {s['SLOW']['worse_gt_10cm']} | {s['ALL']['m6_pick_ade']['mean']:.3f} "
            f"| {s['CONT_m1b_mean']:.3f} |")


def main() -> None:
    st = json.loads((OUT / "state.json").read_text())
    runs = {r: json.loads((OUT / f"run_{r}.json").read_text()) for r in RUNS if (OUT / f"run_{r}.json").exists()}
    mom = torch.load(R.S1DATA / "moments.pt", weights_only=False)["selection"]
    human = {P1.key(m): m["future_poses"] for m in mom}
    sp = R.splits(mom)
    import evalsets as E
    cont = sorted((e["scene_id"], e["keyframe"]) for e in E.load_selection_features())
    start = load(R.DATA / "dumps" / "start.pkl")
    sper = P1.per_moment(start, human, start)
    p1 = load(R.S1DATA / "dumps" / "pass_01.pkl")
    p1per = P1.per_moment(p1, human, start)
    p1sum = R.summarise(p1per, sper, sp, cont)

    hdr = ("| Model | 1b ALL mean / median | 1b SLOW | 1b FAST | 1b SLOW TURNS | SLOW within 2 m | SLOW variety | "
           "menu drift | SLOW better / worse > 10 cm | (scorer pick ADE) | 1b on the 103 |")
    lines = [hdr, "|" + "---|" * 11, main_row("original model", st["original_summary"]),
             main_row("starting model", st["start_summary"]), main_row("phase 1 lr 1e-4, pass 1", p1sum)]
    best = {}
    for r, rec in runs.items():
        if rec.get("best") is None:
            lines.append(f"| run {r}: no check passed the turning check | " + "— |" * 10)
            continue
        c = next(c for c in rec["checks"] if c["pass"] == rec["best"])
        best[r] = c
        lines.append(main_row(f"run {r.replace('w', 'w = ')}, best (pass {c['pass']:.1f})", c["summary"]))
    tbl = ["\n".join(lines), ""]
    for r, rec in runs.items():
        tbl += [f"Run {r.replace('w', 'w = ')}: {rec['stopped']}; best pass {rec['best']}", "",
                "| pass | SLOW 1b | SLOW-TURN 1b | turn ok | FAST 1b | ALL 1b | SLOW var | drift | stay term |",
                "|---|---|---|---|---|---|---|---|---|"]
        for c in rec["checks"]:
            s = c["summary"]
            tbl.append(f"| {c['pass']:.1f} | {s['SLOW']['m1b']['mean']:.3f} | {s['SLOW_TURNS']['m1b']['mean']:.3f} "
                       f"| {'yes' if c['turning_ok'] else 'NO'} | {s['FAST']['m1b']['mean']:.2f} | {s['ALL']['m1b']['mean']:.3f} "
                       f"| {s['SLOW']['m3_variety']['mean']:.3f} | {s['ALL']['m5_drift']['mean']:.3f} "
                       f"| {c.get('stay_recent') if c.get('stay_recent') is None else round(c['stay_recent'], 3)} |")
        tbl.append("")
    (OUT / "tables.md").write_text("\n".join(tbl))
    print("\n".join(tbl))

    # plots
    ss = st["start_summary"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r, rec in runs.items():
        x = [c["summary"]["SLOW"]["m3_variety"]["mean"] for c in rec["checks"]]
        y = [c["summary"]["SLOW"]["m1b"]["mean"] for c in rec["checks"]]
        ax.plot(x, y, "o-", ms=3, c=COL[r], label=f"run {r.replace('w', 'w = ')}")
        if r in best:
            ax.plot(best[r]["summary"]["SLOW"]["m3_variety"]["mean"], best[r]["summary"]["SLOW"]["m1b"]["mean"],
                    "*", ms=14, c=COL[r])
    ax.plot(ss["SLOW"]["m3_variety"]["mean"], ss["SLOW"]["m1b"]["mean"], "ks", ms=8, label="starting model")
    ax.plot(p1sum["SLOW"]["m3_variety"]["mean"], p1sum["SLOW"]["m1b"]["mean"], "kx", ms=9, label="phase 1 lr 1e-4, pass 1")
    ax.set_xlabel("SLOW variety: mean distance between candidates (m)")
    ax.set_ylabel("SLOW 1b: closest candidate's 3 s endpoint error (m)")
    ax.set_title("Below 15 m/s (1,681 moments), one point per check; star = best", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "plot_slow_1b_vs_variety.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r, rec in runs.items():
        ax.plot([c["pass"] for c in rec["checks"]], [c["summary"]["SLOW_TURNS"]["m1b"]["mean"] for c in rec["checks"]],
                "o-", ms=3, c=COL[r], label=f"run {r.replace('w', 'w = ')}")
    ax.axhline(ss["SLOW_TURNS"]["m1b"]["mean"], ls="--", c="k", label="starting model")
    ax.set_xlabel("training progress (passes over the 13,937 moments)")
    ax.set_ylabel("SLOW-TURN 1b (m)")
    ax.set_title(f"Slow turns ({len(sp['SLOW_TURNS'])} moments): below the dashed line = turning check passes", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "plot_slow_turn_1b_by_progress.png", dpi=120); plt.close(fig)

    # pictures: start vs each run's best
    bd = {r: load(R.DATA / r / "dumps" / f"p{c['pass']:04.1f}.pkl") for r, c in best.items()}
    bper = {r: P1.per_moment(d, human, start) for r, d in bd.items()}

    def distinct(keys, n, skip=()):
        out, seen = [], {k[0] for k in skip}
        for k in keys:
            if k[0] not in seen:
                out.append(k); seen.add(k[0])
            if len(out) == n:
                break
        return out
    left = [k for k in human if k[0].startswith("87537280") and k[1] == 135]
    assert len(left) == 1, left
    turns_worst = distinct(sorted(sp["SLOW_TURNS"], key=lambda k: (-p1per[k]["m1b"], k)), 3, skip=left)
    fast = distinct(sorted(sp["FAST"], key=lambda k: (-sper[k]["m1b"], k)), 2)
    ref = min(best, key=lambda r: best[r]["summary"]["SLOW"]["m1b"]["mean"]) if best else None
    improved = distinct(sorted(sp["SLOW"], key=lambda k: (bper[ref][k]["m1b"] - sper[k]["m1b"], k)), 2) if ref else []
    groups = {"left_turn_and_worst_slow_turns": left + turns_worst, "fast": fast, "most_improved_slow": improved}
    cols = [("starting model", start, sper)] + [(f"run {r.replace('w', 'w = ')} best (pass {best[r]['pass']:.1f})", bd[r], bper[r])
                                                 for r in best]
    picked = {}
    for g, keys in groups.items():
        if not keys:
            continue
        fig, ax = plt.subplots(len(keys), len(cols), figsize=(3.6 * len(cols), 4.2 * len(keys)), squeeze=False)
        for i, k in enumerate(keys):
            h = np.asarray(human[k], dtype=np.float64)
            for j, (name, d, pm) in enumerate(cols):
                a = ax[i, j]
                p = np.asarray(d[k]["proposals"], dtype=np.float64)
                ade = np.linalg.norm(p[:, :6, :2] - h[None, :6, :2], axis=-1).mean(1)
                c = int(ade.argmin())
                for q in p:
                    a.plot(-np.r_[0, q[:, 1]], np.r_[0, q[:, 0]], c="0.6", lw=0.7)
                a.plot(-np.r_[0, p[c, :, 1]], np.r_[0, p[c, :, 0]], c="tab:red", lw=2, label="closest candidate")
                a.plot(-np.r_[0, h[:, 1]], np.r_[0, h[:, 0]], "k--", lw=1.8, label="human path")
                a.plot(-h[5, 1], h[5, 0], "ko", ms=4)
                half = max(6.0, 1.1 * float(np.abs(np.r_[p[..., 1].ravel(), h[:, 1]]).max()))
                a.set_xlim(-half, half); a.grid(alpha=0.3)
                a.set_title(f"{name}\n{k[0][:8]} t={k[1]}  1b {pm[k]['m1b']:.2f} m  var {pm[k]['m3_variety']:.2f}", fontsize=8)
                if i == len(keys) - 1:
                    a.set_xlabel("← left   sideways (m)   right →")
                if j == 0:
                    a.set_ylabel("forward (m)")
        ax[0, 0].legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(OUT / f"menus_{g}.png", dpi=100); plt.close(fig)
        picked[g] = [{"moment": list(k), "start_1b": sper[k]["m1b"], "phase1_pass1_1b": p1per[k]["m1b"],
                      **{f"{r}_best_1b": bper[r][k]["m1b"] for r in best}} for k in keys]
    (OUT / "pictures.json").write_text(json.dumps(picked, indent=1) + "\n")
    print(json.dumps(picked, indent=1))


if __name__ == "__main__":
    main()
