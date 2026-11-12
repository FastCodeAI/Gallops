"""Tables, plots and menu pictures for phase 2 (CPU only).

    ./run.sh src/p2report.py
"""

from __future__ import annotations

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import phase1 as P1
import phase2 as Q

OUT, DATA = Q.OUT, Q.DATA
RUNS = ("control", "phase2")
COL = {"control": "tab:blue", "phase2": "tab:red"}
LABEL = {"control": "control (generator only)", "phase2": "phase 2 (generator + LoRA)"}


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def f2(s, g):
    return f"{s[g]['m1b']['mean']:.3f} / {s[g]['m1b']['median']:.3f}"


def row(name, sc):
    s, t = sc["summary"], sc["tails"]
    t2 = lambda r: f"{t[f'T2_worst_increase_vs_{r}']['m']:.2f}"
    return (f"| {name} | {f2(s, 'ALL')} | {f2(s, 'SLOW')} | {f2(s, 'FAST')} | {f2(s, 'SLOW_TURNS')} "
            f"| {s['SLOW']['m2_lt2']['mean']:.2f} | {s['SLOW']['m3_variety']['mean']:.3f} | {s['ALL']['m5_drift']['mean']:.3f} "
            f"| {s['SLOW']['better_gt_10cm']} / {s['SLOW']['worse_gt_10cm']} | {s['ALL']['m6_pick_ade']['mean']:.3f} "
            f"| {s['CONT_m1b_mean']:.3f} | {t['T1_worse_gt_0.5m_vs_pass3_start']} / {t['T1_worse_gt_0.5m_vs_p9.5']} "
            f"| {t2('pass3_start')} / {t2('p9.5')} | {t['T3_worst10pct_mean_1b']:.3f} |")


def main() -> None:
    ctx = Q.Ctx()
    st = json.loads((OUT / "state.json").read_text())
    runs = {r: json.loads((OUT / f"run_{r}.json").read_text()) for r in RUNS if (OUT / f"run_{r}.json").exists()}
    orig = load(P1.DT.DATA / "dumps" / "original.pkl")
    base = {"original model": ctx.score(orig), "starting model (pass-3)": ctx.score(ctx.start),
            "phase 1 best (w = 0, pass 9.5)": ctx.score(ctx.p95)}
    best = {r: next(c for c in rec["checks"] if c["pass"] == rec["best"]) for r, rec in runs.items() if rec.get("best") is not None}

    L = ["| Model | 1b ALL mean / median | 1b SLOW | 1b FAST | 1b SLOW TURNS | SLOW within 2 m | SLOW variety | menu drift "
         "| SLOW better / worse > 10 cm vs pass-3 | (scorer pick ADE) | 1b on the 103 | T1 vs pass-3 / p9.5 | T2 (m) vs pass-3 / p9.5 | T3 (m) |",
         "|" + "---|" * 14]
    L += [row(n, sc) for n, sc in base.items()]
    for r, rec in runs.items():
        L.append(row(f"{LABEL[r]}, best (pass {rec['best']:.1f})", best[r]) if r in best
                 else f"| {LABEL[r]}: no check passed the turning check |" + " — |" * 13)
    out = ["## Main table", "", *L, ""]

    # matched: phase 2 minus control at each common check
    if len(runs) == 2:
        cc = {c["pass"]: c for c in runs["control"]["checks"]}
        out += ["## Matched (phase 2 minus control, same step)", "",
                "| pass | SLOW 1b | SLOW-TURN 1b | FAST 1b | SLOW variety | T1 vs p9.5 | T3 |", "|---|---|---|---|---|---|---|"]
        for c in runs["phase2"]["checks"]:
            if c["pass"] in cc and c["pass"] > 0:
                a, b = c, cc[c["pass"]]
                d = lambda g, k="m1b": a["summary"][g][k]["mean"] - b["summary"][g][k]["mean"]
                out.append(f"| {c['pass']:.1f} | {d('SLOW'):+.3f} | {d('SLOW_TURNS'):+.3f} | {d('FAST'):+.3f} "
                           f"| {d('SLOW', 'm3_variety'):+.3f} | {a['tails']['T1_worse_gt_0.5m_vs_p9.5'] - b['tails']['T1_worse_gt_0.5m_vs_p9.5']:+d} "
                           f"| {a['tails']['T3_worst10pct_mean_1b'] - b['tails']['T3_worst10pct_mean_1b']:+.3f} |")
        out.append("")

    # per-run checks
    for r, rec in runs.items():
        out += [f"## {LABEL[r]}: {rec['stopped']}; best pass {rec['best']}", "",
                "| pass | SLOW 1b | SLOW-TURN 1b | turn ok | FAST 1b | ALL 1b | SLOW var | T1 vs p9.5 | T2 vs p9.5 | T3 | "
                + ("encoder drift (cos) | game loss (picking pairs) |" if r == "phase2" else ""),
                "|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if r == "phase2" else "")]
        for c in rec["checks"]:
            s, t = c["summary"], c["tails"]
            x = (f"| {c['pass']:.1f} | {s['SLOW']['m1b']['mean']:.3f} | {s['SLOW_TURNS']['m1b']['mean']:.3f} "
                 f"| {'yes' if c['turning_ok'] else 'NO'} | {s['FAST']['m1b']['mean']:.2f} | {s['ALL']['m1b']['mean']:.3f} "
                 f"| {s['SLOW']['m3_variety']['mean']:.3f} | {t['T1_worse_gt_0.5m_vs_p9.5']} | {t['T2_worst_increase_vs_p9.5']['m']:.2f} "
                 f"| {t['T3_worst10pct_mean_1b']:.3f} |")
            if r == "phase2":
                x += f" {c['encoder_drift_cos']:.5f} | {c['game_loss_picking_pairs']:.5f} |"
            out.append(x)
        out.append("")

    # backfill for the three phase-1 rerun runs
    out += ["## Phase-1 rerun, backfilled tails", "",
            "| run | pass | SLOW 1b | T1 vs pass-3 / p9.5 | T2 vs pass-3 (m, moment) | T3 |", "|---|---|---|---|---|---|"]
    back = {}
    for w in ("w0", "w0.3", "w1"):
        rec = json.loads((Q.R.OUT / f"run_{w}.json").read_text())
        for c in rec["checks"]:
            dump = ctx.start if c["pass"] == 0 else load(Q.S2DATA / w / "dumps" / f"p{c['pass']:04.1f}.pkl")
            sc = ctx.score(dump)
            t = sc["tails"]
            back.setdefault(w, []).append({"pass": c["pass"], **t, "SLOW_1b": sc["summary"]["SLOW"]["m1b"]["mean"]})
            if c["pass"] in (0.0, 1.0, 3.0, 5.0, 7.0, 10.0) or c["pass"] == rec["best"]:
                out.append(f"| {w} | {c['pass']:.1f}{' ★' if c['pass'] == rec['best'] else ''} | {sc['summary']['SLOW']['m1b']['mean']:.3f} "
                           f"| {t['T1_worse_gt_0.5m_vs_pass3_start']} / {t['T1_worse_gt_0.5m_vs_p9.5']} "
                           f"| {t['T2_worst_increase_vs_pass3_start']['m']:.2f} ({t['T2_worst_increase_vs_pass3_start']['moment'][0][:8]} "
                           f"t={t['T2_worst_increase_vs_pass3_start']['moment'][1]}) | {t['T3_worst10pct_mean_1b']:.3f} |")
    (OUT / "backfill_s2_tails.json").write_text(json.dumps(back, indent=1) + "\n")
    out.append("")
    (OUT / "tables.md").write_text("\n".join(out))
    print("\n".join(out))

    # plots
    p95s = base["phase 1 best (w = 0, pass 9.5)"]["summary"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for r, rec in runs.items():
        xs = [c["pass"] for c in rec["checks"]]
        ax[0].plot(xs, [c["summary"]["SLOW"]["m1b"]["mean"] for c in rec["checks"]], "o-", ms=3, c=COL[r], label=LABEL[r])
        ax[1].plot(xs, [c["summary"]["SLOW_TURNS"]["m1b"]["mean"] for c in rec["checks"]], "o-", ms=3, c=COL[r], label=LABEL[r])
    ax[0].axhline(p95s["SLOW"]["m1b"]["mean"], ls="--", c="k", label="phase 1 best (p9.5)")
    ax[1].axhline(p95s["SLOW_TURNS"]["m1b"]["mean"], ls="--", c="k", label="phase 1 best (p9.5)")
    ax[0].set_title("SLOW 1b (1,681 moments below 15 m/s)", fontsize=10)
    ax[1].set_title("SLOW-TURN 1b (342 moments)", fontsize=10)
    for a in ax:
        a.set_xlabel("passes over the 13,937 training moments"); a.set_ylabel("m"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "plot_slow_1b_by_progress.png", dpi=120); plt.close(fig)

    if len(runs) == 2:
        cc = {c["pass"]: c for c in runs["control"]["checks"]}
        pts = [(c, cc[c["pass"]]) for c in runs["phase2"]["checks"] if c["pass"] in cc]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, gs in zip(axs, ((("SLOW", "tab:green"), ("SLOW_TURNS", "tab:purple")), (("FAST", "tab:gray"),))):
            for g, col in gs:
                ys = [a["summary"][g]["m1b"]["mean"] - b["summary"][g]["m1b"]["mean"] for a, b in pts]
                ax.plot([a["pass"] for a, _ in pts], ys, "o-", ms=3, c=col, label=f"{g.replace('_', ' ')} 1b")
            ax.axhline(0, c="k", lw=0.8)
            ax.set_xlabel("passes"); ax.set_ylabel("phase 2 minus control (m); below 0 = phase 2 better")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("Phase 2 minus control at the same step", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / "plot_phase2_minus_control.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r, rec in runs.items():
        ax.plot([c["summary"]["SLOW"]["m3_variety"]["mean"] for c in rec["checks"]],
                [c["summary"]["SLOW"]["m1b"]["mean"] for c in rec["checks"]], "o-", ms=3, c=COL[r], label=LABEL[r])
        if r in best:
            ax.plot(best[r]["summary"]["SLOW"]["m3_variety"]["mean"], best[r]["summary"]["SLOW"]["m1b"]["mean"], "*", ms=14, c=COL[r])
    ax.plot(p95s["SLOW"]["m3_variety"]["mean"], p95s["SLOW"]["m1b"]["mean"], "ks", ms=8, label="phase 1 best (p9.5) = step 0")
    ax.set_xlabel("SLOW variety (m)"); ax.set_ylabel("SLOW 1b (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Below 15 m/s, one point per check; star = best", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "plot_slow_1b_vs_variety.png", dpi=120); plt.close(fig)

    # pictures: p9.5 vs control best vs phase 2 best
    if len(best) < 2:
        return
    bd = {r: load(DATA / r / "dumps" / f"p{best[r]['pass']:04.1f}.pkl") for r in best}
    bp = {r: P1.per_moment(d, ctx.human, ctx.start) for r, d in bd.items()}
    p95p = ctx.p95_per

    def distinct(keys, n, skip=()):
        res, seen = [], {k[0] for k in skip}
        for k in keys:
            if k[0] not in seen:
                res.append(k); seen.add(k[0])
            if len(res) == n:
                break
        return res
    left = [k for k in ctx.human if k[0].startswith("87537280") and k[1] == 135]
    t2 = distinct(sorted(ctx.sp["SLOW_TURNS"], key=lambda k: (-(bp["phase2"][k]["m1b"] - p95p[k]["m1b"]), k)), 3, skip=left)
    fast = distinct(sorted(ctx.sp["FAST"], key=lambda k: (-ctx.start_per[k]["m1b"], k)), 2)
    imp = distinct(sorted(ctx.sp["SLOW"], key=lambda k: (bp["phase2"][k]["m1b"] - p95p[k]["m1b"], k)), 2)
    groups = {"left_turn_and_largest_slow_turn_increases": left + t2, "fast": fast, "most_improved_slow": imp}
    cols = [("phase 1 best (p9.5)", ctx.p95, p95p)] + [(f"{LABEL[r]} best (pass {best[r]['pass']:.1f})", bd[r], bp[r]) for r in RUNS]
    picked = {}
    for g, keys in groups.items():
        fig, ax = plt.subplots(len(keys), 3, figsize=(11, 4.2 * len(keys)), squeeze=False)
        for i, k in enumerate(keys):
            h = np.asarray(ctx.human[k], dtype=np.float64)
            for j, (name, d, pm) in enumerate(cols):
                a = ax[i, j]
                p = np.asarray(d[k]["proposals"], dtype=np.float64)
                c = int(np.linalg.norm(p[:, :6, :2] - h[None, :6, :2], axis=-1).mean(1).argmin())
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
        picked[g] = [{"moment": list(k), "p9.5": p95p[k]["m1b"], **{r: bp[r][k]["m1b"] for r in RUNS}} for k in keys]
    (OUT / "pictures.json").write_text(json.dumps(picked, indent=1) + "\n")
    print(json.dumps(picked, indent=1))


if __name__ == "__main__":
    main()
