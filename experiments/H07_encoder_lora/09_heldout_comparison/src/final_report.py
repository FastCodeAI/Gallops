"""Tables, paired bootstrap and menu pictures for the final comparison (CPU only, no model).

    ./run.sh src/final_report.py

Reads the saved held-out results (final.py) and the saved picking-moment dumps of the earlier
stages.  Menu metrics are s1 phase1.per_moment unchanged (waypoints 1-6 = 0.5-3 s, x/y):
  a = m1a closest candidate ADE (3 s), b = m1b that candidate's FDE (3 s),
  c = m2_lt2 candidates within 2 m, d = m3_variety.
The scorer's pick is the wrapper's displacement_errors at 3 s (ade, fde@1s, fde@2s, fde@3s).
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import final as FN
import kitdata as DT
import phase1 as P1

OUT, DATA, NAMES = FN.OUT, FN.DATA, FN.NAMES
SLOW_V, TURN_RAD, FEW, N_BOOT = 15.0, 0.3, 20, 10_000
PAIRS = [(2, 1), (3, 2), (4, 3)]
GROUPS = ("ALL", "SLOW", "FAST", "SLOW_TURNS")
GLABEL = {"ALL": "ALL", "SLOW": "SLOW", "FAST": "FAST", "SLOW_TURNS": "SLOW TURNS"}


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def groups(moments) -> dict:
    k = lambda m: (m["scene_id"], m["keyframe_index"])
    slow = [k(m) for m in moments if m["speed_mps"] < SLOW_V]
    turn = {k(m) for m in moments if abs(m["heading_3s_rad"]) > TURN_RAD}
    return {"ALL": sorted(k(m) for m in moments), "SLOW": sorted(slow),
            "FAST": sorted(k(m) for m in moments if m["speed_mps"] >= SLOW_V),
            "SLOW_TURNS": sorted(x for x in slow if x in turn)}


def boot(diff: dict, n: int = N_BOOT) -> list:
    """95 % paired bootstrap of the moment-weighted mean, resampling whole scenes (seed 0)."""
    if not diff:
        return [float("nan"), float("nan")]
    scenes = sorted({k[0] for k in diff})
    s = np.array([sum(v for k, v in diff.items() if k[0] == sc) for sc in scenes])
    c = np.array([sum(1 for k in diff if k[0] == sc) for sc in scenes])
    idx = np.random.default_rng(0).integers(0, len(scenes), size=(n, len(scenes)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def tails(per: dict, ref: dict, turns: list) -> dict:
    if not turns:
        return {"worse_gt_0.5m": 0, "largest_increase_m": float("nan"), "largest_increase_moment": ["none", -1],
                "worst10pct_mean_b": float("nan"), "worst10pct_n": 0}
    n10 = max(1, math.ceil(0.1 * len(turns)))
    d = {k: per[k]["m1b"] - ref[k]["m1b"] for k in turns}
    kmax = max(d, key=lambda k: (d[k], k))
    return {"worse_gt_0.5m": int(sum(v > 0.5 for v in d.values())), "largest_increase_m": float(d[kmax]),
            "largest_increase_moment": list(kmax),
            "worst10pct_mean_b": float(np.mean(sorted((per[k]["m1b"] for k in turns), reverse=True)[:n10])),
            "worst10pct_n": n10}


def main() -> None:
    global OUT
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":      # wiring test on the 2-picking-scene smoke output
        res = load(DATA / "smoke_picking2.pkl")
        OUT = Path(sys.argv[2])
    else:
        res = load(DATA / "heldout223.pkl")
        state = json.loads((OUT / "state.json").read_text())
        assert state["G1"]["pass"] and state["G2"]["pass"]
    human = {(m["scene_id"], m["keyframe_index"]): m["human_poses"] for m in res["moments"]}
    gp = groups(res["moments"])
    per = {i: P1.per_moment(res["dumps"][i], human, None) for i in NAMES}
    pick = {i: {(r["scene_id"], r["keyframe_index"]): r for r in res["rows"][i]} for i in NAMES}
    for i in NAMES:
        for k in per[i]:
            per[i][k].update({"pick_" + x: pick[i][k][x] for x in ("ade", "fde@1s", "fde@2s", "fde@3s")})
    mean = lambda i, g, x: float(np.mean([per[i][k][x] for k in gp[g]]))
    med = lambda i, g, x: float(np.median([per[i][k][x] for k in gp[g]]))
    n = {g: len(gp[g]) for g in GROUPS}
    few = {g: n[g] < FEW for g in GROUPS}
    tag = lambda g: f"{GLABEL[g]} (n = {n[g]}{', too few to read' if few[g] else ''})"

    out = {"n": n, "too_few": few, "models": {}}
    md = ["## Main table (223 held-out moments; mean, with median in brackets for ALL)", "",
          f"Groups: {', '.join(tag(g) for g in GROUPS)}.", "",
          "| Model | ALL: closest candidate ADE (3 s) | ALL: closest candidate FDE (3 s) | ALL: candidates within 2 m | ALL: variety | "
          "SLOW: closest candidate ADE (3 s) / FDE (3 s) | FAST: closest candidate ADE (3 s) / FDE (3 s) | "
          "SLOW TURNS: closest candidate ADE (3 s) / FDE (3 s) | scorer's pick ADE (3 s) | scorer's pick FDE (1 s) | "
          "scorer's pick FDE (2 s) | scorer's pick FDE (3 s) | SLOW TURNS tails vs Encoder only, tuned on KIT: worse > 0.5 m on FDE · largest increase (moment) · mean FDE of worst 10 % |",
          "|---|" + "---|" * 12]
    for i, name in NAMES.items():
        r = {g: {x: {"mean": mean(i, g, x), "median": med(i, g, x)} for x in
                 ("m1a", "m1b", "m2_lt2", "m3_variety", "pick_ade", "pick_fde@1s", "pick_fde@2s", "pick_fde@3s")} for g in GROUPS}
        t = tails(per[i], per[2], gp["SLOW_TURNS"])
        out["models"][name] = {"groups": r, "tails_vs_encoder_only": t}
        A = r["ALL"]
        ab = lambda g: f"{r[g]['m1a']['mean']:.3f} / {r[g]['m1b']['mean']:.3f}" + (" ‡" if few[g] else "")
        tl = "—" if i == 2 else (f"{t['worse_gt_0.5m']} · {t['largest_increase_m']:+.2f} m "
                                 f"(`{t['largest_increase_moment'][0][:8]}` t = {t['largest_increase_moment'][1]}) · ")
        tl += f"{t['worst10pct_mean_b']:.2f}" if i != 2 else f" · {t['worst10pct_mean_b']:.2f}"
        md.append(f"| {name} | {A['m1a']['mean']:.3f} ({A['m1a']['median']:.3f}) | {A['m1b']['mean']:.3f} ({A['m1b']['median']:.3f}) "
                  f"| {A['m2_lt2']['mean']:.1f} | {A['m3_variety']['mean']:.2f} | {ab('SLOW')} | {ab('FAST')} | {ab('SLOW_TURNS')} "
                  f"| {A['pick_ade']['mean']:.3f} ({A['pick_ade']['median']:.3f}) | {A['pick_fde@1s']['mean']:.3f} "
                  f"| {A['pick_fde@2s']['mean']:.3f} | {A['pick_fde@3s']['mean']:.3f} | {tl} |")
    md += ["", "‡ too few to read (fewer than 20 moments).", ""]

    # paired differences
    md += ["## Paired differences (second model minus first; negative = the newer model is better)", "",
           "95 % interval: paired bootstrap, 10,000 resamples of whole scenes, seed 0.", "",
           "| Pair | Metric | Group | n | mean (m) | median (m) | better / worse by > 10 cm | 95 % interval (m) |",
           "|---|---|---|---|---|---|---|---|"]
    out["paired"] = []
    for new, old in PAIRS:
        for x, xl in (("m1a", "closest candidate ADE (3 s)"), ("pick_ade", "scorer's pick ADE (3 s)")):
            for g in GROUPS:
                d = {k: per[new][k][x] - per[old][k][x] for k in gp[g]}
                v = np.array(list(d.values()))
                if not len(v):
                    continue
                ci = boot(d)
                rec = {"new": NAMES[new], "old": NAMES[old], "metric": xl, "group": g, "n": len(v),
                       "mean": float(v.mean()), "median": float(np.median(v)),
                       "better_gt_10cm": int((v < -0.1).sum()), "worse_gt_10cm": int((v > 0.1).sum()), "ci95": ci,
                       "too_few": few[g]}
                out["paired"].append(rec)
                md.append(f"| {NAMES[new]} vs {NAMES[old]} | {xl} | {GLABEL[g]}{' ‡' if few[g] else ''} | {len(v)} | {rec['mean']:+.3f} "
                          f"| {rec['median']:+.3f} | {rec['better_gt_10cm']} / {rec['worse_gt_10cm']} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    md.append("")

    # picking-moment vs held-out
    mom = torch.load(DT.DATA / "moments.pt", weights_only=False)["selection"]
    ph = {P1.key(m): m["future_poses"] for m in mom}
    pspeed = {P1.key(m): float(np.linalg.norm(np.asarray(m["ego_status"])[-1, 3:5])) for m in mom}
    pick_dumps = {2: load(FN.P95_DUMP.parent.parent.parent / "dumps" / "start.pkl"), 3: load(FN.P95_DUMP), 4: load(FN.P10_DUMP)}
    pg = {"ALL": sorted(ph), "SLOW": sorted(k for k in ph if pspeed[k] < SLOW_V)}
    md += ["## Picking moments vs held-out moments (mean, m)", "",
           "Picking moments: the 1,763 moments of the 50 picking scenes the checkpoints were chosen on. "
           "Held-out: the 223 moments above.", "",
           "| Model | Group | closest candidate ADE (3 s): picking → held-out | closest candidate FDE (3 s): picking → held-out |",
           "|---|---|---|---|"]
    out["picking_vs_heldout"] = {}
    pp = {i: P1.per_moment(d, ph, None) for i, d in pick_dumps.items()}
    for i in (2, 3, 4):
        for g in ("ALL", "SLOW"):
            pa, pb = (float(np.mean([pp[i][k][x] for k in pg[g]])) for x in ("m1a", "m1b"))
            ha, hb = mean(i, g, "m1a"), mean(i, g, "m1b")
            out["picking_vs_heldout"][f"{NAMES[i]} {g}"] = {"picking": [pa, pb], "heldout": [ha, hb], "n_picking": len(pg[g])}
            md.append(f"| {NAMES[i]}{' (reference)' if i == 2 else ''} | {GLABEL[g]} | {pa:.3f} → {ha:.3f} | {pb:.3f} → {hb:.3f} |")
    for i in (3, 4):
        for g in ("ALL", "SLOW"):
            gpick = [float(np.mean([pp[i][k][x] - pp[2][k][x] for k in pg[g]])) for x in ("m1a", "m1b")]
            ghold = [float(np.mean([per[i][k][x] - per[2][k][x] for k in gp[g]])) for x in ("m1a", "m1b")]
            out["picking_vs_heldout"][f"gain {NAMES[i]} over {NAMES[2]} {g}"] = {"picking": gpick, "heldout": ghold}
            md.append(f"| gain of {NAMES[i]} over {NAMES[2]} | {GLABEL[g]} | {gpick[0]:+.3f} → {ghold[0]:+.3f} | {gpick[1]:+.3f} → {ghold[1]:+.3f} |")
    md.append("")

    # pictures
    d32 = {k: per[3][k]["m1a"] - per[2][k]["m1a"] for k in gp["ALL"]}
    order = sorted(d32, key=lambda k: (d32[k], k))
    sets = {"most_improved_generator_vs_encoder": order[:3], "most_worsened_generator_vs_encoder": order[::-1][:3]}
    fast = gp["FAST"]
    sets["fast"] = fast if len(fast) <= 6 else sorted(fast, key=lambda k: (-per[1][k]["m1a"], k))[:3]
    picked = {}
    for name, keys in sets.items():
        if not keys:
            continue
        fig, ax = plt.subplots(len(keys), 4, figsize=(15, 4.0 * len(keys)), squeeze=False)
        for r_, k in enumerate(keys):
            h = np.asarray(human[k], dtype=np.float64)
            for c_, i in enumerate(NAMES):
                a = ax[r_, c_]
                p = np.asarray(res["dumps"][i][k]["proposals"], dtype=np.float64)
                cl = int(np.linalg.norm(p[:, :6, :2] - h[None, :6, :2], axis=-1).mean(1).argmin())
                ch = int(res["dumps"][i][k]["chosen"])
                for q in p:
                    a.plot(-np.r_[0, q[:, 1]], np.r_[0, q[:, 0]], c="0.65", lw=0.7)
                a.plot(-np.r_[0, p[ch, :, 1]], np.r_[0, p[ch, :, 0]], c="tab:blue", lw=1.4, ls=":", label="scorer's pick")
                a.plot(-np.r_[0, p[cl, :, 1]], np.r_[0, p[cl, :, 0]], c="tab:red", lw=2, label="closest candidate")
                a.plot(-np.r_[0, h[:, 1]], np.r_[0, h[:, 0]], "k--", lw=1.8, label="human path")
                a.plot(-h[5, 1], h[5, 0], "ko", ms=4)
                half = max(6.0, 1.1 * float(np.abs(np.r_[p[..., 1].ravel(), h[:, 1]]).max()))
                a.set_xlim(-half, half); a.grid(alpha=0.3)
                pm = per[i][k]
                a.set_title(f"{NAMES[i]}\n{k[0][:8]} t={k[1]}  closest ADE {pm['m1a']:.2f} / FDE {pm['m1b']:.2f} m\n"
                            f"pick ADE {pm['pick_ade']:.2f} m  variety {pm['m3_variety']:.2f}", fontsize=8)
                if r_ == len(keys) - 1:
                    a.set_xlabel("← left   sideways (m)   right →")
                if c_ == 0:
                    a.set_ylabel("forward (m)")
        ax[0, 0].legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(OUT / f"menus_{name}.png", dpi=95); plt.close(fig)
        picked[name] = [{"moment": list(k), "speed_mps": next(m["speed_mps"] for m in res["moments"]
                                                              if (m["scene_id"], m["keyframe_index"]) == k),
                         **{NAMES[i]: {"a": per[i][k]["m1a"], "b": per[i][k]["m1b"], "pick_ade": per[i][k]["pick_ade"]}
                            for i in NAMES}} for k in keys]
    out["pictures"] = picked
    (OUT / "final_tables.json").write_text(json.dumps(out, indent=1) + "\n")
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps(picked, indent=1))


if __name__ == "__main__":
    main()
