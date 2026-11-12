"""Tables and plots for the all-three run (CPU only, no model).

    ./run.sh src/report.py            # checks table, plots, worked example; the 223 tables once final5.py has run

Picking-moment metrics come from run.json (allthree.py).  Held-out metrics: s1 phase1.per_moment for
the menu (closest candidate ADE / FDE (3 s), within 2 m, variety) and the wrapper's displacement_errors
rows for the scorer's pick, exactly as 09_heldout_comparison final_report.py.  References on the 223 come from
s4's per-moment rows (heldout223.pkl) and are checked against 09_heldout_comparison/out/final_tables.json.
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

import final_report as FR   # 09_heldout_comparison (groups, scene bootstrap, tails)
import kitdata as DT        # 06_generator_only_fast_rate
import phase1 as P1         # 06_generator_only_fast_rate
import rerun as R           # 07_generator_only_gentle_rate

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "out"
GJK = HERE.parent
DATA = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s5")
S4 = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s4")
STEPS_PER_PASS = 435
NAMES = {1: "Released model", 2: "Encoder only, tuned on KIT", 3: "Generator only, trained on KIT",
         4: "Generator and encoder trained together", 5: "Encoder, generator and scorer trained together"}
GL = {"ALL": "ALL", "SLOW": "SLOW", "FAST": "FAST", "SLOW_TURNS": "SLOW TURNS"}
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8a8a85", "#2b2b29"


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- #
# Picking moments: checks
# --------------------------------------------------------------------------- #


def checks_table(run: dict) -> list:
    md = ["## Every check (1,763 picking moments; means in metres; this run's pick rule = upstream's)", "",
          "Hard checks: turning = SLOW TURNS scorer pick ADE (3 s) not worse than step 0; menu = SLOW closest "
          "candidate ADE (3 s) not worse than step 0.  FLAG = worse than \"Generator and encoder trained together\" "
          "on the same moments (its SLOW TURNS ADEs were not recorded, so only SLOW is compared).", "",
          "| pass | SLOW pick ADE | SLOW closest ADE | SLOW pick gap | SLOW TURNS pick ADE | SLOW TURNS closest ADE | "
          "ALL pick ADE | ALL closest ADE | FAST pick ADE | SLOW within 2 m | SLOW variety | turning | menu | flags (SLOW closest / pick) |",
          "|---|" + "---|" * 13]
    for c in run["checks"]:
        s = c["summary"]
        fl = c["flags"]
        md.append(f"| {c['pass']:.1f} | {s['SLOW']['m6_pick_ade']:.3f} | {s['SLOW']['m1a']:.3f} | {s['SLOW']['pick_gap']:.3f} "
                  f"| {s['SLOW_TURNS']['m6_pick_ade']:.3f} | {s['SLOW_TURNS']['m1a']:.3f} | {s['ALL']['m6_pick_ade']:.3f} "
                  f"| {s['ALL']['m1a']:.3f} | {s['FAST']['m6_pick_ade']:.3f} | {s['SLOW']['m2_lt2']:.1f} | {s['SLOW']['m3_variety']:.2f} "
                  f"| {'ok' if c['turning_ok'] else 'FAIL'} | {'ok' if c['menu_ok'] else 'FAIL'} "
                  f"| {fl['SLOW.m1a']} / {fl['SLOW.m6_pick_ade']} |")
    ref = run["reference_joint_row"]["SLOW"]
    md += ["", f"Reference \"Generator and encoder trained together\" (chosen check, same 1,763): SLOW closest ADE "
               f"{ref['m1a']:.3f}, SLOW pick ADE {ref['m6_pick_ade']:.3f}.", ""]
    md += ["## Every check: full group detail", "",
           "| pass | group | closest ADE (3 s) | closest FDE (3 s) | pick ADE (3 s) | pick FDE 1 s / 2 s / 3 s | pick gap | within 2 m | variety |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in run["checks"]:
        for g in ("ALL", "SLOW", "FAST", "SLOW_TURNS"):
            s = c["summary"][g]
            md.append(f"| {c['pass']:.1f} | {GL[g]} | {s['m1a']:.3f} | {s['m1b']:.3f} | {s['m6_pick_ade']:.3f} | "
                      f"{s['pick_fde1']:.3f} / {s['pick_fde2']:.3f} / {s['pick_fde3']:.3f} | {s['pick_gap']:.3f} | "
                      f"{s['m2_lt2']:.1f} | {s['m3_variety']:.2f} |")
    md += ["", "## Every check: tails, encoder, losses, gradient shares", "",
           "Tails on SLOW TURNS against step 0 (closest candidate FDE (3 s)).  Encoder drift = cosine distance of the "
           "projector output from step 0 on the 103.  Other logits = mean absolute change of the other 5 pred_score "
           "logits from step 0 on the 103.  Ties = picking moments whose top pdm_score is tied.  Shares = share of the "
           "LoRA gradient norm from each loss on one fixed batch.", "",
           "| pass | worse > 0.5 m | largest increase (moment) | worst 10 % mean | encoder drift | hide-and-guess (2,000 pairs) "
           "| other logits | last logit level | ties | train losses gen / scorer / game | shares gen / scorer / game | 103: closest / pick ADE |",
           "|---|" + "---|" * 11]
    for c in run["checks"]:
        t = c["tails"]
        inc = t["T2_worst_increase_vs_step0"]
        tr = c.get("train_recent")
        trs = f"{tr['generator']:.3f} / {tr['scorer']:.3f} / {tr['hide_and_guess']:.4f}" if tr else "—"
        sh = c["lora_grad"]["share"]
        o = c["other_logits"]
        md.append(f"| {c['pass']:.1f} | {t['T1_worse_gt_0.5m_vs_step0']} | {inc['m']:+.2f} (`{inc['moment'][0][:8]}` t = {inc['moment'][1]}) "
                  f"| {t['T3_worst10pct_mean_1b']:.3f} | {c['encoder_drift_cos']:.5f} | {c['game_loss_picking_pairs']:.5f} "
                  f"| {o['other5_mean_abs_change']:.3f} | {o['last_mean_level']:.2f} | {c['pdm_score_ties_1763']} | {trs} "
                  f"| {sh['generator']:.2f} / {sh['scorer']:.2f} / {sh['hide_and_guess']:.3f} "
                  f"| {c['summary']['CONT103']['m1a']:.3f} / {c['summary']['CONT103']['m6_pick_ade']:.3f} |")
    md.append("")
    return md


def plots(run: dict) -> None:
    ch = run["checks"]
    x = [c["pass"] for c in ch]
    ref = run["reference_joint_row"]["SLOW"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, key, title in ((axes[0], "m6_pick_ade", "SLOW: scorer pick ADE (3 s)"),
                           (axes[1], "m1a", "SLOW: closest candidate ADE (3 s)")):
        y = [c["summary"]["SLOW"][key] for c in ch]
        ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=5, label="Encoder, generator and scorer trained together")
        ax.axhline(y[0], color=GRAY, lw=1.5, ls="--", label="step 0 (Encoder only, tuned on KIT)")
        ax.axhline(ref[key], color=INK, lw=1.5, ls=":", label="Generator and encoder trained together")
        if run.get("chosen") is not None:
            cc = next(c for c in ch if c["pass"] == run["chosen"])
            ax.plot([cc["pass"]], [cc["summary"]["SLOW"][key]], "o", ms=10, mfc="none", mec=ORANGE, mew=2, label="chosen check")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("progress (passes over the 13,937 training moments)")
        ax.set_ylabel("metres")
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "plot_slow_pick_and_closest_by_progress.png", dpi=130)
    plt.close(fig)

    H = run["losses"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, (k, title) in zip(axes, (("generator", "generator loss"), ("scorer", "scorer loss"),
                                     ("hide_and_guess", "hide-and-guess loss"))):
        v = np.asarray(H[k])
        if not len(v):
            continue
        p = np.arange(1, len(v) + 1) / STEPS_PER_PASS
        ax.plot(p, v, color=BLUE, lw=0.6, alpha=0.25)
        w = 25
        if len(v) >= w:
            ax.plot(p[w - 1:], np.convolve(v, np.ones(w) / w, "valid"), color=BLUE, lw=2, label="mean of 25 steps")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("progress (passes)")
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "plot_training_losses_by_progress.png", dpi=130)
    plt.close(fig)


def worked_example(run: dict) -> list:
    """One picking moment: the 5 candidates with the highest target, step 0 vs the chosen check."""
    if run.get("chosen") is None:
        return []
    mom = torch.load(DT.DATA / "moments.pt", weights_only=False)["selection"]
    human = {P1.key(m): m["future_poses"] for m in mom}
    sp = R.splits(mom)
    d0 = load(DATA / "dumps" / "p00.0.pkl")
    dc = load(DATA / "dumps" / f"p{run['chosen']:04.1f}.pkl")
    g0 = load(GJK / "05_generator_survey" / "out" / "g_menu_per_moment.pkl")["original"]
    k = sorted(set(g0) & set(sp["SLOW_TURNS"]))[0]       # first baseline picking moment that is a SLOW TURN
    p = np.asarray(d0[k]["proposals"], dtype=np.float64)[:, :6, :2]
    d = np.linalg.norm(p - np.asarray(human[k], dtype=np.float64)[None, :6, :2], axis=-1).mean(1)
    tgt = np.exp(-d / 0.5); tgt /= tgt.sum()
    sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    pr0 = sm(np.asarray(d0[k]["pred_logit"], dtype=np.float64)[:, -1])
    # the chosen check's menu differs; its prediction is shown for its own candidate with the same index
    pc = np.asarray(dc[k]["proposals"], dtype=np.float64)[:, :6, :2]
    dcc = np.linalg.norm(pc - np.asarray(human[k], dtype=np.float64)[None, :6, :2], axis=-1).mean(1)
    tgtc = np.exp(-dcc / 0.5); tgtc /= tgtc.sum()
    prc = sm(np.asarray(dc[k]["pred_logit"], dtype=np.float64)[:, -1])
    md = [f"### Worked example: picking moment `{k[0][:8]}` t = {k[1]} (SLOW TURN)", "",
          "Step 0 menu, the 5 candidates with the highest target; at the chosen check the same candidate slot "
          "(the generator has moved, so its distance and target are given again).", "",
          "| slot | step 0: d_k (m) | step 0: target | step 0: prediction | chosen: d_k (m) | chosen: target | chosen: prediction |",
          "|---|---|---|---|---|---|---|"]
    for i in np.argsort(-tgt)[:5]:
        md.append(f"| {i} | {d[i]:.2f} | {tgt[i]:.3f} | {pr0[i]:.3f} | {dcc[i]:.2f} | {tgtc[i]:.3f} | {prc[i]:.3f} |")
    md += ["", f"Pick at step 0: slot {d0[k]['chosen']} (d = {d[d0[k]['chosen']]:.2f} m); at the chosen check: slot "
               f"{dc[k]['chosen']} (d = {dcc[dc[k]['chosen']]:.2f} m); closest at the chosen check: slot {int(dcc.argmin())} "
               f"(d = {dcc.min():.2f} m).", ""]
    return md


# --------------------------------------------------------------------------- #
# Held-out 223
# --------------------------------------------------------------------------- #


def heldout() -> tuple[list, dict]:
    s4 = load(S4 / "heldout223.pkl")
    s5 = load(DATA / "heldout223.pkl")
    key = lambda m: (m["scene_id"], m["keyframe_index"])
    assert [key(m) for m in s4["moments"]] == [key(m) for m in s5["moments"]]
    human = {key(m): m["human_poses"] for m in s4["moments"]}
    for m4, m5 in zip(s4["moments"], s5["moments"]):
        assert np.array_equal(m4["human_poses"], m5["human_poses"])
    gp = FR.groups(s4["moments"])
    dumps = {**s4["dumps"], 5: s5["dumps"][5]}
    rows = {**s4["rows"], 5: s5["rows"][5]}
    per = {}
    for i in NAMES:
        per[i] = P1.per_moment(dumps[i], human, None)
        pick = {(r["scene_id"], r["keyframe_index"]): r for r in rows[i]}
        for k in per[i]:
            per[i][k].update({"pick_" + x: pick[k][x] for x in ("ade", "fde@1s", "fde@2s", "fde@3s")})
            per[i][k]["pick_gap"] = per[i][k]["pick_ade"] - per[i][k]["m1a"]
    mean = lambda i, g, x: float(np.mean([per[i][k][x] for k in gp[g]]))
    n = {g: len(gp[g]) for g in gp}
    few = {g: n[g] < FR.FEW for g in gp}

    # the four references must equal s4's published table
    ft = json.loads((GJK / "09_heldout_comparison" / "out" / "final_tables.json").read_text())["models"]
    mism = [(i, g, x) for i in (1, 2, 3, 4) for g in gp for x in ("m1a", "m2_lt2", "m3_variety", "pick_ade")
            if mean(i, g, x) != ft[NAMES[i]]["groups"][g][x]["mean"]]
    assert not mism, mism

    out = {"n": n, "references_equal_s4_final_tables": True, "models": {}, "paired": []}
    md = ["## The 223 held-out moments: all five models", "",
          f"Groups: " + ", ".join(f"{GL[g]} n = {n[g]}" + (" (too few to read)" if few[g] else "") for g in FR.GROUPS) + ".", "",
          "| Model | closest candidate ADE (3 s) | closest candidate FDE (3 s) | candidates within 2 m | variety | scorer pick ADE (3 s) "
          "| scorer pick FDE (1 s / 2 s / 3 s) | pick gap | SLOW closest / pick ADE | SLOW TURNS closest / pick ADE | FAST closest / pick ADE ‡ "
          "| SLOW TURNS tails vs Encoder only: worse > 0.5 m · largest increase · worst 10 % mean |",
          "|---|" + "---|" * 11]
    for i, nm in NAMES.items():
        r = {g: {x: mean(i, g, x) for x in ("m1a", "m1b", "m2_lt2", "m3_variety", "pick_ade", "pick_fde@1s", "pick_fde@2s",
                                             "pick_fde@3s", "pick_gap")} for g in FR.GROUPS}
        t = FR.tails(per[i], per[2], gp["SLOW_TURNS"])
        out["models"][nm] = {"groups": r, "tails_vs_encoder_only": t}
        A = r["ALL"]
        tl = "—" if i == 2 else (f"{t['worse_gt_0.5m']} · {t['largest_increase_m']:+.2f} m "
                                 f"(`{t['largest_increase_moment'][0][:8]}` t = {t['largest_increase_moment'][1]}) · {t['worst10pct_mean_b']:.2f}")
        md.append(f"| {nm} | {A['m1a']:.3f} | {A['m1b']:.3f} | {A['m2_lt2']:.1f} | {A['m3_variety']:.2f} | {A['pick_ade']:.3f} "
                  f"| {A['pick_fde@1s']:.3f} / {A['pick_fde@2s']:.3f} / {A['pick_fde@3s']:.3f} | {A['pick_gap']:.3f} "
                  f"| {r['SLOW']['m1a']:.3f} / {r['SLOW']['pick_ade']:.3f} | {r['SLOW_TURNS']['m1a']:.3f} / {r['SLOW_TURNS']['pick_ade']:.3f} "
                  f"| {r['FAST']['m1a']:.3f} / {r['FAST']['pick_ade']:.3f} | {tl} |")
    md += ["", "‡ FAST has 2 moments: too few to read.", "",
           "## Paired differences on the 223 (this run minus the reference; negative = this run is better)", "",
           "95 % interval: paired bootstrap, 10,000 resamples of whole scenes, seed 0 (09_heldout_comparison's function).", "",
           "| Reference | Metric | Group | n | mean (m) | median (m) | better / worse by > 10 cm | 95 % interval (m) |",
           "|---|---|---|---|---|---|---|---|"]
    for old in (3, 4):
        for x, xl in (("m1a", "closest candidate ADE (3 s)"), ("pick_ade", "scorer pick ADE (3 s)")):
            for g in FR.GROUPS:
                dd = {k: per[5][k][x] - per[old][k][x] for k in gp[g]}
                v = np.array(list(dd.values()))
                ci = FR.boot(dd)
                out["paired"].append({"old": NAMES[old], "metric": xl, "group": g, "n": len(v), "mean": float(v.mean()),
                                      "median": float(np.median(v)), "better_gt_10cm": int((v < -0.1).sum()),
                                      "worse_gt_10cm": int((v > 0.1).sum()), "ci95": ci, "too_few": few[g]})
                md.append(f"| {NAMES[old]} | {xl} | {GL[g]}{' (too few to read)' if few[g] else ''} | {len(v)} | {v.mean():+.3f} "
                          f"| {np.median(v):+.3f} | {int((v < -0.1).sum())} / {int((v > 0.1).sum())} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    md += ["", "## Pick gap (scorer pick ADE (3 s) minus closest candidate ADE (3 s), 223 held-out moments)", "",
           "| Model | ALL | SLOW | SLOW TURNS |", "|---|---|---|---|"]
    for i, nm in NAMES.items():
        r = out["models"][nm]["groups"]
        md.append(f"| {nm} | {r['ALL']['pick_gap']:.3f} | {r['SLOW']['pick_gap']:.3f} | {r['SLOW_TURNS']['pick_gap']:.3f} |")
    md.append("")
    return md, out


def main() -> None:
    run = json.loads((OUT / "run.json").read_text())
    plots(run)
    md = ["# Tables (generated by src/report.py)", ""] + checks_table(run) + worked_example(run)
    if (DATA / "heldout223.pkl").exists():
        fs = json.loads((OUT / "final_state.json").read_text())
        assert fs["gate_A_generator_only_unchanged"]["pass"] and fs["gate_B_checkpoint_reproduces_its_dump"]["pass"]
        hmd, hout = heldout()
        md += hmd
        (OUT / "final_tables.json").write_text(json.dumps(hout, indent=1) + "\n")
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
