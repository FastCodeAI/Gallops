#!/usr/bin/env python
"""S10 task 7: the final-checkpoint rule of PREDICTIONS_phase1.md ("Final = among pilot, +1, +2 the lowest held-out swing that still
qualifies") on the float32 SEEN held-out readouts (s10_pilot_readout.py): pass 9 and the pilot from gen/t6_readout/{P9,P0}, +1 and +2 from
gen/t7_readout/{plus1,plus2} (final_P0/pass_01.pt, pass_02.pt). Qualifies = trained finite, plans finite, human-path term at most
1.05 x the pass-9 float32 reference on highway, close-pass and ordinary. CPU, no model. Writes REPORT_DIR/final_choice.json, the four-row
table CSV and one plot. Usage: s10_final_choice.py REPORT_DIR"""
import sys, json, pathlib, datetime, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
GEN = pathlib.Path("{runs}/drivejepa_retrain/12_second_training_run/gen"); OUT = pathlib.Path(sys.argv[1]); HW20 = "highway, car >= 20 m/s"; SL3 = ["highway", "close-pass", "ordinary"]; SLICES = SL3 + ["all", HW20]
FIXED = {"highway": 1.311, "close-pass": 0.442, "ordinary": 0.367}
st = json.load(open(GEN / "final_P0/state.json")); tl = (GEN / "final_P0/train.log").read_text().splitlines(); pd_ = int(st.get("pass_done") or 0)
CAND = [("pilot", "pilot (P0, pass 9 + 1 pass at lr 1e-4)", GEN / "t6_readout/P0/t6_readout.json", GEN / "pilot_fix_l0/pass_01.pt", True),
        ("+1", "pilot + 1 pass at lr 2.5e-5", GEN / "t7_readout/plus1/t6_readout.json", GEN / "final_P0/pass_01.pt", pd_ >= 1),
        ("+2", "pilot + 2 passes at lr 2.5e-5", GEN / "t7_readout/plus2/t6_readout.json", GEN / "final_P0/pass_02.pt", pd_ >= 2)]
R = {"pass 9": json.load(open(GEN / "t6_readout/P9/t6_readout.json"))}
for k, lab, f, ck, ok in CAND:
    if f.exists(): R[k] = json.load(open(f)); assert pathlib.Path(R[k]["init"]) == ck, (R[k]["init"], ck)
rows = [dict(checkpoint=k, precision="float32" if nm == "f32" else "16-bit", slice=sl, **r["precision"][nm][sl]) for k, r in R.items() for nm in ("f32", "amp") for sl in SLICES]
T = pd.DataFrame(rows); T.to_csv(OUT / "final_readout_table.csv", index=False)

def rule(nm):
    ref = {s: R["pass 9"]["precision"][nm][s]["human_path_term"] for s in SL3}; tr = dict(precision=nm, reference_pass9=ref, limits={s: 1.05 * ref[s] for s in SL3}, pass9_swing_hw20=R["pass 9"]["precision"][nm][HW20]["swing_median_all_32_plans_mps"], candidates={})
    for k, lab, f, ck, ok in CAND:
        d = dict(what=lab, checkpoint=str(ck), checkpoint_exists=ck.exists(), trained_finite=bool(ok), readout_present=k in R)
        if k in R:
            p = R[k]["precision"][nm]; d["plans_finite"] = bool(R[k]["all_finite"][nm]); d["human_path_term"] = {s: p[s]["human_path_term"] for s in SL3}; d["ratio_to_reference"] = {s: p[s]["human_path_term"] / ref[s] for s in SL3}
            d["pass_slice"] = {s: bool(p[s]["human_path_term"] <= 1.05 * ref[s]) for s in SL3}; d["pass_slice_with_fixed_figures"] = {s: bool(p[s]["human_path_term"] <= 1.05 * FIXED[s]) for s in SL3}
            d["swing_hw20"] = p[HW20]["swing_median_all_32_plans_mps"]; d["swing_hw20_ci"] = [p[HW20]["swing_median_lo95"], p[HW20]["swing_median_hi95"]]
            d["qualifies"] = bool(d["trained_finite"] and d["checkpoint_exists"] and d["plans_finite"] and all(d["pass_slice"].values()))
        else: d["qualifies"] = False
        tr["candidates"][k] = d
    q = {k: v for k, v in tr["candidates"].items() if v["qualifies"]}
    tr["chosen"] = min(q, key=lambda k: q[k]["swing_hw20"]) if q else None; tr["order_by_swing"] = sorted(q, key=lambda k: q[k]["swing_hw20"]); return tr
T32 = rule("f32"); T16 = rule("amp"); ch = T32["chosen"]
choice = dict(written=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), rule_source="S10/PREDICTIONS_phase1.md (sha256 706a93b0e23643e90d9be09d78cbd8324bcb894de62c3d7385dd15841b534ccb): Final = among pilot, +1, +2 the lowest held-out swing that still qualifies",
              set="SEEN held-out, 4,400 examples / 145 clips, forward only, batch 32 clip order, batch fix on, float32 (autocast off, TF32 off)", chosen=ch, chosen_checkpoint=T32["candidates"][ch]["checkpoint"] if ch else None,
              final_run=dict(state={k: st.get(k) for k in ("run", "lr", "lam", "pass_done", "stopped", "updates_skipped", "sec_per_step", "steps", "train_secs")}, last_log_line=tl[-1][:300] if tl else None),
              rule_on_float32=T32, rule_on_16bit_for_comparison=T16, f32_vs_16bit_plan_diff_m={k: v["f32_vs_amp_plan_diff_m"] for k, v in R.items()}, table=rows)
json.dump(choice, open(OUT / "final_choice.json", "w"), indent=1)

order = list(R); x = np.arange(len(order)); COL = {"pass 9": "#888888", "pilot": "#4c78a8", "+1": "#f58518", "+2": "#54a24b"}; fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for j, (sl, key, yl, tt) in enumerate(((HW20, "swing_median", "median swing over all 32 plans, m/s", "Held-out median swing, highway with car >= 20 m/s"), ("highway", "human_path_term", "human-path term (best of 32)", "Held-out human-path term, highway"))):
    kv = "swing_median_all_32_plans_mps" if j == 0 else key; v = np.array([R[k]["precision"]["f32"][sl][kv] for k in order]); l = np.array([R[k]["precision"]["f32"][sl][key + "_lo95"] for k in order]); h = np.array([R[k]["precision"]["f32"][sl][key + "_hi95"] for k in order])
    ax[j].bar(x, v, 0.6, color=[COL[k] for k in order], edgecolor="k", lw=0.5, yerr=[v - l, h - v], capsize=4)
    for xi, vi, li, hi in zip(x, v, l, h): ax[j].text(xi, hi, f"{vi:.2f}\n({li:.2f}-{hi:.2f})", ha="center", va="bottom", fontsize=9)
    ax[j].set_xticks(x); ax[j].set_xticklabels([k + (" (chosen)" if k == ch else "") for k in order]); ax[j].set_xlabel("checkpoint"); ax[j].set_ylabel(yl + "  (bars: 95 % clip-bootstrap interval)"); ax[j].set_title(tt); ax[j].grid(axis="y", alpha=0.3); ax[j].set_ylim(0, 1.25 * h.max())
ax[0].axhline(5, color="k", ls=":", lw=1, label="5 m/s (target T-G2, defined on UNSEEN)"); ax[1].axhline(T32["limits"]["highway"], color="r", ls="--", lw=1.3, label=f"limit 1.05 x pass 9 float32 = {T32['limits']['highway']:.3f}"); ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
fig.suptitle("S10 final checkpoint choice on SEEN held-out (4,400 examples, 145 clips), float32"); fig.tight_layout(); fig.savefig(OUT / "final_heldout_bars.png", dpi=130); plt.close(fig)
pd.set_option("display.width", 250)
print(T[T.precision == "float32"][["checkpoint", "slice", "human_path_term", "human_path_term_lo95", "human_path_term_hi95", "swing_median_all_32_plans_mps", "swing_median_lo95", "swing_median_hi95", "share_plans_swing_above_5mps", "share_with_plan_within_1p5m_4s"]].round(4).to_string())
for tr in (T32, T16): print(json.dumps(tr, indent=1))
