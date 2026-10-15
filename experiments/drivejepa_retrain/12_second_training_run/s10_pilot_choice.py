#!/usr/bin/env python
"""S10 task 6: five-arm table (float32 and 16-bit), the choice rule of PREDICTIONS_phase1.md applied on the float32 readout (and, for
comparison, on the 16-bit one), training curves and bar plots. CPU, no model. Reads NAS/12_second_training_run/gen/t6_readout/<arm>/t6_readout.json
(s10_pilot_readout.py), each pilot's state.json / train.log / readout_pass_01.json. Usage: s10_pilot_choice.py REPORT_DIR"""
import sys, json, re, time, pathlib, datetime, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; GEN = pathlib.Path("{runs}/drivejepa_retrain/12_second_training_run/gen"); OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ARMS = [("P0", "pilot_fix_l0", 0.0, "std"), ("P1", "pilot_fix_l0p1", 0.1, "std"), ("P2", "pilot_fix_l1", 1.0, "std"), ("P3", "pilot_fix_l10", 10.0, "std"), ("P4", "pilot_fixwide_l0", 0.0, "wide")]
FIXED = {"highway": 1.311, "close-pass": 0.442, "ordinary": 0.367}; HW20 = "highway, car >= 20 m/s"; SL3 = ["highway", "close-pass", "ordinary"]; SLICES = SL3 + ["all", HW20]
R = {"P9": json.load(open(GEN / "t6_readout/P9/t6_readout.json"))}; INFO = {}
for arm, run, lam, rng in ARMS:
    st = json.load(open(GEN / run / "state.json")) if (GEN / run / "state.json").exists() else {}; f = GEN / "t6_readout" / arm / "t6_readout.json"
    tl = (GEN / run / "train.log").read_text().splitlines() if (GEN / run / "train.log").exists() else []; done = bool(tl and "] DONE " in tl[-1])
    INFO[arm] = dict(run=run, lam=lam, ref_range=rng, state=st, done_line=done, last_log_line=tl[-1][:300] if tl else None, trained=st.get("pass_done") == 1 and st.get("stopped") is None and done)
    if f.exists(): R[arm] = json.load(open(f))
    own = GEN / run / "readout_pass_01.json"; INFO[arm]["trainer_readout"] = json.load(open(own))["slices"] if own.exists() else None

# ---------------------------------------------------------------- table
rows = []
for arm, r in R.items():
    for nm in ("f32", "amp"):
        for sl in SLICES: rows.append(dict(arm=arm, precision="float32" if nm == "f32" else "16-bit", slice=sl, **r["precision"][nm][sl]))
T = pd.DataFrame(rows); T.to_csv(OUT / "pilot_readout_table.csv", index=False)

# ---------------------------------------------------------------- rule
def rule(nm):
    ref = {s: R["P9"]["precision"][nm][s]["human_path_term"] for s in SL3}; dev = {s: abs(ref[s] / FIXED[s] - 1) for s in SL3}; tr = dict(precision=nm, reference_pass9=ref, reference_vs_prediction_file_figures_rel=dev, reference_ok=all(v < 0.02 for v in dev.values()), limits={s: 1.05 * ref[s] for s in SL3},
                                                                                                                                                 pass9_swing_hw20=R["P9"]["precision"][nm][HW20]["swing_median_all_32_plans_mps"], arms={})
    if not tr["reference_ok"]: tr["chosen"] = None; tr["stop"] = "pass-9 reference differs from the prediction file's figures by 2 percent or more"; return tr
    for arm, run, lam, rng in ARMS:
        i = INFO[arm]; d = dict(lam=lam, ref_range=rng, trained_finite=bool(i["trained"]), readout_present=arm in R)
        if arm in R:
            p = R[arm]["precision"][nm]; d["plans_finite"] = bool(R[arm]["all_finite"][nm]); d["human_path_term"] = {s: p[s]["human_path_term"] for s in SL3}; d["ratio_to_reference"] = {s: p[s]["human_path_term"] / ref[s] for s in SL3}
            d["pass_slice"] = {s: bool(p[s]["human_path_term"] <= 1.05 * ref[s]) for s in SL3}; d["swing_hw20"] = p[HW20]["swing_median_all_32_plans_mps"]; d["swing_hw20_ci"] = [p[HW20]["swing_median_lo95"], p[HW20]["swing_median_hi95"]]
            d["qualifies"] = bool(d["trained_finite"] and d["plans_finite"] and all(d["pass_slice"].values()))
            d["pass_slice_with_fixed_figures"] = {s: bool(p[s]["human_path_term"] <= 1.05 * FIXED[s]) for s in SL3}
        else: d["qualifies"] = False
        tr["arms"][arm] = d
    q = {k: v for k, v in tr["arms"].items() if v["qualifies"]}
    if not q: tr["chosen"] = None; tr["stop"] = "no arm qualifies"; return tr
    lo = min(v["swing_hw20"] for v in q.values()); tied = [k for k, v in q.items() if v["swing_hw20"] <= lo + 0.5]; tr["lowest_swing"] = lo; tr["tied_within_0p5"] = tied
    tied.sort(key=lambda k: (q[k]["ref_range"] != "std", q[k]["lam"])); tr["chosen"] = tied[0]
    tr["tie_reading"] = "arms whose swing is within 0.5 m/s of the lowest qualifying swing are tied; among them: standard range before P4, then the smaller lambda"; return tr
TR32 = rule("f32"); TR16 = rule("amp"); now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
choice = dict(written=now, rule_source="S10/PREDICTIONS_phase1.md (sha256 706a93b0e23643e90d9be09d78cbd8324bcb894de62c3d7385dd15841b534ccb)", set="SEEN held-out, 4,400 examples / 145 clips, forward only, batch 32 clip order, batch fix on; P4 with the wide range",
              chosen_arm=TR32["chosen"], chosen_run=INFO[TR32["chosen"]]["run"] if TR32["chosen"] else None, rule_on_float32=TR32, rule_on_16bit_for_comparison=TR16,
              f32_vs_16bit_plan_diff_m={k: v["f32_vs_amp_plan_diff_m"] for k, v in R.items()}, table=rows,
              arms_info={k: dict(run=v["run"], lam=v["lam"], ref_range=v["ref_range"], trained=v["trained"], stopped=v["state"].get("stopped"), last_log_line=v["last_log_line"]) for k, v in INFO.items()})
json.dump(choice, open(OUT / "pilot_choice.json", "w"), indent=1)

# ---------------------------------------------------------------- run facts
facts = []
for arm, run, lam, rng in ARMS:
    st = INFO[arm]["state"]; tl = (GEN / run / "train.log").read_text().splitlines() if (GEN / run / "train.log").exists() else []; ts = [re.match(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)", l) for l in tl]; ts = [m.group(1) for m in ts if m]
    wall = (datetime.datetime.strptime(ts[-1], "%m-%d %H:%M:%S") - datetime.datetime.strptime(ts[0], "%m-%d %H:%M:%S")).total_seconds() if ts else float("nan")
    facts.append(dict(arm=arm, run=run, lam=lam, ref_range=rng, first_log=ts[0] if ts else None, last_log=ts[-1] if ts else None, wall_h_first_to_last_log_line=wall / 3600, train_secs=st.get("train_secs"), sec_per_step=st.get("sec_per_step"), steps=st.get("steps"),
                      updates_skipped=st.get("updates_skipped"), gpu_max_reserved_gib=st.get("gpu_max_reserved_gib"), gpu_max_allocated_gib=st.get("gpu_max_allocated_gib"), pass_done=st.get("pass_done"), stopped=st.get("stopped"),
                      **{f"train_mean_{k}": v for k, v in (st.get("train_means") or {}).items()}))
F = pd.DataFrame(facts); F.to_csv(OUT / "pilot_run_facts.csv", index=False)

# ---------------------------------------------------------------- plots
COL = {"P0": "#4c78a8", "P1": "#f58518", "P2": "#54a24b", "P3": "#e45756", "P4": "#7b5ea7", "P9": "#888888"}; LAB = {"P0": "P0 lambda 0", "P1": "P1 lambda 0.1", "P2": "P2 lambda 1", "P3": "P3 lambda 10", "P4": "P4 lambda 0, wide range", "P9": "pass 9"}
pat = re.compile(r"pass 1 step (\d+)/\d+: total ([\d.eE+-]+|nan|inf) \| human ([\d.eE+-]+|nan|inf) \| catalogue ([\d.eE+-]+|nan|inf) \| raw spacing ([\d.eE+-]+|nan|inf)")
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2)); curves = []
for arm, run, lam, rng in ARMS:
    if not (GEN / run / "train.log").exists(): continue
    m = [pat.search(l) for l in (GEN / run / "train.log").read_text().splitlines()]; d = np.array([[float(x) for x in g.groups()] for g in m if g and int(g.group(1)) > 1])
    if not len(d): continue
    for j, (c, nm) in enumerate(((2, "human"), (3, "catalogue"), (4, "raw_spacing"))): ax[j].plot(d[:, 0], d[:, c], color=COL[arm], lw=1.3, label=LAB[arm])
    curves += [dict(arm=arm, step=int(r[0]), total=r[1], human=r[2], catalogue=r[3], raw_spacing=r[4]) for r in d]
for j, tt in enumerate(("Human part of the loss", "Catalogue part of the loss", "Raw spacing term, m^2 (before the weight w)")):
    ax[j].set_yscale("log"); ax[j].set_xlabel("training step (batch 32; 4,377 steps = 1 pass)"); ax[j].set_ylabel("mean over the last 50 steps, 4 stages combined (log scale)"); ax[j].set_title(tt); ax[j].grid(alpha=0.3, which="both")
ax[0].legend(fontsize=9); fig.suptitle("S10 pilots: training loss parts against step, five arms (from pass 9, lr 1e-4)"); fig.tight_layout(); fig.savefig(OUT / "pilot_training_curves.png", dpi=130); plt.close(fig)
pd.DataFrame(curves).to_csv(OUT / "pilot_training_curves.csv", index=False)

order = ["P9"] + [k for k, *_ in ARMS if k in R]; x = np.arange(len(order)); fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
for j, (sl, key, lo, hi, yl, tt) in enumerate(((HW20, "swing_median_all_32_plans_mps", "swing_median_lo95", "swing_median_hi95", "median swing over all 32 plans, m/s", "Held-out median swing, highway with car >= 20 m/s"),
                                               ("highway", "human_path_term", "human_path_term_lo95", "human_path_term_hi95", "human-path term (best of 32)", "Held-out human-path term, highway"))):
    for k, (nm, off, hatch, al) in enumerate((("f32", -0.2, None, 1.0), ("amp", 0.2, "//", 0.55))):
        v = np.array([R[a_]["precision"][nm][sl][key] for a_ in order]); l = np.array([R[a_]["precision"][nm][sl][lo] for a_ in order]); h = np.array([R[a_]["precision"][nm][sl][hi] for a_ in order])
        ax[j].bar(x + off, v, 0.38, color=[COL[a_] for a_ in order], alpha=al, hatch=hatch, edgecolor="k", lw=0.5, yerr=[v - l, h - v], capsize=3, label="float32 (what the car runs)" if nm == "f32" else "16-bit autocast (as the trainer logs)")
        for xi, vi in zip(x + off, v): ax[j].text(xi, 0.02 * ax[j].get_ylim()[1], f"{vi:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax[j].set_xticks(x); ax[j].set_xticklabels([LAB[a_] for a_ in order], rotation=15, fontsize=9); ax[j].set_ylabel(yl + "  (bars: 95 % clip-bootstrap interval)"); ax[j].set_title(tt); ax[j].grid(axis="y", alpha=0.3)
ax[0].axhline(5, color="k", ls=":", lw=1, label="5 m/s (target T-G2, defined on UNSEEN)")
if TR32.get("limits"): ax[1].axhline(TR32["limits"]["highway"], color="r", ls="--", lw=1.3, label=f"limit 1.05 x pass 9 float32 = {TR32['limits']['highway']:.3f}")
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8); fig.suptitle("S10 pilots on SEEN held-out (4,400 examples, 145 clips): float32 and 16-bit readouts"); fig.tight_layout(); fig.savefig(OUT / "pilot_heldout_bars.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- print
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(T[["arm", "precision", "slice", "human_path_term", "human_path_term_lo95", "human_path_term_hi95", "best_of_32_dist_4s_mean_m", "share_with_plan_within_1p5m_4s", "swing_median_all_32_plans_mps", "swing_median_lo95", "swing_median_hi95", "share_plans_swing_above_5mps", "spacing_sq_mean_m2", "spacing_sq_median_m2"]].round(3).to_string())
for tr in (TR32, TR16): print(json.dumps({k: v for k, v in tr.items()}, indent=1))
print(json.dumps(choice["f32_vs_16bit_plan_diff_m"], indent=1)); print(F.round(3).T.to_string())
for arm in INFO:
    o = INFO[arm]["trainer_readout"]
    if o and arm in R: print(arm, "trainer's own 16-bit readout vs mine (16-bit): hw20 swing", o[HW20]["swing_median_all_32_plans_mps"], R[arm]["precision"]["amp"][HW20]["swing_median_all_32_plans_mps"], "| highway human", o["highway"]["human_path_term"], R[arm]["precision"]["amp"]["highway"]["human_path_term"])
