import json, re, sys
from pathlib import Path
M = Path("{working_folder}")
P = json.load(open(M / "out/s6_params.json")); T1 = json.load(open(M / "out/s6_t1.json")); T23 = json.load(open(M / "out/s6_t2t3.json")); T2M = json.load(open(M / "out/s6_t2_map.json")); T4 = json.load(open(M / "out/s6_t4.json"))
V = {}
f = lambda x, d=3: ("—" if x is None else f"{x:.{d}f}")
pct = lambda x: "—" if x is None else f"{100 * x:.1f} %"
n = P["new_params"]
V.update(P_GEN=f"{n['generator_read']:,}", P_SUM=f"{n['scorer_read_summary']:,}", P_COL=f"{n['rating_head_new_columns']:,}", P_UPD=f"{n['update_block']:,}", P_TOT=f"{n['total_new']:,}", P_PCT=f"{n['vs_target_pct']:+.1f} %")
c, w = T1["carry"]["vs_018b_cached_forward"], T1["wiped"]["vs_018b_cached_forward"]
yn = lambda b: "yes" if b else "no"
V.update(T1_N=f"{c['moments']:,}", T1_CARRY="PASS" if c["pass"] else "FAIL", T1_WIPED="PASS" if w["pass"] else "FAIL",
         T1C_P=yn(c["proposals_bit_identical"]), T1C_L=yn(c["pred_logit_bit_identical"]), T1C_C=yn(c["chosen_bit_identical"]), T1C_DP=str(c["max_abs_plan_diff_m"]), T1C_DL=str(c["max_abs_logit_diff"]), T1C_M=f"{T1['carry']['M_read_max_abs_over_all_moments']:.2f}",
         T1W_P=yn(w["proposals_bit_identical"]), T1W_L=yn(w["pred_logit_bit_identical"]), T1W_C=yn(w["chosen_bit_identical"]), T1W_DP=str(w["max_abs_plan_diff_m"]), T1W_DL=str(w["max_abs_logit_diff"]), T1W_M=f"{T1['wiped']['M_read_max_abs_over_all_moments']:.2f}",
         T1_SAME=yn(T1["step0_identical_measures"]))
b = T1["step0_018b"]; fp = b["follows_previous"]
V.update(B_MAP=f"{b['map_moments']:,}", B_GRADE=f(b["pick_grade_mean_map"], 4), B_ADE=f(b["closest_of_32_ade3_all"]), B_FOLLOW=pct(fp["rate"]), B_FOLLOW_N=f"{fp['n']:,}", B_FOLLOW_ADE=f(fp["ade_mean_m"]), B_FOLLOW_MED=f(fp["ade_median_m"]),
         B_HARD=str(b["hard_picks_above10"]), B_ABOVE10=str(b["moments_above10"]), B_GO=pct(b["go_rate_pull_away"]), B_PULL_N=str(b["standstill_pull_away"]),
         B_PICK_ADE=f(b["pick_ade3_all"]), B_PICK_ADE15=f(b["pick_ade3_slow15"]))
mc = T1["motion_source_check"]
V["MOTION_CHECK"] = f"max mismatch {mc['max_mismatch_m']:.4f} m, 99th percentile {mc['p99_mismatch_m']:.4f} m, median {mc['median_mismatch_m']:.5f} m over {mc['pairs']:,} consecutive pairs"
V["B_NOTE"] = (f"Hard-brake gate: 0.1.8b's step-0 count is {b['hard_picks_above10']}; under Sid's build-3 rule (must be 0) " + ("the gate can be passed." if b["hard_picks_above10"] == 0 else "0.1.8b itself would not pass it; 0.1.8b's own rule was \"≤ step 0\" (flag 3, OPEN 2)."))
t2 = T2M["t2_routing"]; cols = ["generator", "generator_lookup", "rating_head", "scorer_summary", "collision_head", "drivable_head", "update_block"]
rows = []
for les in ("generator", "grade", "human", "collision", "drivable"):
    tab = t2["table"][les]
    rows.append(f"| {les} | " + (" | ".join(("**yes**" if tab[g] == "yes" else "no") for g in cols) if isinstance(tab, dict) else tab) + " |")
V.update(T2_TABLE="\n".join(rows), T2="PASS" if t2["pass"] else "FAIL", T2_MISM=("" if t2["pass"] else "Mismatches: " + "; ".join(t2["mismatches"])), T2_MAP=str(T2M["t2_batch"]["map_rows"]))
t3 = T23["t3_wiped"]; ug = t3["update_block_grad_after_5_steps"]
V.update(T3="PASS" if t3["pass"] else "FAIL", T3_MAX=f"{max(t3['M_read_max_abs_per_step']):.1e}" if max(t3['M_read_max_abs_per_step']) else "0.0",
         T3_GRAD=f"{ug['none']} of {ug['tensors']} tensors None, the rest all-zero" if ug["all_zero_or_none"] else "NON-ZERO gradient", T3_A="pass" if t3["M_zero_at_every_read"] else "FAIL", T3_B="pass" if ug["all_zero_or_none"] else "FAIL",
         T3_CARRY=", ".join(f"{x:.2f}" for x in T23["t3_carry_M_after_first_moment_norm"]), T3_C="pass" if all(x > 0 for x in T23["t3_carry_M_after_first_moment_norm"]) else "FAIL")
def t4_table(mode):
    out = []
    for e in T4[mode]["every_10"]:
        out.append(f"| {e['step']} | {e['loss']:.4f} | {e['generator']:.4f} | {e['grade']:.4f} | {e['human']:.4f} | {e['collision']:.4f} | {e['drivable']:.4f} | {e['gate_mean']:.3f} | {e['M_norm']:.2f} | {e['grade_wait_s']:.2f} | {e['step_s']:.2f} |")
    return "\n".join(out)
V.update(T4_CARRY_TABLE=t4_table("carry"), T4_WIPED_TABLE=t4_table("wiped"))
fl = T4["carry"]["losses_first10_vs_last10"]; fw = T4["wiped"]["losses_first10_vs_last10"]
V["T4_FIRST_LAST"] = "carry " + ", ".join(f"{k} {a:.3f} → {b2:.3f}" for k, (a, b2) in fl.items()) + "; wiped " + ", ".join(f"{k} {a:.3f} → {b2:.3f}" for k, (a, b2) in fw.items())
falling = {k: T4["carry"]["losses_falling"][k] for k in fl}
V["T4_FALL"] = "carry " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in falling.items()) + "; wiped " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in T4["wiped"]["losses_falling"].items())
V.update(T4_NAN="no" if (T4["carry"]["finite_all"] and T4["wiped"]["finite_all"]) else "YES", T4_FIN_C=yn(T4["carry"]["finite_all"]), T4_FIN_W=yn(T4["wiped"]["finite_all"]),
         T4_MNORM=f"{T4['carry']['M_norm_step50']:.2f}", T4_GATE=f"{sum(T4['carry']['gate_mean_per_slot_step50']) / 16:.3f}", T4_GATE_SLOTS=", ".join(f"{x:.2f}" for x in T4["carry"]["gate_mean_per_slot_step50"]),
         T4_MAP=f"{T4['carry']['map_rows_per_step']:.1f}", T4_GPU=f"{max(T4['carry']['gpu_peak_alloc_gb'], T4['wiped']['gpu_peak_alloc_gb']):.1f}")
dw = [r["data_wait_s"] for r in T4["carry"]["records"]]
V["T4_NOTE"] = (f"Data wait (NAS read of the two clips, prefetched 4 batches ahead, excluded from the step time): mean {sum(dw) / len(dw):.1f} s, max {max(dw):.1f} s per step in carry mode; "
                f"GPU shared with other users' jobs during the run (`gpu_before`: {T4['gpu_before']}).")
t5 = T4["t5"]; tm = t5["timing_steps_10_50_carry"]; pr = t5["projection"]; g = pr["grading"]
V.update(T5_STEP=f"{tm['step_s']:.2f}", T5_STEP_W=f"{t5['timing_steps_10_50_wiped']['step_s']:.2f}", T5_FWD=f"{tm['forward_s']:.2f}", T5_WAIT=f"{tm['grade_wait_s']:.2f}", T5_BWD=f"{tm['backward_s']:.2f}",
         T5_CHECK=f"{t5['check_s_step0']:.0f}" if t5["check_s_step0"] else "n/a")
for key, suf in (("oversample_on", "ON"), ("oversample_off", "OFF")):
    p = pr[key]
    V.update({f"T5_STEPS_{suf}": f"{p['steps_per_pass']:,}", f"T5_CHK_{suf}": str(p["checks_per_pass"]), f"T5_TR_{suf}": f"{p['train_h_per_pass']:.2f}", f"T5_H1_{suf}": f"{p['wall_h_per_pass']:.2f}", f"T5_H3_{suf}": f"{p['wall_h_3_passes']:.1f}"})
V.update(T5_SHARE=pct(g["wait_share_of_step"]), T5_DOM="grading dominates the step" if g["grading_dominates"] else "grading does not dominate the step",
         T5_CORES=f"Estimated grading wait with 4 cores {g['estimated_wait_s_4_cores']:.2f} s (step ≈ {g['step_s_est_4_cores']:.2f} s), with 8 cores {g['estimated_wait_s_8_cores']:.2f} s (step ≈ {g['step_s_est_8_cores']:.2f} s)")
pp = T4["pass_plan"]["oversample_on"]; po = T4["pass_plan"]["oversample_off"]
V.update(W_SKIP=str(pp["clips_skipped_no_16_run"]), W_USED=f"{pp['clips_used']:,}", W_GAP="115", W_STEPS_ON=f"{pp['steps']:,}", W_WIN_ON=f"{pp['windows']:,}", W_DUP=f"{pp['duplicated_event_windows']:,}", W_STEPS_OFF=f"{po['steps']:,}", W_WIN_OFF=f"{po['windows']:,}")
V.update(START=T1["started"][11:16] if "started" in T1 else "", END=T4["ended"][11:16])
V["T5_CHECK"] = f"{t5['check_s_step0']:.0f}"
# ---- flags and OPEN (numbers from the files above) ----
old = json.load(open(M / "out/attempt_before_slot_emb/s6_t4.json"))
og = old["carry"]["gate_mean_per_slot_step50"]
FL = [
 f"**Environment shim (`src/s6_env.py`).** On 2026-11-09 a root-disk relief move (another session's script, `{nas_store}/_move_log_20261109.txt`) replaced `Paddock/paddock-sid/Alpasim/analysis_3.0` with a symlink to the NAS. `s35_common.py` (read-only, inside that tree) locates the vendored navsim through its own resolved path, which then pointed at a NAS folder that does not exist, and every 0.1.8b import failed (`logs/s6_all_attempt1_save_dump_crash.log` is the second launch; the first died on this). The shim re-points the module constant at the local tree before `setup_env()` runs, in this process and in each grading worker; nothing under the earlier lanes was edited. While the move ran the NAS delivered 0.6–4 MB/s (18 MB test read in 4.2 s; the 1.27 GB start model took 572 s to read once), so the first T1 took 82 min and its check timing is not representative; after the move finished (16:18) the NAS read at 69 MB/s and all tests were rerun (`logs/s6_chain.log`).",
 "**Rating-head widening is two matmuls, not one widened `Linear`** (same linear map; bit-identity at step 0, see \"The model\"). The routing table's \"rating head\" group holds the 10 `pred_score` tensors plus the new 256 × 1024 columns (11 tensors).",
 f"**Hard-brake gate rule.** Sid's build-3 rule is \"must be 0\"; 0.1.8b's trainer used \"≤ step 0\" (`train_018b.py:1281`). 0.1.8b-p06 has **{b['hard_picks_above10']}** hard-brake picks above 10 m/s at step 0 on the picking set (0.1.7 had 43, `run.json: reference_step0`), so both rules coincide at step 0 here; the trainer implements Sid's rule with the 0.1.8b rule selectable (`HARD_GATE_RULE`). OPEN 2.",
 f"**Slot embedding added to the update block (+4,096 parameters), a deviation from the design as written.** First smoke run (archived `out/attempt_before_slot_emb/s6_t4.json`): gate mean per slot at step 50 = {og[0]:.2f} on all 16 slots in carry mode ({old['wiped']['gate_mean_per_slot_step50'][0]:.2f} in wiped), i.e. the slots never differentiated (zeros at the start + identical queries). With the slot embedding the 16 gates differ (T4 table). OPEN 4.",
 f"**New parameters {n['total_new']:,} against the ≈ 1.5 M target ({n['vs_target_pct']:+.1f} %).** The update block is {n['update_block']:,} of them (FFN 256 → 1024 → 256 = 0.53 M). Within reach of the target by halving the FFN width (−0.26 M) if Sid wants it exact.",
 f"**Smoke-run trajectories are not bit-reproducible across launches.** Step 1 is identical in every run (loss 2.1986), but the killed run (`logs/s6_all_attempt2_killed_by_session_restart.log`) and the rerun before the slot embedding differ at step 50 (carry loss 1.8959 vs 1.8496): the deformable-attention backward uses atomic adds, as in 0.1.8b. T4/T5 numbers are from one run each.",
 f"**Windows and pass length.** {pp['clips_skipped_no_16_run']} of 4,015 training clips have no run of 16 consecutive moments and are skipped; the pass has {pp['steps']:,} steps with event over-sampling ({pp['duplicated_event_windows']:,} of {pp['windows'] - pp['duplicated_event_windows']:,} windows duplicated) and {po['steps']:,} without, against 0.1.8b's 4,650 steps of 32 single moments; moments seen per pass {pp['moments_per_pass']:,} (with duplicates) vs 0.1.8b's 148,796 draws.",
 f"**Motion input source.** The car motion since the previous moment is taken from the previous moment's logged human future (waypoint 0). Consistency against the current moment's future: median mismatch {mc['median_mismatch_m'] * 1000:.1f} mm, 99th percentile {mc['p99_mismatch_m']:.3f} m, max {mc['max_mismatch_m']:.3f} m on {mc['pairs']:,} consecutive pairs (converter interpolation). In the car this input is the simulated ego motion between calls; at a window's or chain's first moment it is zero.",
 f"**0.1.8b's pick is not steady from moment to moment**: the pick-follows-previous rate at step 0 is {pct(fp['rate'])} ({fp['n']:,} pairs), the ADE between consecutive picks (moved into the same frame) has median {fp['ade_median_m']:.2f} m and mean {fp['ade_mean_m']:.2f} m. Reported, not gated; it is the baseline the memory arms are compared against.",
 f"**Session restart killed the first smoke run** at wiped step 22 (14:45 IST; harness background jobs die with the session). Reruns were launched detached (`setsid nohup`). The human-closeness lesson does not fall within 50 steps in either mode ({fl['human'][0]:.3f} → {fl['human'][1]:.3f} carry); the other four fall. 50 steps is a smoke test, not evidence either way.",
 f"**Grading share.** With the pool on the two pinned cores the grading wait is {tm['grade_wait_s']:.2f} s of a {tm['step_s']:.2f} s step ({pct(g['wait_share_of_step'])}); map rows average {T4['carry']['map_rows_per_step']:.1f} of 24 per step (whole-clip windows give 0, 12 or 24). The 4- and 8-core figures in T5 are a scaling estimate, not a measurement.",
 "**Idle grading-worker processes from an earlier session** (10 `drive-jepa` python processes started 2026-11-07, 0 % CPU, `ps`) are still alive on shakaal; not touched.",
]
V["FLAGS"] = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(FL))
OP = [
 "**Replay routing: M detached in the replay's generator read (built) vs attached.** Recommend: keep it detached — the scorer lessons then teach the update block only through the scorer's own summary, and the generator's use of memory is taught by the generator lesson alone (mirrors 0.1.8b, where scorer lessons never reached the generator; T2 table holds). Alternative: attach M in the replay's look-up (weights still detached), so grade / human lessons also shape what the generator reads from M; one line in `MemoryModel.replay`.",
 f"**Hard-brake gate.** Recommend: Sid's \"must be 0\" as built (0.1.8b-p06 is at 0 at step 0, so no check is lost by it). Alternative: 0.1.8b's \"≤ step 0\" (`HARD_GATE_RULE = \"step0\"`), identical here unless the count rises.",
 "**Camera sidecar.** `lidar2img` and `img_shape` are constant within every sampled clip and identical across the 140 sampled clips. Recommend: one verification pass over all 4,015 + 84 clips' first and last moments (≈ 8,200 small NAS reads, no GPU) and then a single constant file under `A3/cache/` read once per run. Alternative: keep one npz read per clip (as now; negligible cost).",
 "**Slot embedding (flag 4).** Recommend: accept the learned 16 × 256 slot embedding on the update query (+4,096 parameters) as part of the design. Alternative: a learned non-zero initial M instead of zeros (breaks the symmetry too, but then \"zero at the start\" is no longer literally true and wiped mode would reset to that learned state).",
 f"**Event over-sampling default.** Recommend: on (pass {pp['steps']:,} steps, +{100 * (pp['steps'] / po['steps'] - 1):.0f} % over {po['steps']:,}; the memory moments are 5 % of the data). Alternative: off, for a pass closer to 0.1.8b's length.",
 f"**Parameter budget.** Recommend: keep {n['total_new']:,} (+{n['vs_target_pct']:.0f} %). Alternative: FFN 256 → 512 → 256 in the update block → ≈ 1.40 M.",
]
V["OPEN"] = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(OP))
extra = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
V.update(extra)
txt = (M / "reports/BUILD3_MODEL_TRAINER.md").read_text()
for k, v in V.items():
    txt = txt.replace("{{" + k + "}}", str(v))
left = re.findall(r"\{\{[A-Z0-9_]+\}\}", txt)
(M / "reports/BUILD3_MODEL_TRAINER.md").write_text(txt)
print("left:", left)
