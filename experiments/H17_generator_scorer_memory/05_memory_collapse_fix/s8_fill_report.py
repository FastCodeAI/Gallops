"""Fills reports/BUILD4_MEMORY_FIX.md from the build-4 outputs (every number from a file). Usage: python3 -I s8_fill_report.py [extras.json]"""
import json, re, sys
from pathlib import Path
import numpy as np
M = Path("{working_folder}"); R = Path("{experiment_store}/runs/build4_short")
J = lambda n: json.load(open(M / "out" / n))
PA, T1, T2, T3, T4, BEF, SH = J("s6_params_v2.json"), J("s6_t1_v2.json"), J("s6_t2_map_v2.json"), J("s6_t2t3_v2.json"), J("s6_t4_v2.json"), J("build4_before.json"), J("build4_short.json")
run = json.load(open(R / "run.json")); V = {}
f = lambda x, d=4: "—" if x is None else f"{x:.{d}f}"
pf = lambda b: "PASS" if b else "FAIL"
gmm = lambda st: f"{st['gate_min']:.3f} / {st['gate_median']:.3f} / {st['gate_max']:.3f}"
np_ = PA["new_params"]
V.update(NEW_TOTAL=f"{np_['total_new']:,}", P_GEN=f"{np_['generator_read']:,}", P_SUM=f"{np_['scorer_read_summary']:,}", P_COLS=f"{np_['rating_head_new_columns']:,}", P_UPD=f"{np_['update_block']:,}", P_M0=f"{np_['start_state']:,}")
V.update(T1_C=pf(T1["carry"]["vs_018b_cached_forward"]["pass"]), T1_W=pf(T1["wiped"]["vs_018b_cached_forward"]["pass"]), T1_N=f"{T1['carry']['vs_018b_cached_forward']['moments']:,}",
         T1_M0=f"{T1['M0_read_at_every_moment']['carry']:.2f}")
tab = T2["t2_routing"]["table"]; m0 = [k for k, v in tab.items() if isinstance(v, dict) and v.get("start_state") == "yes"]
V.update(T2_PASS=pf(T2["t2_routing"]["pass"]), T2_MAP=str(T2["t2_batch"]["map_rows"]), T2_M0=", ".join(m0) if m0 else "none")
t3 = T3["t3_wiped"]; u = t3["update_block_grad_after_5_steps"]
V.update(T3_PASS=pf(t3["pass"]), T3_DEV=f"{max(t3['M_read_max_dev_from_start_per_step']):.1e}", T3_UPD=f"{u['none']} of {u['tensors']} tensors None, the rest all-zero" if u["all_zero_or_none"] else "NON-ZERO",
         T3_CARRY=", ".join(f"{x:.2f}" for x in T3["t3_carry_M_after_first_moment_dev_from_start_norm"]))
V["T4_FINITE"] = "no" if (T4["carry"]["finite_all"] and T4["wiped"]["finite_all"]) else "YES"
st = SH["step930"]; b = BEF["stats"]; rule = SH["rule"]; cond = rule["conditions"]
V.update(VERDICT=SH["verdict"], FAILED_TXT=(" (failed: " + ", ".join(rule["failed"]) + ")") if rule["failed"] else "")
V.update(B_COS=f(b["cos_read_mean"]), A_COS=f(st["cos_read_mean"]), B_GATE=gmm(b), A_GATE=gmm(st), A_GSLOTS=str(st["slots_gate_ge_0.05"]), A_GRATIO=f(st["gate_max_over_min"], 1),
         B_CHG=f(b["memory_change_per_moment"]), A_CHG=f(st["memory_change_per_moment"]), B_FIN="none" if b["finite_all"] else "YES", A_FIN="none" if st["finite_all"] else "YES",
         A_PG=f(st["check"]["pick_grade_mean_map"]), B_MN=f(b["M_read_norm_mean"], 1), A_MN=f(st["M_read_norm_mean"], 1), B_BC=f(b.get("between_clip_cos_mean")), A_BC=f(st.get("between_clip_cos_mean")),
         A_PGW=f(st["wiped_same_weights"]["check"]["pick_grade_mean_map"]) if "wiped_same_weights" in st else "—",
         R_A="yes" if cond["a_slot_diversity"] else "NO", R_B="yes" if cond["b_gate_spread"] else "NO", R_C="yes" if cond["c_memory_change"] else "NO", R_D="yes" if cond["d_finite"] else "NO", R_E="yes" if cond["e_pick_grade"] else "NO",
         B_AGREE=", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in BEF["agrees_with_pilot_check"].items()))
V["PREREG_TS"] = re.search(r"Written: (.*)", (M / "reports/BUILD4_PREREG.md").read_text()).group(1); V["LAUNCH_TS"] = run["started"]
extra = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
V.update(extra)
txt = (M / "reports/BUILD4_MEMORY_FIX.md").read_text()
for k, v in V.items():
    txt = txt.replace("{{" + k + "}}", str(v))
(M / "reports/BUILD4_MEMORY_FIX.md").write_text(txt)
print("left:", re.findall(r"\{\{[A-Z0-9_]+\}\}", txt))
