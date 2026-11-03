#!/usr/bin/env python
"""S8 L1, ADDED AFTER THE PRE-REGISTERED L1 TABLES WERE READ (descriptive, not part of P-L1). The sample showed one slot (8) repeating
the same shape (step 4 very fast, step 5 very slow). Question: is the uneven spacing a fixed signature of each menu slot? For UNSEEN
highway instants, all 32 plans: per slot, the mean profile of (step speed / plan mean speed - 1), and how well single plans of that slot
correlate with their slot's mean profile (leave-one-out not needed at these counts), against the correlation with the all-slot mean
profile. Also the same split by car speed, and which slots supply the best plan on no-good-plan instants."""
import pathlib, numpy as np, pandas as pd
from s7_common import load, step_speeds
HERE = pathlib.Path(__file__).resolve().parent; D = load("unseen", ["menu_9", "grade_9"]); per = pd.read_parquet(HERE / "l1_structure_per_instant.parquet"); assert (per.token.to_numpy() == D["token"]).all()
sp = step_speeds(D["menu_9"]); m = sp.mean(2, keepdims=True); rel = sp / np.maximum(m, 0.5) - 1; R = []; S = []
def corr_rows(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True); den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1)); return np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1), np.nan)
for lab, k in (("highway, car speed >= 20 m/s", (per.slice == "highway") & (per.v0 >= 20)), ("highway, car speed < 20 m/s", (per.slice == "highway") & (per.v0 < 20)), ("ordinary, car speed >= 10 m/s", (per.slice == "ordinary") & (per.v0 >= 10)), ("ordinary, car speed < 10 m/s", (per.slice == "ordinary") & (per.v0 < 10) & (per.v0 >= 2))):
    k = k.to_numpy(); x = rel[k]; slot_mean = x.mean(0); all_mean = x.reshape(-1, 8).mean(0); c_slot = corr_rows(x, slot_mean[None]); c_all = corr_rows(x, all_mean[None]); swing = (sp[k].max(2) - sp[k].min(2)); relsw = swing / np.maximum(m[k][..., 0], 0.5)
    R.append(dict(group=lab, instants=int(k.sum()), corr_with_own_slot_profile_median=float(np.nanmedian(c_slot)), corr_with_own_slot_profile_share_above_0p7=float(np.nanmean(c_slot > 0.7)), corr_with_all_slot_profile_median=float(np.nanmedian(c_all)), relative_swing_median_all_plans=float(np.median(relsw)),
                  slot_profile_size_median=float(np.median(slot_mean.max(1) - slot_mean.min(1))), slots_with_profile_size_over_0p3=int(((slot_mean.max(1) - slot_mean.min(1)) > 0.3).sum()), between_instant_spread_of_step_ratio_median=float(np.median(x.std(0)))))
    for j in range(32): S.append(dict(group=lab, slot=j, relative_swing_median=float(np.median(relsw[:, j])), profile_size=float(slot_mean[j].max() - slot_mean[j].min()), corr_with_own_profile_median=float(np.nanmedian(c_slot[:, j])), fastest_step=int(slot_mean[j].argmax() + 1), slowest_step=int(slot_mean[j].argmin() + 1),
                                      **{f"step_{t + 1}": round(float(slot_mean[j, t]), 3) for t in range(8)}, mean_grade=float(D["grade_9"][k][:, j, -1].mean()), comfort_pass=float(D["grade_9"][k][:, j, 7].mean()), plan_mean_speed_over_car_speed=float(np.median(m[k][:, j, 0] / np.maximum(per.v0.to_numpy()[k], 0.5)))))
ng = per[(per.slice == "highway") & (per.group == "no good plan")]; vc = ng.best_slot.value_counts(); print("best plan's slot on no-good-plan highway instants: top 8 slots hold %.1f %%" % (100 * vc.iloc[:8].sum() / len(ng))); print(vc.head(10).to_string())
pd.DataFrame(R).to_csv(HERE / "l1_posthoc_slot_signature.csv", index=False); pd.DataFrame(S).to_csv(HERE / "l1_posthoc_slot_profiles.csv", index=False); pd.set_option("display.width", 300, "display.max_rows", 200); print(pd.DataFrame(R).round(3).T.to_string()); s = pd.DataFrame(S); print(s[s.group == "highway, car speed >= 20 m/s"].drop(columns="group").round(2).to_string(index=False))
