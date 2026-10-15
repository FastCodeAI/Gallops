#!/usr/bin/env python
"""S10 task 6: UNSEEN tripwire limits of PREDICTIONS_phase1.md from the pass-9 batch-1 tables (s10_base_tables.py on NAS/12_second_training_run/base_b1),
and the batch-1 figures beside the batch-8 audit figures (task 4's run of the same script on S6's dump). CPU, reads CSV/JSON only.
Usage: s10_phase1_limits.py B1_TABLE_DIR B8_TABLE_DIR OUT_DIR   -> S10/phase1_limits.json, OUT_DIR/base_b1_vs_b8.csv"""
import sys, json, pathlib, datetime, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; B1, B8, OUT = map(pathlib.Path, sys.argv[1:4])
t1 = pd.read_csv(B1 / "base_tables_base_b1.csv"); t8 = pd.read_csv(B8 / "base_tables_s6_batch8.csv"); k1 = json.load(open(B1 / "base_tables_base_b1.json")); k8 = json.load(open(B8 / "base_tables_s6_batch8.json"))
s1 = pd.read_csv(B1 / "slot_comfort_highway_ge20_base_b1.csv"); s8 = pd.read_csv(B8 / "slot_comfort_highway_ge20_s6_batch8.csv")
GOOD = "share of instants with a menu plan graded >= 0.95"; HUM = "human-path term (L1 x,y,heading; mean 8 poses; best of 32), mean"; HW20 = "highway, car >= 20 m/s"
g = lambda t, m, s: t[(t.measure == m) & (t["slice"] == s)].iloc[0]
lim = dict(written=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), source="pass 9 at batch 1 on UNSEEN graded (NAS/12_second_training_run/base_b1, 11,842 instants), tables by S10/s10_base_tables.py",
           definition="PREDICTIONS_phase1.md: W1 good-plan rate on ordinary and close-pass not more than 2 points under the batch-1 pass-9 value; W2 human-path error on highway, close-pass, ordinary at most 1.05 x the batch-1 pass-9 value",
           W1_good_plan_share_min_pct={s: 100 * g(t1, GOOD, s).value - 2 for s in ("ordinary", "close-pass")}, W1_base_pct={s: 100 * g(t1, GOOD, s).value for s in ("ordinary", "close-pass")},
           W2_human_path_term_max={s: 1.05 * g(t1, HUM, s).value for s in ("highway", "close-pass", "ordinary")}, W2_base={s: g(t1, HUM, s).value for s in ("highway", "close-pass", "ordinary")})
json.dump(lim, open(HERE / "phase1_limits.json", "w"), indent=1)
rows = []
def add(name, m, s, scale=1.0):
    a, b = g(t1, m, s), g(t8, m, s); rows.append(dict(quantity=name, slice=s, batch1=scale * a.value, batch1_lo95=scale * a.lo95, batch1_hi95=scale * a.hi95, batch8=scale * b.value, batch8_lo95=scale * b.lo95, batch8_hi95=scale * b.hi95, instants=int(a.instants), clips=int(a.clips)))
for s in ("highway", "close-pass", "ordinary", "all", HW20): add("good-plan share, %", GOOD, s, 100)
add("swing, all 32 plans, median m/s", "swing, all 32 plans, median m/s", HW20); add("swing, best plan, median m/s", "swing, best plan (S7 tie-break), median m/s", HW20); add("swing, best plan on no-good-plan instants, median m/s", "swing, best plan on no-good-plan instants, median m/s", HW20)
add("swing, best plan on no-good-plan instants, median m/s", "swing, best plan on no-good-plan instants, median m/s", "highway")
rows.append(dict(quantity="slots passing comfort on > 25 % of instants (of 32)", slice=HW20, batch1=int((s1.comfort_pass_rate > 0.25).sum()), batch8=int((s8.comfort_pass_rate > 0.25).sum())))
rows.append(dict(quantity="highest slot comfort pass rate, %", slice=HW20, batch1=100 * s1.comfort_pass_rate.max(), batch8=100 * s8.comfort_pass_rate.max()))
for s in ("highway", "close-pass", "ordinary", "all", HW20): add("human-path term", HUM, s)
for s in ("all", "highway", "close-pass", "ordinary"):
    add("scorer picks graded 0, %", "scorer picks graded exactly 0, share", s, 100); add("regret, mean", "scorer regret (best grade - pick grade), mean", s); add("rank agreement, median", "rank agreement (Spearman), median over instants", s)
T = pd.DataFrame(rows); T.to_csv(OUT / "base_b1_vs_b8.csv", index=False); pd.set_option("display.width", 250); print(T.round(4).to_string()); print(json.dumps(lim, indent=1)); print(json.dumps(k1, indent=1))
