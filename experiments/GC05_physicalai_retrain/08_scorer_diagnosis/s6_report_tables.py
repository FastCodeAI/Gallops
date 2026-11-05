#!/usr/bin/env python
"""Builds the markdown tables of REPORT.md section 4 from the CSVs written by s6_analyze.py (UNSEEN; SEEN is added when its dump is in)."""
import pandas as pd, pathlib
import sys
H = pathlib.Path(__file__).resolve().parent; SET = sys.argv[1]; U = SET.upper(); o = pd.read_csv(H / f"table_a_c_{SET}.csv"); mn = pd.read_csv(H / f"table_menu_{SET}.csv"); md = pd.read_csv(H / f"table_menu_distance_{SET}.csv"); mg = pd.read_csv(H / f"table_top2_margin_{SET}.csv"); K = {"unseen": "4.", "seen": "4.S"}[SET]
p = lambda r, k: f"{100 * r[k]:.1f} ({100 * r[k + '_lo']:.1f}–{100 * r[k + '_hi']:.1f})"; q = lambda r, k, d=3: f"{r[k]:.{d}f} ({r[k + '_lo']:.{d}f}–{r[k + '_hi']:.{d}f})"; L = []
def rows(df, name):
    for _, r in df.iterrows():
        L.append(f"| {name(r)} | {p(r, 'strict')} | {p(r, 'ge09')} | {p(r, 'within005')} | {q(r, 'regret')} | {q(r, 'regret_p90', 2)} | {p(r, 'within1p5m')} | " + (f"{q(r, 'rank_agreement_median', 2)}" if r.rank_agreement_median == r.rank_agreement_median else "—") + " |")
hd = "| {} | pick = true best, % | pick graded ≥ 0.9, % | pick within 0.05 of best, % | regret, mean | regret, 90th pct | pick within 1.5 m of human, % | rank agreement, median |\n|---|---|---|---|---|---|---|---|"
nm = lambda r: r.scorer.replace("baseline: ", "baseline — ").replace("ceiling: ", "ceiling — ") + ("" if r.selection in ("-", "raw score") else ", " + r.selection)
own = o[o.table == "own"]; n = own[own.slice == "all"].iloc[0]
L.append(f"**Table {K}1 — PRIMARY: each checkpoint whole, on its own menu. {U}, all {n.examples:,} graded examples, {n.clips} clips.** Brackets: 95 % interval, clip-level bootstrap.\n"); L.append(hd.format("scorer")); rows(own[own.slice == "all"], nm)
for sl in ("highway", "close-pass", "ordinary"):
    d = own[(own.slice == sl)]; n = d.iloc[0]; L.append(f"\n**Table {K}1 {sl} — {n.examples:,} examples, {n.clips} clips.**\n"); L.append(hd.format("scorer")); rows(d, nm)
sw = o[(o.table == "swap") & (o.slice == "all")]; L.append(f"\n**Table {K}2 — OFF-DESIGN (head swap): another checkpoint's score head on the pass-9 trunk, pass-9 menu and grades. {U}, all.** A head was never trained on this trunk's features; this is not how any checkpoint runs.\n"); L.append(hd.format("score head")); rows(sw, lambda r: r.scorer)
L.append(f"\n**Table {K}3 — what the menu allows ({U}).** Share of examples, %, with 95 % interval; last two columns are counts of the 32 plans.\n\n| menu | slice | examples | a plan graded ≥ 0.95 on the menu | a plan graded ≥ 0.9 | a plan within 1.5 m of the human | a best-graded plan within 1.5 m of the human | plans tied for best, mean | plans graded ≥ 0.9, mean |\n|---|---|---|---|---|---|---|---|---|")
for _, r in mn.iterrows(): L.append(f"| {r.menu} | {r.slice} | {r.examples:,} | {p(r, 'menu_has_grade_ge095')} | {p(r, 'menu_has_grade_ge09')} | {p(r, 'menu_has_plan_within_1p5m')} | {p(r, 'best_graded_plan_within_1p5m_any_tie')} | {r.plans_tied_for_best:.1f} | {r.plans_graded_ge09:.1f} |")
L.append(f"\n**Table {K}4 — top-2 margin (P-a5), {U}.** Score gap = top score minus second (0–1 scale); grade gap = true-grade difference between those two plans.\n\n| scorer | slice | examples | score gap, median | score gap < 0.01, % | of those: grade gap > 0.1, % | of those: top-scored is the worse by > 0.1, % | grade gap > 0.1, all examples, % |\n|---|---|---|---|---|---|---|---|")
for _, r in mg.iterrows(): L.append(f"| {r.scorer} | {r.slice} | {r.examples:,} | {q(r, 'score_gap_median', 4)} | {p(r, 'share_score_gap_lt_0p01')} | {p(r, 'of_near_ties_share_grade_gap_gt_0p1')} | {p(r, 'of_near_ties_share_top_is_the_worse_by_gt_0p1')} | {p(r, 'share_grade_gap_gt_0p1')} |")
L.append(f"\n**Table {K}5 — how different the menus are ({U}).** Mean distance from each plan of one menu to the nearest plan of the other (average over the 8 poses), metres.\n\n| pair | slice | examples | pass-9 plan → nearest plan of the other | other's plan → nearest pass-9 plan |\n|---|---|---|---|---|")
for _, r in md.iterrows(): L.append(f"| {r.pair} | {r.slice} | {r.examples:,} | {r.nearest_plan_dist_9_to_other_m:.2f} ({r.lo1:.2f}–{r.hi1:.2f}) | {r.nearest_plan_dist_other_to_9_m:.2f} ({r.lo2:.2f}–{r.hi2:.2f}) |")
open(H / f"section4_tables_{SET}.md", "w").write("\n".join(L) + "\n"); print(len(L), "lines")
