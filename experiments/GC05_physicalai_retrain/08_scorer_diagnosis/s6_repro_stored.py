#!/usr/bin/env python
"""S6 Part 0.4, stored half: S4's headline numbers recomputed from S4's own per-example readouts on val_full (no model run)."""
import pathlib, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; RD = HERE.parent / "06_first_training_run/run/readouts"; rows = []
for p, name in [(0, "shipped (pass 0)"), (1, "pass 1"), (9, "pass 9")]:
    d = pd.read_parquet(RD / f"val_full_pass{p:02d}.parquet"); g = d[d.has_label2]; cp = d[d.type == "B_pass"]
    rows.append(dict(scorer=name, graded=len(g), V2_pick_is_best=g.top1.mean(), V2_highway=g[g.type == "A"].top1.mean(), V2_closepass=g[g.type == "B_pass"].top1.mean(), V2_ordinary=g[g.type == "ordinary"].top1.mean(),
                     rank_agreement=g.rank_corr.mean(), closepass_examples=len(cp), V3_pick_within_1p5m=(cp.dist4 <= 1.5).mean(), menu_has_plan_within_1p5m=(cp.best_dist4 <= 1.5).mean()))
t = pd.DataFrame(rows); t.to_csv(HERE / "repro_stored.csv", index=False); print(t.round(4).T.to_string())
