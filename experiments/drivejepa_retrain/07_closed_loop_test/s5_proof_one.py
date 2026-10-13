#!/usr/bin/env python
"""S5 TRIAL one-scene chain proof (media fork): for ONE finished 0.1.6 run whose menus and overlay video exist, write a PROVISIONAL
per-scene row with the 0.1.4b readout's own function (b014b_readout.arm_block / momentum_rows) into tables/standin_one/, build the review
page from it into review_standin_one/ and leave the headless check to the caller. The provisional tables are NOT the parent's tables.
  cd paddock-sid/Alpasim && alpasim/.venv/bin/python analysis_3.0/Retrain-data/07_closed_loop_test/s5_proof_one.py <short8>"""
import json, pathlib, subprocess, sys
import pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; B14 = HERE.parents[1] / "Re-ranker" / "baseline_014b"; ALP = HERE.parents[2]; sys.path.insert(0, str(B14)); import b014b_readout as BR
s = sys.argv[1]; run = ALP / "runs" / "drivejepa" / f"drivejepa_016_{s}"; T = HERE / "tables" / "standin_one"; T.mkdir(parents=True, exist_ok=True)
base = pd.read_csv(B14 / "tables" / "per_scene.csv").set_index("scene"); blk, keep = BR.arm_block(run, s); row = {"scene": s, "lane": base.loc[s, "lane"], **{c: base.loc[s, c] for c in base.columns if c.startswith("014b_")}, **{f"016_{k}": v for k, v in blk.items()}, "016_stall": False}
mr = pd.DataFrame(BR.momentum_rows(keep["sc"], s)); pd.DataFrame([row]).to_csv(T / "per_scene.csv", index=False); mr.to_csv(T / "replans_016.csv", index=False)
d = row["016_scene_score"] - row["014b_scene_score"]; g = "improved" if d >= 0.02 else ("decreased" if d <= -0.02 else "same")
G = {k: dict(title=t, lead="PROVISIONAL one-scene proof of the media chain; not the trial's result.", points=["provisional"], scenes=[s] if g == k else []) for k, t in (("improved", "Score improved"), ("decreased", "Score decreased"))}
G["same"] = dict(title="No change in score", lead="PROVISIONAL one-scene proof of the media chain; not the trial's result.", points=["provisional"], buckets=[dict(title="Provisional", note="", scenes=[s] if g == "same" else [])]); G["why"] = {s: "provisional row from b014b_readout.arm_block"}
json.dump(G, open(T / "review_groups.json", "w"), indent=1)
print(f"{s}: 0.1.4b score {row['014b_scene_score']:.3f} -> 0.1.6 {row['016_scene_score']:.3f} ({g}); replans {len(mr)}, changed {int(mr.changed.sum())}; codes {mr.code.value_counts().to_dict()}", flush=True)
sys.exit(subprocess.call([sys.executable, str(HERE / "s5_build_review.py"), "--tables", str(T), "--out", str(HERE / "review_standin_one"), "--scenes", s]))
