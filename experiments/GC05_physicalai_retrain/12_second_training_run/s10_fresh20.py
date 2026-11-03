#!/usr/bin/env python
"""S10 task 4, part B.1: fix the 20 fresh public_2601 scenes. Pool = the audit's 852 never-used scenes, minus any scene the audit's
913-row table shows mentioned anywhere other than a suite-wide listing (its `mentions` entry `pool_listing`; every one of the 852 carries
that entry, so "non-empty" taken literally would empty the pool). Sorted scene ids, numpy default_rng(20261009).choice(ids, 20,
replace=False), draw order kept. Artifact row = the 26.1.112 row of alpasim/data/scenes/sim_scenes.csv (the version of the 30 trial scenes).
Usage: s10_fresh20.py AUDIT_REPORT_DIR SIM_SCENES_CSV OUT_CSV"""
import sys, numpy as np, pandas as pd
T, SC, OUT = sys.argv[1:4]
p = pd.read_csv(f"{T}/D3_pool_all_untouched_incl_not_local.csv"); a = pd.read_csv(f"{T}/D3_public_2601_all_913.csv").set_index("scene_id")
ment = a.loc[p.scene_id, "mentions"].fillna("").map(lambda s: sorted(set(x for x in s.split(";") if x and x != "pool_listing")))
keep = p[(ment.map(len) == 0).to_numpy() & ~a.loc[p.scene_id, "local_any_version"].to_numpy()]; ids = np.array(sorted(keep.scene_id)); assert len(set(ids)) == len(ids)
print("pool file", len(p), "| literally empty mentions", int((a.loc[p.scene_id, "mentions"].fillna("") == "").sum()), "| mentioned beyond pool_listing", int((ment.map(len) > 0).sum()), "| pool used", len(ids), "| numpy", np.__version__)
draw = np.random.default_rng(20261009).choice(ids, 20, replace=False)
s = pd.read_csv(SC); s = s[s.nre_version_string.str.startswith("26.1.112")].set_index("scene_id"); assert s.index.is_unique
r = s.loc[draw].reset_index(); assert (r.uuid.to_numpy() == a.loc[draw, "uuid"].to_numpy()).all() and (r.artifact_repository == "huggingface").all()
out = pd.DataFrame(dict(order=np.arange(1, 21), scene_id=draw, suite="public_2601", uuid=r.uuid, nre_version_string=r.nre_version_string, hf_revision=r.hf_revision, hf_path=r.path)); out.to_csv(OUT, index=False); print(out.to_string())
