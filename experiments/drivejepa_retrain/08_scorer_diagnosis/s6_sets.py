#!/usr/bin/env python
"""S6 Part 0.3: UNSEEN and SEEN sets, exam-overlap check, slice counts. Read-only on 04_training_set and 01_find_clips; writes only here."""
import pathlib, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; R = HERE.parent; S3 = R / "04_training_set"; S0 = R / "01_find_clips"; ALP = R.parents[1]
SEED = 20261007; SLICE = {"A": "highway", "B_pass": "close-pass", "B_raw": "pull-out", "ordinary": "ordinary"}
ids = lambda p: {l.strip() for l in open(p) if l.strip() and not l.startswith("#")}
suites = pd.read_csv(ALP / "alpasim/data/scenes/sim_suites.csv"); suites["clip_id"] = suites.scene_id.str.replace("clipgt-", "", regex=False)
exam = set(suites[suites.test_suite_id == "public_2601"].clip_id) | ids(S0 / "exam_exclusions_strict_our30.txt"); assert ids(S0 / "exam_exclusions.txt") == exam, "exam list differs from 01_find_clips/exam_exclusions.txt"
tr = pd.read_parquet(S3 / "manifest.parquet"); va = pd.read_parquet(S3 / "val_full/manifest_val_full.parquet"); vs = pd.read_parquet(S3 / "val/manifest_val.parquet")
un = va[va.has_label2].copy(); pool = tr[tr.has_label2].copy(); n = len(un)
# SEEN: every clip contributes in proportion to its graded examples (largest-remainder rounding), examples drawn without replacement
rng = np.random.default_rng(SEED); cnt = pool.groupby("clip_uuid").size().sort_index(); q = cnt * n / cnt.sum(); k = np.floor(q).astype(int)
extra = (q - k).sort_values(ascending=False, kind="stable").index[: n - int(k.sum())]; k[extra] += 1; assert int(k.sum()) == n and (k <= cnt).all()
seen = pd.concat([g.iloc[np.sort(rng.choice(len(g), size=int(k[u]), replace=False))] for u, g in pool.sort_values("sample_id").groupby("clip_uuid") if k[u] > 0])
rows = []
for name, d in [("UNSEEN", un), ("SEEN", seen), ("SEEN pool (all graded training examples)", pool), ("val small (graded)", vs[vs.has_label2])]:
    ov = set(d.clip_uuid) & exam; d = d[~d.clip_uuid.isin(exam)]
    for t, g in d.groupby("type"): rows.append(dict(set=name, slice=SLICE[t], type=t, examples=len(g), clips=g.clip_uuid.nunique(), exam_clips_dropped=len(ov)))
    rows.append(dict(set=name, slice="all", type="all", examples=len(d), clips=d.clip_uuid.nunique(), exam_clips_dropped=len(ov)))
    assert not ov, f"{name}: exam overlap {len(ov)}"
assert not (set(un.clip_uuid) & set(tr.clip_uuid)) and not (set(un.group) & set(tr.group)) and seen.sample_id.is_unique and un.sample_id.is_unique
t = pd.DataFrame(rows); t.to_csv(HERE / "sets_table.csv", index=False); print(t.to_string(index=False))
un.assign(slice=un.type.map(SLICE)).to_parquet(HERE / "set_unseen.parquet", index=False); seen.assign(slice=seen.type.map(SLICE)).to_parquet(HERE / "set_seen.parquet", index=False)
print("seed", SEED, "| exam ids", len(exam), "| train manifest clips", tr.clip_uuid.nunique(), "| graded-but-not-has_label2 none by construction")
print("ungraded slices: UNSEEN pull-out graded =", int((un.type == "B_raw").sum()), "| SEEN pull-out graded =", int((seen.type == "B_raw").sum()))
print("info only: clips also listed anywhere in sim_suites.csv (any suite): UNSEEN", len(set(un.clip_uuid) & set(suites.clip_id)), "SEEN", len(set(seen.clip_uuid) & set(suites.clip_id)), "by suite", {s: len(set(un.clip_uuid) & set(g.clip_id)) for s, g in suites.groupby("test_suite_id")})
# example spacing inside a clip (for the momentum-aware selection, which needs the previous plan)
d = un.sort_values(["clip_uuid", "anchor_t"]).groupby("clip_uuid").anchor_t.diff().dropna(); print("UNSEEN anchor spacing within a clip, s: median %.3f, min %.3f, share == 0.5 s: %.3f" % (d.median(), d.min(), float(np.isclose(d, 0.5).mean())))
