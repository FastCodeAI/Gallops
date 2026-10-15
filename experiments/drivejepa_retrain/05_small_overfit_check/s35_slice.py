"""S3.5 STEP 2: seeded 1 % slice of S3 train BY CLIP, stratified by type (and, inside 'ordinary', by source); held-out = the whole
S3.1 validation manifest (608 rows, already 1 %-scale; not sub-sampled). Asserts: split guards, disjoint chunks (S6)."""
import sys, random, pathlib, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; S3 = HERE.parent / "04_training_set"; sys.path.insert(0, str(S3))
import s3_common as C3
SEED = 20261007; tr = pd.read_parquet(S3 / "manifest.parquet"); va = pd.read_parquet(S3 / "val" / "manifest_val.parquet")
rng = random.Random(SEED); pick = []; rows = []
cl = tr.groupby("clip_uuid").agg(type=("type", "first"), source=("source", "first"), group=("group", "first"), n=("sample_id", "size")).reset_index()
for (typ, src), g in cl.groupby(["type", "source"]):
    k = max(1, int(round(0.01 * len(g)))); p = rng.sample(sorted(g.clip_uuid), k); pick += p
    rows.append(dict(type=typ, source=src, clips_in_train=len(g), clips_in_slice=k))
sl = tr[tr.clip_uuid.isin(pick)].copy(); C3._assert_train(set(sl.clip_uuid)); C3._assert_train(set(sl[sl.source == "nurec"].clip_uuid), nurec=True)
C3.assert_val(set(va.clip_uuid)); C3.assert_val(set(va[va.source == "nurec"].clip_uuid), nurec=True)
assert set(va.split) == {"val"} and set(sl.split) == {"train"}
assert not (set(tr.group) & set(va.group)), "S6: a val chunk appears in the FULL train chunk list"       # stronger than slice-vs-val
assert not (set(sl.group) & set(va.group)) and not (set(sl.clip_uuid) & set(va.clip_uuid)) and not (set(sl.sample_id) & set(va.sample_id))
sl.to_parquet(HERE / "slice_train.parquet", index=False); va.to_parquet(HERE / "heldout_val.parquet", index=False)
t = pd.DataFrame(rows).merge(sl.groupby(["type", "source"]).agg(samples=("sample_id", "size"), has_label2=("has_label2", "sum")).reset_index(), on=["type", "source"])
print("seed", SEED, "| train manifest", len(tr), "rows,", tr.clip_uuid.nunique(), "clips,", tr.group.nunique(), "chunks"); print(t.to_string(index=False))
print("slice:", sl.clip_uuid.nunique(), "clips,", len(sl), "samples (", round(100 * len(sl) / len(tr), 2), "% of samples ),", int(sl.has_label2.sum()), "with label 2,", sl.group.nunique(), "chunks")
print("held-out:", va.clip_uuid.nunique(), "clips,", len(va), "samples,", int(va.has_label2.sum()), "with label 2,", va.group.nunique(), "chunks |", va.groupby(["type", "source"]).size().to_dict())
print("S6 asserts passed: 0 shared chunks (slice vs val, and full train", tr.group.nunique(), "chunks vs", va.group.nunique(), "val chunks), 0 shared clips, 0 shared samples")
