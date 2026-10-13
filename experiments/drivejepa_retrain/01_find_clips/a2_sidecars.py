"""A2: download clip_ratings csv + README (small) to NAS; summarise. labels.json: none in tree (see a1)."""
import json, hashlib, shutil, collections, common
from huggingface_hub import hf_hub_download
import pandas as pd
tree = json.load(open(common.HERE / "a1_tree_26.04.json"))
labels = [r for r in tree if r["path"].endswith("labels.json")]
print("labels.json in 26.04 tree:", len(labels))
common.NAS.mkdir(parents=True, exist_ok=True)
for fname in ["clip_ratings_26.04.csv", "README.md", ".gitattributes"]:
    p = hf_hub_download(common.REPO, fname, repo_type="dataset", local_dir=common.NAS / "root_files")
    b = open(p, "rb").read()
    common.manifest_add("-", fname, len(b), p, hashlib.sha256(b).hexdigest())
    print("downloaded", p, len(b), "bytes")
df = pd.read_csv(common.NAS / "root_files" / "clip_ratings_26.04.csv")
print("clip_ratings shape", df.shape); print("columns", list(df.columns)); print(df.head(3).to_string())
uu = set(open(common.HERE / "a1_uuids_26.04.txt").read().split())
cid = df[df.columns[0]].astype(str)
print("ratings uuids", cid.nunique(), "in tree", len(set(cid) & uu), "tree not in ratings", len(uu - set(cid)), "ratings not in tree", len(set(cid) - uu))
for c in df.columns[1:]:
    if df[c].dtype == object or df[c].nunique() < 20:
        print("\n", c, dict(collections.Counter(df[c].astype(str))))
    else:
        print("\n", c, df[c].describe().to_string())
