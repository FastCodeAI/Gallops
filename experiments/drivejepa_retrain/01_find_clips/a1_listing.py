"""A1: list every entry under sample_set/26.04_release (metadata only)."""
import json, collections, common
from huggingface_hub import HfApi
api = HfApi()
rows = []
for e in api.list_repo_tree(common.REPO, repo_type="dataset", path_in_repo=common.RELEASE, recursive=True):
    rows.append({"path": e.path, "type": type(e).__name__, "size": getattr(e, "size", None),
                 "lfs_sha": (e.lfs.sha256 if getattr(e, "lfs", None) else None)})
json.dump(rows, open(common.HERE / "a1_tree_26.04.json", "w"))
print("entries", len(rows))
depths = collections.Counter(r["path"].count("/") for r in rows)
print("depth histogram", dict(depths))
dirs = [r for r in rows if r["type"].endswith("Folder")]
files = [r for r in rows if not r["type"].endswith("Folder")]
print("folders", len(dirs), "files", len(files))
# uuid folders directly under release
uuids = sorted({r["path"].split("/")[2] for r in rows if r["path"].count("/") >= 3})
print("distinct uuid folders", len(uuids))
byname = collections.Counter(r["path"].split("/")[-1] if r["path"].count("/")==3 else "DEEPER:"+r["path"].split("/")[3] for r in files)
for k, v in byname.most_common(40): print(f"{v:6d}  {k}")
# usdz sizes
us = [r["size"] for r in files if r["path"].endswith(".usdz") and r["size"]]
import statistics
print("usdz n", len(us), "min/med/max MB", round(min(us)/1e6,1), round(statistics.median(us)/1e6,1), round(max(us)/1e6,1), "total GB", round(sum(us)/1e9,1))
open(common.HERE / "a1_uuids_26.04.txt", "w").write("\n".join(uuids) + "\n")
