#!/usr/bin/env python
"""S10 task 4, part B.3: fetch the first N scenes of fresh20.csv into the trial's scene cache, one at a time.

Mechanism = the wizard's (alpasim_wizard/scenes/sceneset.py:459): huggingface_hub.hf_hub_download(repo nvidia/PhysicalAI-Autonomous-Vehicles-NuRec,
repo_type dataset, filename = the `path` of the scene's 26.1.112 row in sim_scenes.csv, revision = its hf_revision), file then renamed to
all-usdzs/<uuid>.usdz. Token: HF_TOKEN as exported by Alpasim/alpasim_env.sh (never printed). Differences from the wizard, on purpose:
one file at a time (the wizard runs up to 5), plain HTTP (xet / hf_transfer off: one connection), a persistent staging folder so an
interrupted file resumes, a free-space floor, and the archive is NOT opened (the wizard reads metadata.yaml to compare the uuid); instead
the size and sha256 are compared with the repository's own record for that path.
Usage: s10_fresh_fetch.py FRESH20_CSV N HASHES_CSV [--list]      (--list: only query sizes and print the plan)"""
import os, sys, time, json, hashlib, shutil, pathlib
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import csv
from huggingface_hub import HfApi, hf_hub_download
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"; CACHE = pathlib.Path("{runs}/alpasim_bulk/nre-artifacts/all-usdzs"); STAGE = CACHE / ".s10_fresh_staging"; FLOOR = 40e9
rows = list(csv.DictReader(open(sys.argv[1])))[:int(sys.argv[2])]; HASHES = pathlib.Path(sys.argv[3]); LIST = "--list" in sys.argv
say = lambda *x: print(time.strftime("[%Y-%m-%dT%H:%M:%S]"), *x, flush=True)
assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set (source Alpasim/alpasim_env.sh): the dataset is gated"
assert all(r["nre_version_string"].startswith("26.1.112") for r in rows)
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
api = HfApi(); info = {}
for rev in sorted(set(r["hf_revision"] for r in rows)):
    for x in api.get_paths_info(REPO, [r["hf_path"] for r in rows if r["hf_revision"] == rev], revision=rev, repo_type="dataset"): info[x.path] = (x.size, x.lfs.sha256 if x.lfs else None)
missing = [r["hf_path"] for r in rows if r["hf_path"] not in info]; assert not missing, ("not in the repository at that revision", missing)
plan = [dict(order=int(r["order"]), scene_id=r["scene_id"], uuid=r["uuid"], hf_path=r["hf_path"], hf_revision=r["hf_revision"], target=str(CACHE / f"{r['uuid']}.usdz"), expected_bytes=info[r["hf_path"]][0], expected_sha256=info[r["hf_path"]][1]) for r in rows]
if LIST: print(json.dumps(dict(plan=plan, total_bytes=sum(p["expected_bytes"] for p in plan), bulk_free_bytes=shutil.disk_usage(CACHE).free), indent=1)); sys.exit(0)
done = set()
if HASHES.exists(): done = {r["uuid"] for r in csv.DictReader(open(HASHES)) if r["sha256_matches_repository"] == "True"}
else: open(HASHES, "w").write("order,scene_id,uuid,path,bytes,sha256,expected_bytes,expected_sha256,sha256_matches_repository,completed_at,download_s\n")
STAGE.mkdir(exist_ok=True); say(f"start: {len(plan)} scenes, {sum(p['expected_bytes'] for p in plan) / 1e9:.2f} GB, bulk free {shutil.disk_usage(CACHE).free / 1e9:.1f} GB, already recorded {len(done)}")
for p in plan:
    tgt = pathlib.Path(p["target"]); t0 = time.time()
    if p["uuid"] in done and tgt.exists(): say(f"{p['order']} {p['uuid']}: already complete, skipped"); continue
    if not (tgt.exists() and tgt.stat().st_size == p["expected_bytes"]):
        assert not tgt.exists(), f"{tgt} exists with a different size; not overwriting"
        staged = STAGE / p["hf_path"]; inc = list((STAGE / ".cache").rglob("*.incomplete")); have = staged.stat().st_size if staged.exists() else sum(f.stat().st_size for f in inc)
        free = shutil.disk_usage(CACHE).free
        if free - (p["expected_bytes"] - have) < FLOOR: say(f"STOP: bulk disk free {free / 1e9:.1f} GB would drop under {FLOOR / 1e9:.0f} GB with {p['uuid']}"); sys.exit(4)
        say(f"{p['order']} {p['scene_id']} -> {tgt.name}: {p['expected_bytes'] / 1e9:.2f} GB, {have / 1e9:.2f} GB already staged")
        for attempt in range(1, 31):
            try: got = hf_hub_download(repo_id=REPO, repo_type="dataset", local_dir=str(STAGE), filename=p["hf_path"], revision=p["hf_revision"]); break
            except Exception as e:
                say(f"  attempt {attempt} failed: {type(e).__name__}: {str(e)[:300]}")
                if type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError", "LocalTokenNotFoundError") or "401" in str(e) or "403" in str(e): sys.exit(5)
                time.sleep(60)
        else: say("STOP: 30 attempts failed"); sys.exit(6)
        assert os.path.getsize(got) == p["expected_bytes"], (os.path.getsize(got), p["expected_bytes"]); os.rename(got, tgt)
    h = sha256(tgt); ok = h == p["expected_sha256"]
    with open(HASHES, "a") as f: f.write(f"{p['order']},{p['scene_id']},{p['uuid']},{tgt},{tgt.stat().st_size},{h},{p['expected_bytes']},{p['expected_sha256']},{ok},{time.strftime('%Y-%m-%dT%H:%M:%S+05:30')},{time.time() - t0:.0f}\n")
    say(f"  done {tgt.name}: {tgt.stat().st_size} bytes, sha256 {h}, matches repository {ok}, {time.time() - t0:.0f} s")
    if not ok: say("STOP: sha256 differs from the repository record; file left in place, not overwritten"); sys.exit(7)
say("ALL DONE")
