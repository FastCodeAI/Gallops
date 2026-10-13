"""STEP B: official AlpaSim suites (B1, verified from disk) x our 30 diagnosed scenes (B2) -> exclusion candidates (B3)."""
import csv, re, collections, pathlib, common
ALP = pathlib.Path("{runs}/paddock_sid/Alpasim")
SUITES = ALP / "alpasim/data/scenes/sim_suites.csv"
SCENES = ALP / "alpasim/data/scenes/sim_scenes.csv"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
suites = collections.defaultdict(set)
for r in csv.DictReader(open(SUITES)):
    suites[r["test_suite_id"]].add(r["scene_id"])
cat = list(csv.DictReader(open(SCENES)))
folder_of = {}   # (scene_id, hf_revision) -> folder uuid in path
for r in cat:
    folder_of[(r["scene_id"], r["hf_revision"])] = r["path"].split("/")[2]
tree = set(open(common.HERE / "a1_uuids_26.04.txt").read().split())
# --- B2: our 30 ---
ours = {}
for d in sorted((ALP / "runs/drivejepa").glob("drivejepa_014_????????")):
    ours[UUID.search(open(d / "wizard-config.yaml").read()).group(0)] = ("pilot10", str(d))
for d in sorted((ALP / "analysis_2.0/drivejepa_014_ext20").glob("run_*")):
    ours[UUID.search(open(d.resolve() / "wizard-config.yaml").read()).group(0)] = ("ext20", str(d.resolve()))
census = {l.split(",")[0] for l in open(common.HERE / "b2_census_scenes.txt")}
assert len(ours) == 30, len(ours)
assert {u[:8] for u in ours} == census, "census.csv scenes != run dirs"
print("B2: our 30 scenes =", len(ours), "; census.csv short ids match run dirs:", {u[:8] for u in ours} == census)
rows = []
for u, (lane, path) in sorted(ours.items()):
    sid = f"clipgt-{u}"
    r = dict(uuid=u, lane=lane, in_2601=sid in suites["public_2601"], in_2601_vm=sid in suites["public_2601_video_model"],
             in_2604=sid in suites["public_2604"], folder_2604=folder_of.get((sid, "26.04"), ""), in_2604_tree=u in tree,
             artifact_2604=next((r["uuid"] for r in cat if r["scene_id"]==sid and r["hf_revision"]=="26.04"), ""))
    rows.append(r)
with open(common.HERE / "b2_our30.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for k in ["in_2601", "in_2601_vm", "in_2604", "in_2604_tree"]:
    print(f"  our30 {k}: {sum(r[k] for r in rows)}")
print("  our30 folder==uuid for 26.04 rows:", all((r['folder_2604'] in ('', r['uuid'])) for r in rows))
# --- suite folders in 26.04 tree ---
def folders_2604(suite):
    out = set()
    for sid in suites[suite]:
        u = sid.replace("clipgt-", "")
        if u in tree: out.add(u)
    return out
# check 26.04 catalog: scene uuid == folder uuid for all rows?
mism = [(r["scene_id"], r["path"]) for r in cat if r["hf_revision"]=="26.04" and r["scene_id"].replace("clipgt-","") != r["path"].split("/")[2]]
print("26.04 catalog rows where scene uuid != folder uuid:", len(mism), mism[:3])
cat2604 = {r["path"].split("/")[2] for r in cat if r["hf_revision"]=="26.04"}
print("26.04 catalog folders", len(cat2604), "== tree:", cat2604 == tree)
f2604 = folders_2604("public_2604"); f2601 = folders_2604("public_2601"); f2601vm = folders_2604("public_2601_video_model")
invalid = tree - f2604
print("public_2604 folders in tree:", len(f2604), "; tree minus public_2604 (known-invalid):", sorted(invalid))
print("public_2601 scene ids present as 26.04 folders:", len(f2601), "; video_model subset:", len(f2601vm))
# --- B3 candidates ---
ours_set = set(ours)
candA = {s.replace("clipgt-","") for s in suites["public_2601"]} | ours_set          # all 913 26.01 ids + our 30 (uuid namespace = clip uuid)
candA_tree = candA & tree
print(f"CANDIDATE A (public_2601 U our30): {len(candA)} ids; in 26.04 tree {len(candA_tree)}; training pool = {len(tree - candA - invalid)}")
candS = ours_set
print(f"STRICT (no eval-labelled list found -> our30 only): {len(candS)} ids; in tree {len(candS & tree)}; pool = {len(tree - candS - invalid)}")
hdr = ("# CANDIDATE A — public_2601 (913 scene ids, AlpaSim data/scenes/sim_suites.csv) U our 30 diagnosed scenes.\n"
       "# NOT an eval/val/test-labelled list; pending Sid's confirmation (see REPORT.md STEP B GATE). uuid namespace = clip uuid (clipgt-<uuid>).\n")
open(common.HERE / "exam_exclusions.txt", "w").write(hdr + "\n".join(sorted(candA)) + "\n")
open(common.HERE / "exam_exclusions_strict_our30.txt", "w").write("# our 30 diagnosed scenes only\n" + "\n".join(sorted(candS)) + "\n")
open(common.HERE / "known_invalid_2604.txt", "w").write("\n".join(sorted(invalid)) + "\n")
open(common.HERE / "training_pool_candA.txt", "w").write("\n".join(sorted(tree - candA - invalid)) + "\n")
