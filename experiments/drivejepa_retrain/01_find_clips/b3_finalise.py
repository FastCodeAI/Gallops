"""STEP B final (decision by Sid, 2026-10-02): exam = AlpaSim suite public_2601 (913) U our 30 diagnosed scenes.
Writes two views: nurec_2604_exclusions.txt (exam clips present as 26.04 folders) and raw_exclusions.txt (all exam clip uuids)."""
import csv, pathlib, common
ALP = pathlib.Path("{runs}/paddock_sid/Alpasim")
p2601 = {r["scene_id"].replace("clipgt-", "") for r in csv.DictReader(open(ALP/"alpasim/data/scenes/sim_suites.csv")) if r["test_suite_id"] == "public_2601"}
ours = {r["uuid"] for r in csv.DictReader(open(common.HERE/"b2_our30.csv"))}
tree = set(open(common.HERE/"a1_uuids_26.04.txt").read().split())
invalid = set(open(common.HERE/"known_invalid_2604.txt").read().split())
assert len(p2601) == 913 and len(ours) == 30
raw = p2601 | ours; nurec = raw & tree; pool = tree - raw - invalid
print("public_2601", len(p2601), "our30", len(ours), "our30 in public_2601", len(ours & p2601), "raw", len(raw), "nurec_2604", len(nurec), "invalid", len(invalid), "pool", len(pool), "tree", len(tree))
assert len(raw) == 913 and len(nurec) == 159 and len(pool) == 1447 and len(tree) == 1607 == len(nurec) + len(pool) + len(invalid)
H = "# FINAL 2026-10-02 (Sid): exam = AlpaSim public_2601 (913; alpasim/data/scenes/sim_suites.csv:2-914) U our 30 diagnosed scenes. Key = clip uuid.\n"
open(common.HERE/"raw_exclusions.txt", "w").write(H + "# view: raw_exclusions — every exam clip uuid, for any raw-dataset step\n" + "\n".join(sorted(raw)) + "\n")
open(common.HERE/"nurec_2604_exclusions.txt", "w").write(H + "# view: nurec_2604_exclusions — exam clips that exist as sample_set/26.04_release folders\n" + "\n".join(sorted(nurec)) + "\n")
open(common.HERE/"exam_exclusions.txt", "w").write(H + "# identical to raw_exclusions.txt (kept for the Run-1 scripts)\n" + "\n".join(sorted(raw)) + "\n")
open(common.HERE/"training_pool.txt", "w").write("\n".join(sorted(pool)) + "\n")
assert sorted(pool) == open(common.HERE/"training_pool_candA.txt").read().split()
