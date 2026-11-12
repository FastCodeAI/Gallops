"""Build step 2b: run the existing held-out stage scripts (NAS/heldout/src/b3_download.py, b3_convert.py, b3_manifest.py) UNCHANGED on the
fresh slice by pointing their shared config (b3_common: OUT, RAW, MANIFEST) at {experiment_store}/raw_heldout before the script
runs. RAW stays the old held-out's raw folder, read-only: it already holds the egomotion.offline zip of every strict-pool chunk
(b3_mine.py stage A), and the converter reads egomotion from there (s1_convert.py:86). Nothing is written under NAS/heldout.
Usage: s2_run_b3.py <script.py> [args...]"""
import sys, runpy, pathlib
SRC=pathlib.Path("{retrain_store}/heldout/src"); sys.path.insert(0, str(SRC))
import b3_common as B
B.OUT=pathlib.Path("{experiment_store}/raw_heldout"); B.MANIFEST=B.OUT/"download_manifest.csv"
assert B.RAW==pathlib.Path("{retrain_store}/heldout/raw")   # read-only egomotion source
(B.OUT/"logs").mkdir(parents=True, exist_ok=True)
script=sys.argv[1]; sys.argv=[str(SRC/script)]+sys.argv[2:]
runpy.run_path(str(SRC/script), run_name="__main__")
