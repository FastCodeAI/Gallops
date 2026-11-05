#!/usr/bin/env python
"""Runs the S1 converter (unmodified, imported) on the S2 pilot scenes with its output roots pointed at the S2 folders.
  download:  conda drive-jepa python s2_s1_run.py download
  convert :  cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python analysis_3.0/Retrain-data/03_grade_candidates/s2_s1_run.py convert"""
import sys, runpy, pathlib
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s2_common as C2  # noqa: E402
import s1_common as C1  # noqa: E402  (S1 module object; only its path attributes are repointed, no S1 file is edited)
C1.HERE = HERE; C1.NAS = C2.NAS / "s1"; C1.MANIFEST = C2.MANIFEST
import pandas as pd  # noqa: E402
C2.assert_nurec_train(pd.read_csv(HERE / "pilot_clips.csv").uuid)
if sys.argv[1] == "download":
    runpy.run_path(str(C2.S1 / "s1_download.py"), run_name="__main__")
else:
    import s1_convert as S  # noqa: E402
    S.HERE = HERE; S.CACHE = C1.NAS / "cache"; S.MENUS = C1.NAS / "menus"; S.RECT = C1.NAS / "rectified"
    sys.argv = [sys.argv[0]]; S.main()
