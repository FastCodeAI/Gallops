#!/usr/bin/env python
"""S2.1 D2: rerun the S1 converter on the SAME 20 S1 pilot clips with the bonnet mask (default on); inputs from the S1 NAS root,
outputs under the S2 NAS root. cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python <this>"""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s2_common as C2; import s1_common as C1; import s1_convert as S
OUT = C2.NAS / "s1_masked"; S.HERE = HERE / "s21_masked"; S.CACHE = OUT / "cache"; S.MENUS = OUT / "menus"; S.RECT = OUT / "rectified"
sys.argv = [sys.argv[0]]; S.main()
