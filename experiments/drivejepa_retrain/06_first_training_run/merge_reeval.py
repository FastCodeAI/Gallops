#!/usr/bin/env python
"""Merge val_full evaluations done by the side process (run_reeval/) into run/: readout files are copied, evals.csv rows appended when
run/ has no row for that (set, pass). Run only while the trainer is mid-pass (it rewrites run/evals.csv at pass ends)."""
import shutil, pathlib, pandas as pd
H = pathlib.Path(__file__).resolve().parent; A, B = H / "run", H / "run_reeval"
ea = pd.read_csv(A / "evals.csv"); eb = pd.read_csv(B / "evals.csv"); eb = eb[eb.set == "val_full"]; have = set(zip(ea.set, ea.passno)); new = eb[[(s, p) not in have for s, p in zip(eb.set, eb.passno)]]
for p in new.passno:
    for f in (B / "readouts").glob(f"val_full_pass{int(p):02d}*"): shutil.copy2(f, A / "readouts" / f.name)
pd.concat([ea, new], ignore_index=True).to_csv(A / "evals.csv.tmp", index=False); (A / "evals.csv.tmp").replace(A / "evals.csv"); print("merged passes", sorted(int(p) for p in new.passno), "| run/ now has val_full for", sorted(int(p) for p in pd.read_csv(A / "evals.csv").query("set == 'val_full'").passno))
