#!/usr/bin/env python
"""Mean scene score of 0.1.4b / 0.1.6 / 0.1.7 over the first N scenes of tables/scenes_to_run.txt that have a finished 0.1.7 run (same readout code on all three). Prints only."""
import sys, pathlib, contextlib, io, warnings
warnings.filterwarnings("ignore"); HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import s017_readout as RD
    S = [l.split()[0] for l in open(HERE / "tables" / "scenes_to_run.txt")]; S = [s for s in S if (RD.run_dir("017", s) / "aggregate" / "metrics_results.parquet").is_file()]
    if len(sys.argv) > 1: S = S[: int(sys.argv[1])]
    sc = {a: [float(RD.BR.arm_block(RD.run_dir(a, s), s)[0]["scene_score"]) for s in S] for a in RD.ARMS}
print(f"{len(S)} scenes | mean score 0.1.4b {sum(sc['014b']) / len(S):.3f} | 0.1.6 {sum(sc['016']) / len(S):.3f} | 0.1.7 {sum(sc['017']) / len(S):.3f}")
