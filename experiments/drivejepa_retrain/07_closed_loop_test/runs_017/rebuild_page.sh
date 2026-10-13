#!/bin/bash
# Generator-only trial: readout of every finished 0.1.7 run -> groups -> page on the NAS (served on 8778). Safe to run while the sweep is going.
cd {runs}/paddock_sid/Alpasim; R=analysis_3.0/Retrain-data/07_closed_loop_test/runs_017; export TMPDIR={runs}/drivejepa_retrain/07_closed_loop_test_017/tmp; PY=alpasim/.venv/bin/python
$PY -B $R/s017_readout.py 2>&1 | grep -E "^skipped" | cut -c1-200
$PY -B $R/s017_models_table.py >/dev/null 2>&1
$PY -B $R/s017_groups.py 2>&1 | grep -v -i warn && $PY -B $R/s017_build_review.py 2>&1 | grep -v -i warn | tail -7 | cut -c1-300
