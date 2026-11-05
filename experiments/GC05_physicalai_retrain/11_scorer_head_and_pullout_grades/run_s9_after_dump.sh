#!/bin/bash
# Detached chain after the pull-out dump: C.3 merge + tables, C.4 retrain + re-evaluation, trial-scene inspection (read-only).
cd "$(dirname "$0")"; until grep -q "^DONE" c_dump.log || ! pgrep -f "[s]9_c_dump.py" > /dev/null; do sleep 30; done
grep -q "^DONE" c_dump.log || { echo "dump did not finish" > after_dump.status; exit 1; }
./run_s9_py.sh 2 s9_c_analyze.py > c_analyze.log 2>&1 && ./run_s9_py.sh 4 s9_b.py --c4 > c4.log 2>&1 && ./run_s9_py.sh 2 s9_trial_gate.py > c4_trial_gate.log 2>&1; echo "chain exit $?" > after_dump.status
