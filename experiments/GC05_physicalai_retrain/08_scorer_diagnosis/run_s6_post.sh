#!/bin/bash
# Detached post-processing (no model run): part a/c tables on SEEN, then part b (scorer outputs on both sets + probes trained on SEEN).
# Start with: setsid nohup ./run_s6_post.sh > /dev/null 2>&1 &
cd "$(dirname "$0")"; ./load_gate.sh
./run_s6_py.sh s6_analyze.py --set=seen > analyze_seen.log 2>&1; echo "exit $?" >> analyze_seen.log
./run_s6_py.sh s6_partb.py > partb.log 2>&1; echo "exit $?" >> partb.log
