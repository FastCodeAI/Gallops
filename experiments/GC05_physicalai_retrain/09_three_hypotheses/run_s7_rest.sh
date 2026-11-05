#!/bin/bash
# Detached chain after H1: H2 (exam inspection, read-only), then H3. Start: setsid nohup ./run_s7_rest.sh > /dev/null 2>&1 &
cd "$(dirname "$0")"; while pgrep -f s7_h1.py > /dev/null; do sleep 5; done
./run_s7_py.sh 2 s7_h2.py > h2.log 2>&1; ./run_s7_py.sh 4 s7_h3.py > h3.log 2>&1
