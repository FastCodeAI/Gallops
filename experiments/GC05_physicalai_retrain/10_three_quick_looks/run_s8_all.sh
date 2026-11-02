#!/bin/bash
# Detached chain L1 -> L2 -> L3. Start: setsid nohup ./run_s8_all.sh > /dev/null 2>&1 &
cd "$(dirname "$0")"; ./run_s8_py.sh 2 s8_l1.py > l1.log 2>&1; ./run_s8_py.sh 2 s8_l2.py > l2.log 2>&1; ./run_s8_py.sh 2 s8_l3.py > l3.log 2>&1; touch ALL_DONE
