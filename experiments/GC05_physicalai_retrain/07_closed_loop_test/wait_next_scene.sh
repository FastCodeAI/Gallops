#!/bin/bash
# Block until the sweep has logged more than $1 finished runs (or stopped), then print the newest scene's comparison line(s).
cd "$(dirname "$0")"; N=${1:?count seen}; n=0
until [ "$(grep -c '=== RUN [0-9]* rc=' logs/sweep.log)" -gt "$N" ] || grep -q "STOP:\|SWEEP DONE" logs/sweep.log || [ $n -ge 1380 ]; do sleep 5; n=$((n+1)); done
date +%H:%M; grep -E "=== RUN [0-9]+ rc=|STOP:|SWEEP DONE|FATAL|BLOCKED|RED GATES|HARD" logs/sweep.log | tail -4 | cut -c1-200
new=$(grep -B200 -E '=== RUN [0-9]+ rc=' logs/sweep.log | grep -oE '=== RUN [0-9]+: [0-9a-f]{8}' | awk '{print $4}' | tr -d ':' | tail -n +$((N+1)) | paste -sd, -)
done_now=$(grep -c '=== RUN [0-9]* rc=' logs/sweep.log); echo "finished so far: $done_now"
scenes=$(grep -oE '=== RUN [0-9]+: [0-9a-f]{8}' logs/sweep.log | awk '{print $4}' | head -n "$done_now" | tail -n +$((N+1)) | paste -sd, -)
[ -n "$scenes" ] && cd ../../.. && alpasim/.venv/bin/python analysis_3.0/Retrain-data/07_closed_loop_test/s5_readout.py --scenes "$scenes" --print-only 2>&1 | grep -E "^[0-9a-f]{8}: " | cut -c1-330
grep -E "WAIT|BIND_TIME" logs/sweep.log 2>/dev/null | tail -2 | cut -c1-140; df -h / | tail -1 | awk '{print "root free", $4}'
