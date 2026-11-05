#!/bin/bash
# Block until the sweep has logged more than $1 finished runs (or stopped), then print one comparison line per newly finished scene (0.1.4b / 0.1.6 / 0.1.7). Pattern of 07_closed_loop_test/wait_next_scene.sh.
cd "$(dirname "$0")"; N=${1:?count seen}; n=0
until [ "$(grep -c '=== RUN [0-9]* rc=' logs/sweep.log)" -gt "$N" ] || grep -q "STOP:\|SWEEP DONE\|STOP file" logs/sweep.log || [ $n -ge 114 ]; do sleep 5; n=$((n+1)); done
done_now=$(grep -c '=== RUN [0-9]* rc=' logs/sweep.log); echo "$(date +%H:%M) finished so far: $done_now of 30"
grep -E "=== RUN [0-9]+ rc=|STOP|SWEEP DONE|FATAL|BLOCKED|RED GATES|HARD|WAIT" logs/sweep.log | tail -3 | cut -c1-200
scenes=$(grep -oE '=== RUN [0-9]+: [0-9a-f]{8}' logs/sweep.log | awk '{print $4}' | head -n "$done_now" | tail -n +$((N+1)) | paste -sd, -)
[ -n "$scenes" ] && TMPDIR={runs}/drivejepa_retrain/07_closed_loop_test_017/tmp {runs}/paddock_sid/Alpasim/alpasim/.venv/bin/python -B s017_readout.py --scenes "$scenes" --print-only 2>&1 | grep -E "^[0-9a-f]{8}: |Error|error" | cut -c1-520
df -h / | tail -1 | awk '{print "root free", $4}'
