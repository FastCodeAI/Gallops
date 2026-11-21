#!/bin/bash
# Restarts the 18-scene queue when (and only when) its run script gave up at the machine gate: the run script waits at most 3 h and then
# exits 7 ("BLOCKED: preflight not met"), which halts the queue before anything launched. Any other stop, or QUEUE DONE, ends this loop.
# The queue skips finished runs, so a restart never repeats one. Log: logs/cl18_relaunch.log
M={working_folder}; L=$M/logs/cl18_queue.log
exec >> $M/logs/cl18_relaunch.log 2>&1
while :; do
  sleep 120
  tmux has-session -t cl18queue 2>/dev/null && continue
  grep -q "QUEUE DONE" <(tail -3 $L) && { echo "[$(date -Is)] queue done -- exit"; exit 0; }
  last=$(grep -E "STOP: run [0-9]+ .* rc=" $L | tail -1)
  if [[ "$last" == *"rc=7 "* ]] && tail -3 $L | grep -q "queue halted"; then
    echo "[$(date -Is)] queue stopped on the 3 h gate wait ($last) -- relaunching"
    tmux new-session -d -s cl18queue "taskset -c 0-1 nice -n 19 $M/src/s18_queue.sh"
  else
    echo "[$(date -Is)] queue ended otherwise: '$(tail -1 $L | cut -c1-200)' -- not relaunching"; exit 0
  fi
done
