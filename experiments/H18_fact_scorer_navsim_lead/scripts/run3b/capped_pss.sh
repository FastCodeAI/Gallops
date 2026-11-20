#!/bin/bash
# Runs "$@" with the NAVSIM env under the 50 GB memory cap (user, 15 Nov), counted as PSS (shared pages split between the
# processes that map them), so forked loader workers sharing the parent's pages are not counted many times over. A watchdog
# stops the whole process group above CAP_GB. Launch detached: setsid nohup capped_pss.sh <script> args > log 2>&1 &
source {runs}/mmcv-sm120/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PG=$(ps -o pgid= $$ | tr -d ' ')
( while sleep 10; do
    kb=0
    for p in $(ps -o pid= -g $PG); do
      v=$(awk '/^Pss:/ {print $2}' /proc/$p/smaps_rollup 2>/dev/null); kb=$((kb + ${v:-0}))
    done
    if [ "$kb" -gt $((${CAP_GB:-50} * 1024 * 1024)) ]; then echo "CAP: group PSS $((kb / 1048576)) GB > ${CAP_GB:-50} GB at $(date +%T), stopping"; kill -TERM -- -$PG; fi
  done ) &
echo "start $(date +%F' '%T) pgid $PG: $*"
cd ${RUN_CWD:-{runs}/run3b/r3b} && PYTHONPATH=$NAVSIM_DEVKIT_ROOT nice -n 5 $PY "$@"
echo "EXIT $? $(date +%T)"
kill %1 2>/dev/null
