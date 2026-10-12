#!/bin/bash
# S9 entry (no model forward pass): conda drive-jepa python, CPU only, nice 10, cwd = vendored navsim_v2 for its imports.
# Start gate (brief + the RAM rule kept from S6): 5-min load < 32 AND free swap > 15 GB AND available RAM > 40 GB. Waits, logs, starts nothing else.
# Usage: run_s9_py.sh <threads> <script.py> [args]
A=$HOME/renesas-sid/Paddock/paddock-sid/Alpasim; H=$(cd "$(dirname "$0")" && pwd); T=$1; S=$2; shift 2
state() { read l1 l5 l15 _ < /proc/loadavg; echo "$(date -Is),$l1,$l5,$l15,$(awk '/SwapFree/{printf "%.1f",$2/1048576}' /proc/meminfo),$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo),$(df -BG --output=avail / | tail -1 | tr -dc 0-9),$(df -BG --output=avail {nas} | tail -1 | tr -dc 0-9 | awk '{printf "%.2f",$1/1024}'),$1" >> $H/state_log.csv; }
while true; do
  read _ l5 _ < /proc/loadavg; sf=$(awk '/SwapFree/{print int($2/1048576)}' /proc/meminfo); ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if awk -v a=$l5 -v s=$sf -v m=$ma 'BEGIN{exit !(a<32 && s>15 && m>40)}'; then state "gate open: start $S $*"; break; fi
  state "gate closed: waiting for $S"; sleep 120
done
cd $A/adapters/drivejepa/src/navsim_v2 && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T S9_THREADS=$T PYTHONPATH=.:$H:$H/../09_three_hypotheses:$H/../04_training_set MPLBACKEND=Agg ionice -c3 nice -n 10 /usr/bin/time -f "cpu_seconds %U+%S wall %e maxrss_kb %M" {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python $H/$S "$@"
rc=$?; state "end $S exit $rc"; exit $rc
