#!/bin/bash
# S6 dump entry: conda drive-jepa python with the torch 2.8 overlay (as S4), CPU ONLY, nice 10, idle I/O class.
# Refuses to start unless the start gate holds: 5-minute load < 32 and available RAM > 40 GB, and the root disk has > 5 GB free.
read l1 l5 l15 _ < /proc/loadavg; ma=$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo); dk=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
awk -v a=$l5 -v s=$ma -v d=$dk 'BEGIN{exit !(a<32 && s>40 && d>5)}' || { echo "start gate closed: 5-min load $l5, available RAM $ma GB, root disk free $dk GB"; exit 3; }
OV={runs}/alpasim_bulk/conda-envs/dj-torch-cu128-overlay; T=4
for x in "$@"; do case $prev in --threads) T=$x;; esac; prev=$x; done
cd "$(dirname "$0")" && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$OV OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T ionice -c3 nice -n 10 {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python s6_dump.py "$@"
