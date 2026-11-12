#!/bin/bash
# S10 dump entry: environment of 08_scorer_diagnosis/run_s6_dump.sh (conda drive-jepa python + torch 2.8 overlay, CPU ONLY, nice 10, idle I/O class).
# Refuses to start unless 5-minute load < 32, available RAM > 40 GB and the root disk has > 15 GB free.
read l1 l5 l15 _ < /proc/loadavg; ma=$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo); dk=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
awk -v a=$l5 -v s=$ma -v d=$dk 'BEGIN{exit !(a<32 && s>40 && d>15)}' || { echo "start gate closed: 5-min load $l5, available RAM $ma GB, root disk free $dk GB"; exit 3; }
OV={runs}/alpasim_bulk/conda-envs/dj-torch-cu128-overlay; T=6
for x in "$@"; do case $prev in --threads) T=$x;; esac; prev=$x; done
cd "$(dirname "$0")" && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$OV OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T ionice -c3 nice -n 10 {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python s10_dump_b1.py "$@"
