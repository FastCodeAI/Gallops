#!/bin/bash
# S10 task 7 dump entry: environment of run_s10_dump_b1.sh (conda drive-jepa python + torch 2.8 overlay, CPU ONLY, nice 10); first argument = the script. I/O class: best-effort, lowest priority (7), NOT idle as in run_s10_dump_b1.sh: on 2026-10-09 17:23-17:43 an idle-class start stayed blocked on reads of the HDD for over 20 minutes (17 s of CPU).
# Start gate as run_s10_dump_b1.sh (5-min load < 32, available RAM > 40 GB) unless NOGATE=1 (short checks only). The root-disk condition of run_s10_dump_b1.sh (> 15 GB) is
# NOT a start condition here: on 2026-10-09 17:53 the root disk had 14.8 GiB free because of other users' files, and these dumps write nothing to it (output, logs on the NAS;
# no bytecode). The free space is printed at start instead.
read l1 l5 l15 _ < /proc/loadavg; ma=$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo); dk=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$NOGATE" = 1 ] || awk -v a=$l5 -v s=$ma 'BEGIN{exit !(a<32 && s>40)}' || { echo "start gate closed: 5-min load $l5, available RAM $ma GB, root disk free $dk GB"; exit 3; }
echo "start $(date -Is): 5-min load $l5, available RAM $ma GB, root disk free $dk GB (rounded up)"
OV={runs}/alpasim_bulk/conda-envs/dj-torch-cu128-overlay; T=4; S=$1; shift
for x in "$@"; do case $prev in --threads) T=$x;; esac; prev=$x; done
cd "$(dirname "$0")" && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$OV OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T ionice -c2 -n7 nice -n 10 {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python $S "$@"
