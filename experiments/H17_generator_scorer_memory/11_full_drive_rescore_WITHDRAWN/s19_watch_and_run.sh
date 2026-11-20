#!/bin/bash
# Waits until the carry / 0.1.8b run folders are readable on the NAS, then runs the full-drive re-score once (read-only on the NAS).
cd "$(dirname "$0")"
until timeout 30 ls {experiment_store}/runs/cl18/runs >/dev/null 2>&1 && timeout 30 ls {experiment_store}/runs/cl12/runs >/dev/null 2>&1 \
      && timeout 30 ls {retrain_store}/018b/closed_loop/runs >/dev/null 2>&1; do sleep 300; done
echo "NAS readable $(date)" > ../logs/fulldrive_all.log
MPLCONFIGDIR=/tmp/s19_mpl nice -n 19 taskset -c 0,1 {paddock_sid}/Alpasim/alpasim/.venv/bin/python -B s19_fulldrive_rescore.py >> ../logs/fulldrive_all.log 2>&1
echo "rc=$? $(date)" >> ../logs/fulldrive_all.log
