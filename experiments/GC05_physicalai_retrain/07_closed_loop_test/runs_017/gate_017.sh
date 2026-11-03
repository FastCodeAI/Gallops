#!/bin/bash
# 0.1.7 build gate: training side (conda drive-jepa python + torch 2.8 overlay, as 12_second_training_run/run_s10_dump_gen.sh) then adapter side (alpasim venv -> the adapter's own worker). CPU only; temp and caches on the NAS; no bytecode.
set -uo pipefail
R=$(cd "$(dirname "$0")" && pwd); OUT={runs}/drivejepa_retrain/07_closed_loop_test_017/gate; TMP={runs}/drivejepa_retrain/07_closed_loop_test_017/tmp; mkdir -p "$OUT" "$TMP/mpl" "$TMP/cache"
export TMPDIR=$TMP MPLCONFIGDIR=$TMP/mpl XDG_CACHE_HOME=$TMP/cache PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
OV={runs}/alpasim_bulk/conda-envs/dj-torch-cu128-overlay
echo "[$(date -Is)] training side"; [ -f "$OUT/train_side.npz" ] || ( cd "$R" && PYTHONPATH=$OV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python "$R/gate_train_side.py" "$OUT" ) || { echo "TRAIN SIDE FAILED"; exit 1; }
echo "[$(date -Is)] adapter side"; nice -n 10 {runs}/paddock_sid/Alpasim/alpasim/.venv/bin/python -B "$R/gate_adapter_side.py" "$OUT" || { echo "ADAPTER SIDE FAILED"; exit 1; }
cp "$OUT/gate_017.json" "$R/tables/gate_017.json"; echo "[$(date -Is)] GATE SCRIPT DONE"
