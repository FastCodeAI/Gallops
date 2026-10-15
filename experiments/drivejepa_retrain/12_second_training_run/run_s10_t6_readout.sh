#!/bin/bash
# S10 task 6: environment exactly as run_s10.sh, entry s10_pilot_readout.py (float32 + 16-bit SEEN held-out readout of one checkpoint).
OV={runs}/alpasim_bulk/conda-envs/dj-torch-cu128-overlay
cd "$(dirname "$0")" && PYTHONPATH=$OV OMP_NUM_THREADS=2 nice -n 5 {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python s10_pilot_readout.py "$@"
