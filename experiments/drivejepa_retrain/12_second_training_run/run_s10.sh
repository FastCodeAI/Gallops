#!/bin/bash
# S10 entry: environment exactly as 06_first_training_run/run_s4.sh (conda drive-jepa python + torch 2.8.0+cu128 overlay, GPU 0, niced).
OV={runs}/alpasim_bulk/conda-envs/dj-torch-cu128-overlay
cd "$(dirname "$0")" && PYTHONPATH=$OV OMP_NUM_THREADS=2 nice -n 5 {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python s10_gen_train.py "$@"
