#!/bin/bash
# drive-jepa python with the torch 2.8 overlay (as run_s9_dump.sh), CPU only, nice 10.
OV={runs}/alpasim_bulk/conda-envs/dj-torch-cu128-overlay
cd "$(dirname "$0")" && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$OV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ionice -c3 nice -n 10 /usr/bin/time -f "cpu_seconds %U+%S wall %e" {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python s9_trial_features.py
