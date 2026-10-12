#!/bin/bash
# S6 analysis entry (no model run): conda drive-jepa python, CPU only, 2 threads, nice 10, cwd = vendored navsim_v2 for its imports.
A=$HOME/renesas-sid/Paddock/paddock-sid/Alpasim; H=$(cd "$(dirname "$0")" && pwd); S=$1; shift
cd $A/adapters/drivejepa/src/navsim_v2 && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:$H/../04_training_set ionice -c3 nice -n 10 {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python $H/$S "$@"
