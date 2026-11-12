#!/bin/bash
# S3.5 training entry: conda drive-jepa python with torch 2.8.0+cu128 layered on top (overlay on the HDD), GPU 0, niced.
OV={runs}/alpasim_bulk/conda-envs/dj-torch-cu128-overlay
cd "$(dirname "$0")" && PYTHONPATH=$OV OMP_NUM_THREADS=2 nice -n 5 {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python s35_train.py "$@"
