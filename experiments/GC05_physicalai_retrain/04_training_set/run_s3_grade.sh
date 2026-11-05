#!/bin/bash
# S3 grading: conda drive-jepa python (nuplan + vendored navsim_v2, read-only), CPU only, niced (shared machine).
D={runs}/paddock_sid/Alpasim; S=$D/analysis_3.0/Retrain-data
cd $D/adapters/drivejepa/src/navsim_v2 && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:$S/04_training_set:$S/03_grade_candidates:$S/01_find_clips nice -n 10 {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python $S/04_training_set/${SCRIPT:-s3_grade.py} "$@"
