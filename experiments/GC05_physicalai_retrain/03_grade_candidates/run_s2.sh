#!/bin/bash
# S2 grading entry point: conda drive-jepa python (nuplan + navsim_v2), CPU only, cwd = vendored navsim_v2 (read-only).
D={runs}/paddock_sid/Alpasim; S=$D/analysis_3.0/Retrain-data
cd $D/adapters/drivejepa/src/navsim_v2 && CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:$S/03_grade_candidates:$S/01_find_clips {runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python $S/03_grade_candidates/s2_run.py "$@"
