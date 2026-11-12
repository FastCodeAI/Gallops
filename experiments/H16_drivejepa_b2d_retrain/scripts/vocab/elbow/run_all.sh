#!/bin/bash
source {drivejepa_env}/env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd {ops}/b2d_drivejepa/vocab/elbow
for k in 512 1024 2048 4096 8192 768 1536 3072 6144 256 384; do for s in 20261105 20261106 20261107; do nice -n 15 ionice -c3 $PY elbow.py $k $s 2>&1 | grep -E "DONE|Error|Traceback"; done; done
echo ELBOW_ALL_DONE
