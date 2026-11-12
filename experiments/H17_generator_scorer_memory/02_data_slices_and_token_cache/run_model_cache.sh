#!/bin/bash
# run_model.sh without the 2-core pin: the full-cache job's 8 spawned loader workers may use all cores (Sid's one-off exception for
# this job only); nice 19 is kept. Same interpreter, overlay and PYTHONPATH as run_model.sh.
E={runs}/alpasim_bulk/conda-envs
G="{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim"
ENC="{working_notes}/LORA Encoder/Encoder-LoRA-KIT"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$E/dj-torch-cu128-overlay:$HERE:$G/s1_build/src:$G/s2_train/src:$G/s8_018b_build/src:$ENC/s1_build/src:$ENC/s2_train/src"
export PYTHONDONTWRITEBYTECODE=1 DRIVE_JEPA_MSDA=pytorch OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$G/s1_build" && exec nice -n 19 "$E/drive-jepa/bin/python" "$@"
