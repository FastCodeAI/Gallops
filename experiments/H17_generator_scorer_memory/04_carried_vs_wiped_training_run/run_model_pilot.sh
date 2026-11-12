#!/bin/bash
# run_model.sh with the core pin taken from CORES (default 0-1): the pilot runs two arms in parallel, each on its own two cores
# (carry CORES=0-1, wiped CORES=2-3; Sid's one-off exception for this pilot). nice 19, same interpreter / overlay / PYTHONPATH.
E={runs}/alpasim_bulk/conda-envs
G="{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim"
ENC="{working_notes}/LORA Encoder/Encoder-LoRA-KIT"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$E/dj-torch-cu128-overlay:$HERE:$G/s1_build/src:$G/s2_train/src:$G/s8_018b_build/src:$ENC/s1_build/src:$ENC/s2_train/src"
export PYTHONDONTWRITEBYTECODE=1 DRIVE_JEPA_MSDA=pytorch OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$G/s1_build" && exec taskset -c "${CORES:-0-1}" nice -n 19 "$E/drive-jepa/bin/python" "$@"
