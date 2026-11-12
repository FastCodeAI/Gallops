#!/bin/bash
# Same interpreter and overlay as the lane's s1_build/run.sh (conda drive-jepa python + torch 2.8.0+cu128 overlay), with the lane's
# s1_build / s2_train / s8_018b_build sources on PYTHONPATH (pai_common, pai_model, pai_data, pai_slim, pai_model_b: imported, never edited).
E={alpasim_bulk}/conda-envs
G="{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim"
ENC="{working_notes}/LORA Encoder/Encoder-LoRA-KIT"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$E/dj-torch-cu128-overlay:$HERE:$G/s1_build/src:$G/s2_train/src:$G/s8_018b_build/src:$ENC/s1_build/src:$ENC/s2_train/src"
export PYTHONDONTWRITEBYTECODE=1 DRIVE_JEPA_MSDA=pytorch OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$G/s1_build" && exec taskset -c 0-1 nice -n 19 "$E/drive-jepa/bin/python" "$@"
