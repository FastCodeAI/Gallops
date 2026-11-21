#!/bin/bash
# Approach-4 runner: approach 3's run_model.sh (conda drive-jepa python + torch 2.8.0+cu128 overlay, the lane's sources on PYTHONPATH,
# imported never edited) with this folder and approach 3 src first, then the lane in s9_018b_train/run.sh order (s2 pai_train before s1). CPU pinned.
E={alpasim_bulk}/conda-envs
G="{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim"
ENC="{working_notes}/LORA Encoder/Encoder-LoRA-KIT"
A3="{working_notes}/memory/3_generator_scorer_memory_PAI/src"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$E/dj-torch-cu128-overlay:$HERE:$A3:$G/s9_018b_train/src:$G/s8_018b_build/src:$G/s3_offline/src:$G/s2_train/src:$G/s1_build/src:$ENC/s1_build/src:$ENC/s2_train/src"
export PYTHONDONTWRITEBYTECODE=1 DRIVE_JEPA_MSDA=pytorch OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$G/s1_build" && exec taskset -c 0-1 nice -n 19 "$E/drive-jepa/bin/python" "$@"
