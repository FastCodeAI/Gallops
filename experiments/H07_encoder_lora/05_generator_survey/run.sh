#!/bin/bash
# Same interpreter and add-on as experiments/drivejepa_kitscenes/02_encoder_build/run.sh (itself Gallops
# 06_first_training_run/run_s4.sh:2-4). s1 and s2 sources are imported unchanged.
E={envs}
HERE="$(cd "$(dirname "$0")" && pwd)"
ENC="$HERE/.."
export PYTHONPATH="$E/dj-torch-cu128-overlay:$ENC/02_encoder_build/src:$ENC/03_encoder_training/src:$HERE/src"
export PYTHONDONTWRITEBYTECODE=1
export DRIVE_JEPA_MSDA=pytorch
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$HERE" && exec nice -n 5 "$E/drive-jepa/bin/python" "$@"
