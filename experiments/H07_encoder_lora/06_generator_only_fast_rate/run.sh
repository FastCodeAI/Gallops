#!/bin/bash
# Same interpreter and add-on as 05_generator_survey/run.sh (experiments/drivejepa_kitscenes 02_encoder_build/run.sh).
# Imports, unchanged: experiments/drivejepa_kitscenes 02_encoder_build/src, 03_encoder_training/src, 05_generator_survey/src.
E={envs}
HERE="$(cd "$(dirname "$0")" && pwd)"
ENC="$HERE/.."
export PYTHONPATH="$E/dj-torch-cu128-overlay:$ENC/02_encoder_build/src:$ENC/03_encoder_training/src:$HERE/../05_generator_survey/src:$HERE/src"
export PYTHONDONTWRITEBYTECODE=1
export DRIVE_JEPA_MSDA=pytorch
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$HERE" && exec nice -n 5 "$E/drive-jepa/bin/python" "$@"
