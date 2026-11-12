#!/bin/bash
# Same interpreter as 08_generator_and_encoder_together/run.sh; earlier stages and experiments/drivejepa_kitscenes src imported unchanged.
E={envs}
HERE="$(cd "$(dirname "$0")" && pwd)"
ENC="$HERE/.."
export PYTHONPATH="$E/dj-torch-cu128-overlay:$ENC/02_encoder_build/src:$ENC/03_encoder_training/src:$HERE/../05_generator_survey/src:$HERE/../06_generator_only_fast_rate/src:$HERE/../07_generator_only_gentle_rate/src:$HERE/../08_generator_and_encoder_together/src:$HERE/src"
export PYTHONDONTWRITEBYTECODE=1
export DRIVE_JEPA_MSDA=pytorch
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$HERE" && exec nice -n 5 "$E/drive-jepa/bin/python" "$@"
