#!/bin/bash
# Closing checks: same interpreter as s1/s2; s1 and s2 src imported unchanged.
# Usage: ./run.sh src/some_script.py [args]
E={envs}
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$E/dj-torch-cu128-overlay:$HERE/src:$HERE/../03_encoder_training/src:$HERE/../02_encoder_build/src"
export PYTHONDONTWRITEBYTECODE=1
export DRIVE_JEPA_MSDA=pytorch
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$HERE" && exec nice -n 5 "$E/drive-jepa/bin/python" "$@"
