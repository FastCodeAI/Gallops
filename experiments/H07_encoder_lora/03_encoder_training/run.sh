#!/bin/bash
# Stage 2: same interpreter and add-on as 02_encoder_build/run.sh; 02_encoder_build/src is
# imported as is (on PYTHONPATH after our own src).
# Usage: ./run.sh src/some_script.py [args]
E={envs}
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$E/dj-torch-cu128-overlay:$HERE/src:$HERE/../02_encoder_build/src"
export PYTHONDONTWRITEBYTECODE=1
export DRIVE_JEPA_MSDA=pytorch
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$HERE" && exec nice -n 5 "$E/drive-jepa/bin/python" "$@"
