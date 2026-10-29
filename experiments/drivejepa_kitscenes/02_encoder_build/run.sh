#!/bin/bash
# Same interpreter and add-on as Gallops 06_first_training_run/run_s4.sh:2-4.
# Usage: ./run.sh src/some_script.py [args]   or   ./run.sh -m pytest tests -q
E={envs}
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$E/dj-torch-cu128-overlay:$HERE/src"
export PYTHONDONTWRITEBYTECODE=1
export DRIVE_JEPA_MSDA=pytorch
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$HERE" && exec nice -n 5 "$E/drive-jepa/bin/python" "$@"
