#!/bin/bash
# Part B.3 entry: HF_TOKEN from Alpasim/alpasim_env.sh (as drafts/ext20/run_ext20_scene.sh:100), the wizard's own venv, idle I/O class.
A={runs}/paddock_sid/Alpasim; H=$(cd "$(dirname "$0")" && pwd); source $A/alpasim_env.sh
cd /tmp && ionice -c3 nice -n 10 $A/alpasim/.venv/bin/python -B $H/s10_fresh_fetch.py $H/fresh20.csv 8 $H/fresh_scene_hashes.csv "$@"
