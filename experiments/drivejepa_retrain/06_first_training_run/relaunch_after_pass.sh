#!/bin/bash
# Operational relaunch at a pass boundary (resume from run/state.json): waits for "PASS $1 done", stops the trainer, restarts it with new
# process-count arguments. Weights, optimiser, scaler and the per-pass sampler seed are restored by s4_train.py; nothing about the recipe changes.
cd "$(dirname "$0")"; P=$1; shift
until grep -q "PASS $P done" run_stdout.log; do sleep 2; done; sleep 6   # let the tripwire check after the pass write its line
grep -q "TRIPWIRE" run_stdout.log && { echo "tripwire present: not relaunching" >> run_stdout.log; exit 0; }
tmux kill-session -t 06_first_training_run; sleep 8; pkill -u "$USER" -f "conda-envs/drive-jepa/bin/python 06_first_training_run[.]py"; sleep 5
echo "[relaunch after pass $P with: $*] $(date +%H:%M:%S)" >> run_stdout.log
tmux new-session -d -s 06_first_training_run "cd $PWD && ./run_s4.sh $* 2>&1 | grep -v --line-buffered -i 'warn' >> run_stdout.log; echo EXIT \$? >> run_stdout.log"
