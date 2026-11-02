#!/bin/bash
# S10 task 6: final run = 2 more passes from the chosen pilot (P0, lambda 0, standard range, batch fix on) at lr 2.5e-5.
# The task's limit is a 5-min load under 30; at 16:15 it was 33 (other users), so this waits for it to fall below 30 and then starts the run.
G={runs}/drivejepa_retrain/12_second_training_run/gen; cd "$(dirname "$0")"; mkdir -p $G/final_P0
until awk '{exit !($2 < 30)}' /proc/loadavg; do echo "$(date -Is) waiting, loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $G/final_P0/launch_gate.log; sleep 30; done
echo "$(date -Is) load ok ($(cut -d' ' -f1-3 /proc/loadavg)); starting" >> $G/final_P0/launch_gate.log
exec ./run_s10.sh --mode train --run final_P0 --init $G/pilot_fix_l0/pass_01.pt --lr 2.5e-5 --passes 2 --workers 2 --lam 0 >> $G/final_P0/stdout.log 2>&1 < /dev/null
