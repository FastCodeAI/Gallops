#!/bin/bash
# S10 task 7: launches the three batch-1 dumps of the new generator checkpoint, detached (setsid nohup), CPU, float32, batch fix installed, each resumable by shard
# with its own in-run load / memory gate (s10_dump_b1.gate: pauses at 5-min load >= 32 or available RAM < 20 GB) and a DONE.json marker.
# Each launcher retries every 2 minutes while the start gate of run_s10_dump_gen.sh is closed (exit 3). Usage: launch_t7_dumps.sh CKPT [THREADS POOL LOADERS]
CK=$1; T=${2:-4}; P=${3:-2}; LD=${4:-1}; D={runs}/drivejepa_retrain/12_second_training_run/dump_gen; cd "$(dirname "$0")"; [ -f "$CK" ] || { echo "no checkpoint $CK"; exit 2; }
go() { name=$1; shift; mkdir -p $D/$name
  setsid nohup bash -c "while :; do ./run_s10_dump_gen.sh $* --ckpt $CK --out dump_gen/$name --bs 1 --fix --threads $T --pool $P --loaders $LD >> $D/$name/dump.log 2>&1 < /dev/null; e=\$?; echo \"\$(date -Is) exit \$e\" >> $D/$name/launcher.log; [ \$e = 3 ] || break; sleep 120; done" > /dev/null 2>&1 < /dev/null &
  echo "$name launcher pid $!"; }
go unseen s10_dump_gen.py --set unseen
go seen s10_dump_gen.py --set seen
go pullout s10_dump_pullout.py
