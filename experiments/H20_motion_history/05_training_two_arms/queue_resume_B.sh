#!/bin/bash
# Arm B after the 2026-11-18 11:10 machine reboot: the queue's one crash-resume, from the newest save (ckpt/p00.6.pt).
# A further crash stops here and writes the error to the report (the queue's second-crash rule). Then report_arm.py B.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; MH="$(dirname "$HERE")"
RUNS={approach4_store}/runs; QLOG="$RUNS/queue.log"; REPORT="$MH/reports/TRAIN_ARMS.md"; log="$RUNS/arm_B_stdout.log"
say() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
say "arm B: killed by the machine reboot at 2026-11-18 11:10:53 (last step 2850); resuming once from $RUNS/arm_B/ckpt/p00.6.pt (tmux a4train)"
"$HERE/run.sh" "$HERE/train_a4.py" --arm B --go-from-sid --resume "$RUNS/arm_B/ckpt/p00.6.pt" >> "$log" 2>&1; rc=$?
if [ $rc -ne 0 ]; then
  say "arm B: SECOND CRASH rc $rc; queue stopped"
  { echo; echo "## Arm B: STOPPED after a second crash ($(date '+%F %T'))"; echo; echo '```'; grep -v -i warn "$log" | tail -40; echo '```'; } >> "$REPORT"
  echo "$(date '+%F %T') arm B STOPPED after a second crash (see TRAIN_ARMS.md)" > "$RUNS/STATUS.txt"; exit 1
fi
say "arm B: finished"
"$HERE/run.sh" "$HERE/report_arm.py" B >> "$QLOG" 2>&1 || say "arm B: report_arm.py failed"
say "queue done"
echo "$(date '+%F %T') queue done: arms A and B finished (see TRAIN_ARMS.md)" > "$RUNS/STATUS.txt"
