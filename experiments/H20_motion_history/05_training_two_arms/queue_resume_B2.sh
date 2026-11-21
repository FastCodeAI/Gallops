#!/bin/bash
# Arm B, third start (Sid's go 2026-11-19 after the second crash, a GPU out-of-memory at check 0.8): resume once from
# resume/last.pt (step 3500). Waits for the memory gate AND >= 12 GB free on the GPU. Any further crash stops here and is reported.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; MH="$(dirname "$HERE")"
RUNS={approach4_store}/runs; QLOG="$RUNS/queue.log"; REPORT="$MH/reports/TRAIN_ARMS.md"; log="$RUNS/arm_B_stdout.log"
say() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
while :; do
  l=$(cut -d' ' -f2 /proc/loadavg); r=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  g=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{printf "%.1f", ($1-$2)/1024}')
  if awk -v l="$l" -v r="$r" -v g="$g" 'BEGIN{exit !(l < 32 && r > 40 && g >= 12)}'; then say "gate ok: load5 $l, RAM $r GB, GPU free $g GB"; break; fi
  say "gate wait: load5 $l, RAM $r GB, GPU free $g GB"
  echo "$(date '+%F %T') arm B waiting to resume (GPU free $g GB < 12, or memory gate)" > "$RUNS/STATUS.txt"; sleep 300
done
say "arm B: third start (Sid's go), resuming from $RUNS/arm_B/resume/last.pt (step 3500)"
"$HERE/run.sh" "$HERE/train_a4.py" --arm B --go-from-sid --resume "$RUNS/arm_B/resume/last.pt" >> "$log" 2>&1; rc=$?
if [ $rc -ne 0 ]; then
  say "arm B: CRASH rc $rc on the third start; stopped"
  { echo; echo "## Arm B: STOPPED, crash on the third start ($(date '+%F %T'))"; echo; echo '```'; grep -v -i warn "$log" | grep -v "custom_fwd\|def backward" | tail -25; echo '```'; } >> "$REPORT"
  echo "$(date '+%F %T') arm B STOPPED: crash on the third start (see TRAIN_ARMS.md)" > "$RUNS/STATUS.txt"; exit 1
fi
say "arm B: finished"
"$HERE/run.sh" "$HERE/report_arm.py" B >> "$QLOG" 2>&1 || say "arm B: report_arm.py failed"
say "queue done"
echo "$(date '+%F %T') queue done: arms A and B finished (see TRAIN_ARMS.md)" > "$RUNS/STATUS.txt"
