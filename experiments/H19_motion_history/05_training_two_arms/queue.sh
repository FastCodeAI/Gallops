#!/bin/bash
# Approach 4 training queue (TRAIN brief): arm A, then arm B, in one detached tmux session (a4train).
# Each arm: wait for the memory gate (5-min load < 32 and available RAM > 40 GB), run; on a crash, wait for the gate again and resume
# once from the newest save (resume/last.pt or ckpt/p*.pt, whichever is newer); a second crash stops the queue and writes the error to
# the report. A STOP file in the run folder stops the queue after that arm. After each arm: report_arm.py appends its section.
# Every python step runs pinned (run.sh: taskset -c 0-1 nice -n 19).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MH="$(dirname "$HERE")"
RUNS={approach4_store}/runs
REPORT="$MH/reports/TRAIN_ARMS.md"
QLOG="$RUNS/queue.log"
say() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }

gate() {
  while :; do
    l=$(cut -d' ' -f2 /proc/loadavg); r=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
    if awk -v l="$l" -v r="$r" 'BEGIN{exit !(l < 32 && r > 40)}'; then say "memory gate ok: load5 $l, RAM $r GB"; return; fi
    say "memory gate wait: load5 $l, RAM $r GB"; sleep 300
  done
}

newest_save() { ls -t "$RUNS/arm_$1/resume/last.pt" "$RUNS/arm_$1"/ckpt/p*.pt 2>/dev/null | head -1; }

crash_report() {  # arm, log
  { echo; echo "## Arm $1: STOPPED after a second crash ($(date '+%F %T'))"; echo; echo '```'; grep -v -i warn "$2" | tail -40; echo '```'; } >> "$REPORT"
}

say "queue start (tmux session a4train, pid $$)"
for arm in A B; do
  log="$RUNS/arm_${arm}_stdout.log"
  gate
  say "arm $arm: start"
  "$HERE/run.sh" "$HERE/train_a4.py" --arm "$arm" --go-from-sid >> "$log" 2>&1; rc=$?
  if [ $rc -ne 0 ]; then
    ck=$(newest_save "$arm")
    say "arm $arm: CRASH rc $rc; resuming once from ${ck:-nothing (no save yet: fresh start)}"
    gate
    if [ -n "$ck" ]; then
      "$HERE/run.sh" "$HERE/train_a4.py" --arm "$arm" --go-from-sid --resume "$ck" >> "$log" 2>&1; rc=$?
    else
      mv "$RUNS/arm_$arm" "$RUNS/arm_${arm}_crash1_$(date +%s)" 2>/dev/null
      "$HERE/run.sh" "$HERE/train_a4.py" --arm "$arm" --go-from-sid >> "$log" 2>&1; rc=$?
    fi
    if [ $rc -ne 0 ]; then
      say "arm $arm: SECOND CRASH rc $rc; queue stopped"; crash_report "$arm" "$log"
      echo "$(date '+%F %T') arm $arm STOPPED after a second crash (see TRAIN_ARMS.md)" > "$RUNS/STATUS.txt"
      exit 1
    fi
  fi
  say "arm $arm: finished"
  "$HERE/run.sh" "$HERE/report_arm.py" "$arm" >> "$QLOG" 2>&1 || say "arm $arm: report_arm.py failed"
  if [ -f "$RUNS/arm_$arm/STOP" ]; then say "STOP file in arm_$arm: queue stopped"; exit 0; fi
done
say "queue done"
echo "$(date '+%F %T') queue done: arms A and B finished (see TRAIN_ARMS.md)" > "$RUNS/STATUS.txt"
