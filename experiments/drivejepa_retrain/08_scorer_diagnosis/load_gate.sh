#!/bin/bash
# Logs load, swap and available RAM every 2 min; exits 0 the first time the start gate holds (5-min load < 32 AND available RAM > 40 GB).
# Starts nothing. Gate agreed with Sid 2026-10-07 (replaces the free-swap gate).
L=$(dirname "$0")/load_log.csv; [ -f "$L" ] || echo "time,load1,load5,load15,swap_free_gb,mem_avail_gb,gate" > "$L"
while true; do
  read l1 l5 l15 _ < /proc/loadavg; sf=$(awk '/SwapFree/{printf "%.1f",$2/1048576}' /proc/meminfo); ma=$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo)
  ok=$(awk -v a=$l5 -v s=$ma 'BEGIN{print (a<32 && s>40)?"open":"closed"}'); echo "$(date -Is),$l1,$l5,$l15,$sf,$ma,$ok" >> "$L"
  [ "$ok" = open ] && exit 0; sleep 120
done
