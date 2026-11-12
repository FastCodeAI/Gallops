#!/bin/bash
# Reboot watcher, Sid's s9 watcher.sh with our paths (cron: @reboot and every 5 min). Restarts the run with --resume <latest save> when it is marked ACTIVE, has no DONE, has no live process and
# has a save; at most 5 restarts. Sid's gates were 40 GB free and load 32; this shared box sits at 15-25 GB and load 30-50 with run 3A-M on it, so here 12 GB and load 96. Logs instead of Sid's ping.
N=${1:-p1_05nov}; R={runs}/b2d_drivejepa/phase1/runs/$N; H={ops}/b2d_drivejepa/phase1; LOG=$H/logs/watcher.log
PAT="train_b2d.py --name $N"
[ -e $R/ACTIVE ] || exit 0
[ -e $R/DONE ] && exit 0
pgrep -f "$PAT" >/dev/null && exit 0
sleep 20; pgrep -f "$PAT" >/dev/null && exit 0
n=$(grep -c "RESTART" $LOG 2>/dev/null); n=${n:-0}
if [ "$n" -ge 5 ]; then [ -e $R/WATCHER_GAVE_UP ] || { touch $R/WATCHER_GAVE_UP; echo "$(date -Is) gave up after 5 restarts" >> $LOG; }; exit 0; fi
last=$(ls $R/LATEST.* 2>/dev/null | sort | tail -1)
if [ -z "$last" ]; then echo "$(date -Is) run died before its first save" >> $LOG; exit 0; fi
ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); l=$(cut -d' ' -f2 /proc/loadavg)
if [ "$ma" -le 12 ] || [ "$(echo "$l >= 96" | bc)" = 1 ] || ! nvidia-smi >/dev/null 2>&1; then echo "$(date -Is) waiting: avail $ma GB load5 $l" >> $LOG; exit 0; fi
s=$(cat $last)
echo "$(date -Is) RESTART --resume $s (uptime: $(uptime -p))" >> $LOG
setsid nohup $H/run_train.sh $N --resume "$s" > /dev/null 2>&1 < /dev/null &
