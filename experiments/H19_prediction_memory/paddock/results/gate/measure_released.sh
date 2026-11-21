#!/bin/bash
# Peak memory of one released Drive-JEPA KITScenes run, 20 keyframes (Sid's change 1, 6 Nov).
# Its own session; every 2 s sums RSS and PSS over all processes of that session (the paddock CLI
# and the model node it starts). Guard: load < 32, GPU >= 12 GB, available RAM > 20 GB (Sid's stop
# floor); aborts the run if available RAM falls below 20 GB.
set -u
W={paddock}; D={runs}/smoke
P={paddock_python}
for i in $(seq 720); do
  load=$(awk '{print $2}' /proc/loadavg); avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "$(date '+%F %T %Z') guard load=$load avail=${avail}G gpu=${gpu}MiB"
  awk "BEGIN{exit !($load < 32 && $avail > 20 && $gpu >= 12288)}" && break
  sleep 60
done
cp $W/paddock.toml $D/paddock_released.toml
cd $W
echo "$(date '+%F %T %Z') start released (measured)"
setsid env PYTHONUNBUFFERED=1 PYTHONPATH=$W/src PADDOCK_BASE_PORT=10400 PADDOCK_CONFIG=$D/paddock_released.toml \
  taskset -c 0-3 nice $P -m paddock.cli run --bench kitscenes --model drivejepa --limit 20 > $D/released.log 2>&1 &
pid=$!; sid=$pid; peak_rss=0; peak_pss=0
while kill -0 $pid 2>/dev/null; do
  rss=0; pss=0
  for p in $(ps -eo pid=,sid= | awk -v s=$sid '$2==s{print $1}'); do
    r=$(awk '/^Rss:/{print $2}' /proc/$p/smaps_rollup 2>/dev/null); q=$(awk '/^Pss:/{print $2}' /proc/$p/smaps_rollup 2>/dev/null)
    rss=$((rss + ${r:-0})); pss=$((pss + ${q:-0}))
  done
  [ $rss -gt $peak_rss ] && peak_rss=$rss; [ $pss -gt $peak_pss ] && peak_pss=$pss
  echo "$(date +%T) rss_kb=$rss pss_kb=$pss" >> $D/released_mem.log
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ $avail -lt 20 ]; then echo "ABORT: available RAM ${avail}G < 20"; kill -INT -- -$sid; fi
  sleep 2
done
wait $pid; rc=$?
echo "$(date '+%F %T %Z') end released rc=$rc peak_rss_gb=$(awk "BEGIN{printf \"%.2f\", $peak_rss/1048576}") peak_pss_gb=$(awk "BEGIN{printf \"%.2f\", $peak_pss/1048576}")"
