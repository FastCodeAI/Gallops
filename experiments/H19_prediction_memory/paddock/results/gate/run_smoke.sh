#!/bin/bash
# H15 gate D2 smoke: KITScenes, 20 keyframes, released / zero nudge / (a) / (b), one after another.
# Guard inside: 5-min load < 32, available RAM > 9.8 GB (Sid 6 Nov: 1.5 x the 6.52 GB peak of the measured
# released run, measure_released.sh), GPU free >= 12 GB; waits (polls 60 s) up to 12 h. Released ran (measured).
set -u
W={paddock}; D={runs}/smoke
P={paddock_python}
declare -A MEM=([released]="" [zero]={runs}/zero-nudge-a-L4
  [a]={weights}/drivejepa_memory/h15-r3-a-L4-s0/step_000001638
  [b]={weights}/drivejepa_memory/h15-r3-b-L0-s0/step_000001638)
guard() {
  for i in $(seq 720); do
    load=$(awk '{print $2}' /proc/loadavg); avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
    gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "$(date '+%F %T %Z') guard load=$load avail=${avail}G gpu=${gpu}MiB"
    if awk "BEGIN{exit !($load < 32 && $avail > 9.8 && $gpu >= 12288)}"; then return 0; fi
    sleep 60
  done
  echo "guard never opened"; exit 3
}
cd $W
for arm in zero a b; do
  [ -f $D/$arm.done ] && { echo "skip $arm: done"; continue; }
  guard
  cfg=$D/paddock_$arm.toml; cp $W/paddock.toml $cfg
  if [ -n "${MEM[$arm]}" ]; then
    sed -i "s|^\[models.drivejepa.env\]|[models.drivejepa.env]\nDRIVEJEPA_MEMORY = \"${MEM[$arm]}\"|" $cfg
  fi
  echo "$(date '+%F %T %Z') start $arm"
  PYTHONUNBUFFERED=1 PYTHONPATH=$W/src PADDOCK_BASE_PORT=10400 PADDOCK_CONFIG=$cfg taskset -c 0-3 nice \
    $P -m paddock.cli run --bench kitscenes --model drivejepa --limit 20 > $D/$arm.log 2>&1
  rc=$?; echo "$(date '+%F %T %Z') end $arm rc=$rc"
  [ $rc -ne 0 ] && { echo "STOP: $arm failed"; exit 4; }
  touch $D/$arm.done
done
echo DONE
