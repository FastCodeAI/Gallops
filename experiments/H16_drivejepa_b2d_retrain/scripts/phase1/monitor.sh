#!/bin/bash
# 30 s monitor, Sid's s9 monitor.sh with our paths. Run 3A-M has priority: writes STOP (the trainer saves and exits) when host memory or GPU memory runs short. Ends at DONE.
N=$1; R={runs}/b2d_drivejepa/phase1/runs/$N; H={ops}/b2d_drivejepa/phase1
while [ ! -e $R/DONE ]; do
  ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); l=$(cut -d' ' -f2 /proc/loadavg); gf=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "$(date -Is) avail_gb $ma load5 $l gpu_free_mib $gf" >> $H/logs/monitor_$N.log
  if [ ! -e $R/STOP ]; then
    [ "$ma" -lt 8 ] && echo "$(date -Is) host memory: avail $ma GB" > $R/STOP
    [ -n "$gf" ] && [ "$gf" -lt 4096 ] && echo "$(date -Is) GPU memory: free $gf MiB" > $R/STOP
  fi
  sleep 30
done
