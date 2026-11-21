#!/bin/bash
# Link-2 chain 2: machine gate -> target statistics -> probes (CPU, taskset 0-1, nice 19 via run_model.sh) -> summary. Detached (setsid nohup).
# The probes are resumable (one file per feature set and slice); if RAM falls below 20 GB the chain stops the probe and waits for the gate.
M={working_folder}; L=$M/logs; CH=$L/link2_chain.log
gate() { python3 -I -c "
import os
l5=os.getloadavg()[1]; ram=[int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith('MemAvailable')][0]/2**20
print(f'load5 {l5:.1f} ram {ram:.1f} GB'); raise SystemExit(0 if (l5 < 32 and ram > 40) else 1)"; }
ram_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
wait_gate() { while ! g=$(gate); do echo "gate closed ($g) $(date)" >> $CH; sleep 120; done; echo "gate open ($g) $(date)" >> $CH; }
echo "link2 chain 2 start $(date) pid $$" >> $CH
wait_gate
$M/src/run_model.sh $M/src/s10_probe.py stats > $L/link2_stats.log 2>&1; echo "stats rc=$? $(date)" >> $CH
while :; do
  wait_gate
  $M/src/run_model.sh $M/src/s10_probe.py run >> $L/link2_probe.log 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    if [ "$(ram_gb)" -lt 20 ]; then echo "ram below 20 GB, stopping probe $(date)" >> $CH; kill $P; fi
    sleep 30
  done
  wait $P; rc=$?; echo "probe rc=$rc $(date)" >> $CH
  [ $rc -eq 0 ] && break
  grep -q "Traceback" $L/link2_probe.log && break
done
if [ $rc -eq 0 ]; then $M/src/run_model.sh $M/src/s10_probe.py summary > $L/link2_summary.log 2>&1; echo "summary rc=$? $(date)" >> $CH; fi
echo "link2 chain 2 end $(date)" >> $CH
