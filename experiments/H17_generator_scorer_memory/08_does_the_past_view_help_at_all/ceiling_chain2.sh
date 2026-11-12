#!/bin/bash
# Ceiling chain 2 (information only, pre-registration amendment 1): after chain 1 ends, gate -> longer-budget MLP -> its summary. Detached.
M={working_folder}; L=$M/logs; CH=$L/ceiling_chain.log
gate() { python3 -I -c "
import os
l5=os.getloadavg()[1]; ram=[int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith('MemAvailable')][0]/2**20
print(f'load5 {l5:.1f} ram {ram:.1f} GB'); raise SystemExit(0 if (l5 < 32 and ram > 40) else 1)"; }
ram_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
until grep -q "ceiling chain end" $CH; do sleep 30; done
echo "ceiling chain 2 start $(date) pid $$" >> $CH
while :; do
  while ! g=$(gate); do echo "gate closed ($g) $(date)" >> $CH; sleep 120; done; echo "gate open ($g) $(date)" >> $CH
  $M/src/run_model.sh $M/src/s11_mlp_long.py run >> $L/ceiling_mlp_long.log 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    if [ "$(ram_gb)" -lt 20 ]; then echo "ram below 20 GB, stopping $(date)" >> $CH; kill $P; fi
    sleep 30
  done
  wait $P; rc=$?; echo "mlp long rc=$rc $(date)" >> $CH
  [ $rc -eq 0 ] && break
  grep -q "Traceback" $L/ceiling_mlp_long.log && break
done
if [ $rc -eq 0 ]; then $M/src/run_model.sh $M/src/s11_mlp_long.py summary >> $L/ceiling_mlp_long.log 2>&1; echo "mlp long summary rc=$? $(date)" >> $CH; fi
echo "ceiling chain 2 end $(date)" >> $CH
