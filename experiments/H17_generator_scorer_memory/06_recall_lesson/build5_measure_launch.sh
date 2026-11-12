#!/bin/bash
M={working_folder}; CH=$M/logs/build5_chain.log
gate() { python3 -I -c "
import os
l5=os.getloadavg()[1]; ram=[int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith('MemAvailable')][0]/2**20
print(f'load5 {l5:.1f} ram {ram:.1f} GB'); raise SystemExit(0 if (l5 < 32 and ram > 40) else 1)"; }
echo "build5 measure chain start $(date)" >> $CH
while ! g=$(gate); do echo "gate closed ($g) $(date)" >> $CH; sleep 300; done; echo "gate open ($g) $(date) gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> $CH
$M/src/run_model.sh $M/out/build5_measure.py > $M/logs/build5_measure.log 2>&1; echo "measure rc=$? $(date)" >> $CH
