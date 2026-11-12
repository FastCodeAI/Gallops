#!/bin/bash
# Build 5 chain 1: gate -> target statistics (CPU) -> T1-T5 on the version-3 model -> the "before" columns. Detached (setsid nohup).
M={working_folder}; L=$M/logs; CH=$L/build5_chain.log
gate() { python3 -I -c "
import os
l5=os.getloadavg()[1]; ram=[int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith('MemAvailable')][0]/2**20
print(f'load5 {l5:.1f} ram {ram:.1f} GB'); raise SystemExit(0 if (l5 < 32 and ram > 40) else 1)"; }
wait_gate() { while ! g=$(gate); do echo "gate closed ($g) $(date)" >> $CH; sleep 300; done; echo "gate open ($g) $(date) gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> $CH; }
echo "build5 chain 1 start $(date) pid $$" >> $CH
wait_gate; $M/src/run_model.sh $M/src/s9_target_stats.py > $L/build5_stats.log 2>&1; echo "stats rc=$? $(date)" >> $CH
test -f $M/out/recall_target_stats.pt || { echo "no target statistics: stop" >> $CH; exit 1; }
wait_gate; echo "tests start $(date)" >> $CH
MEMORY_VERSION=3 $M/src/run_model.sh $M/src/s6_tests.py all > $L/build5_tests.log 2>&1; echo "tests rc=$? $(date)" >> $CH
wait_gate; $M/src/run_model.sh $M/src/s9_short.py before > $L/build5_before.log 2>&1; echo "before rc=$? $(date)" >> $CH
echo "build5 chain 1 end $(date)" >> $CH
