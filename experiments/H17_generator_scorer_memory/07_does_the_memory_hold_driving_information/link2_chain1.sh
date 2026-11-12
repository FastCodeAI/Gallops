#!/bin/bash
# Link-2 chain 1: machine gate -> forward replays (GPU, batch 1). Retries if the job's own gate refuses. Detached (setsid nohup).
M={working_folder}; L=$M/logs; CH=$L/link2_chain.log
gate() { python3 -I -c "
import os
l5=os.getloadavg()[1]; ram=[int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith('MemAvailable')][0]/2**20
print(f'load5 {l5:.1f} ram {ram:.1f} GB'); raise SystemExit(0 if (l5 < 32 and ram > 40) else 1)"; }
wait_gate() { while ! g=$(gate); do echo "gate closed ($g) $(date)" >> $CH; sleep 120; done; echo "gate open ($g) $(date)" >> $CH; }
echo "link2 chain 1 start $(date) pid $$" >> $CH
while :; do wait_gate; $M/src/run_model.sh $M/src/s10_link2.py replay > $L/link2_replay.log 2>&1; rc=$?; echo "replay rc=$rc $(date)" >> $CH
  grep -q "not starting" $L/link2_replay.log || break; done
echo "link2 chain 1 end $(date)" >> $CH
