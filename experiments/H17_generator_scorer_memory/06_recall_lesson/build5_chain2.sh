#!/bin/bash
# Build 5 chain 2 (launched only after reports/BUILD5_PREREG.md is written): gate -> the short check up to the planned stop at step 310
# -> gate -> the same run resumed (--resume) to step 930. Detached (setsid nohup).
M={working_folder}; L=$M/logs; CH=$L/build5_chain.log
gate() { python3 -I -c "
import os
l5=os.getloadavg()[1]; ram=[int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith('MemAvailable')][0]/2**20
print(f'load5 {l5:.1f} ram {ram:.1f} GB'); raise SystemExit(0 if (l5 < 32 and ram > 40) else 1)"; }
wait_gate() { while ! g=$(gate); do echo "gate closed ($g) $(date)" >> $CH; sleep 300; done; echo "gate open ($g) $(date) gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> $CH; }
nas() { python3 -I -c "
import time,glob,random
fs=sorted(glob.glob('{runs}/generator_scorer_memory/cache/training/*.pt')); random.seed(int(time.time())); r=[]
for f in random.sample(fs,3):
    t=time.time(); n=len(open(f,'rb').read()); r.append(n/2**20/(time.time()-t))
print('NAS MB/s', [round(x,1) for x in r])"; }
test -f $M/reports/BUILD5_PREREG.md || { echo "no prereg: not launching" >> $CH; exit 1; }
echo "build5 chain 2 start $(date) pid $$" >> $CH
wait_gate; echo "short check part 1 start $(date) load: $(cat /proc/loadavg) $(nas)" >> $CH
$M/src/run_model.sh $M/src/s9_short.py run > $L/build5_short_part1.log 2>&1; rc=$?; echo "short part 1 rc=$rc $(date)" >> $CH
# the job re-checks the gate itself after its imports; if the load moved meanwhile it refuses, having done nothing: wait and start again
while grep -q "not starting" $L/build5_short_part1.log; do wait_gate; $M/src/run_model.sh $M/src/s9_short.py run > $L/build5_short_part1.log 2>&1; echo "short part 1 (again) rc=$? $(date)" >> $CH; done
grep -q "PLANNED STOP" $L/build5_short_part1.log || { echo "part 1 did not reach the planned stop: not resuming automatically" >> $CH; exit 1; }
wait_gate; echo "short check part 2 (resume) start $(date) load: $(cat /proc/loadavg) $(nas)" >> $CH
$M/src/run_model.sh $M/src/s9_short.py run --resume > $L/build5_short_part2.log 2>&1; echo "short part 2 rc=$? $(date)" >> $CH
while grep -q "not starting" $L/build5_short_part2.log; do wait_gate; $M/src/run_model.sh $M/src/s9_short.py run --resume > $L/build5_short_part2.log 2>&1; echo "short part 2 (again) rc=$? $(date)" >> $CH; done
echo "build5 chain 2 end $(date)" >> $CH
