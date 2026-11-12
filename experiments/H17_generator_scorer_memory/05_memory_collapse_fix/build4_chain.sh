#!/bin/bash
# Build 4 chain: gate -> T1-T4 on the version-2 model -> "before" column -> gate -> the short check (s8_short.py run). Detached (setsid nohup).
M={working_folder}; L=$M/logs; CH=$L/build4_chain.log
gate() { python3 -I -c "
import os
l5=os.getloadavg()[1]; ram=[int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith('MemAvailable')][0]/2**20
print(f'load5 {l5:.1f} ram {ram:.1f} GB'); raise SystemExit(0 if (l5 < 32 and ram > 40) else 1)"; }
wait_gate() { while ! g=$(gate); do echo "gate closed ($g) $(date)" >> $CH; sleep 300; done; echo "gate open ($g) $(date)" >> $CH; }
gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
nas() { python3 -I -c "
import time,glob,random
fs=sorted(glob.glob('{runs}/generator_scorer_memory/cache/training/*.pt')); random.seed(int(time.time())); r=[]
for f in random.sample(fs,3):
    t=time.time(); n=len(open(f,'rb').read()); r.append(n/2**20/(time.time()-t))
print('NAS MB/s', [round(x,1) for x in r])"; }
echo "build4 chain start $(date) pid $$" >> $CH
wait_gate; echo "tests start $(date) gpu: $(gpu) load: $(cat /proc/loadavg)" >> $CH
MEMORY_VERSION=2 $M/src/run_model.sh $M/src/s6_tests.py all > $L/build4_tests.log 2>&1; echo "tests rc=$? $(date)" >> $CH
python3 -I -c "
import json
t1=json.load(open('$M/out/s6_t1_v2.json')); t3=json.load(open('$M/out/s6_t2t3_v2.json')); t4=json.load(open('$M/out/s6_t4_v2.json'))
ok=t1['pass'] and t3['t3_wiped']['pass'] and t4['carry']['finite_all'] and t4['wiped']['finite_all']
print('T1', t1['pass'], 'T3', t3['t3_wiped']['pass'], 'T4 finite', t4['carry']['finite_all'], t4['wiped']['finite_all']); raise SystemExit(0 if ok else 1)" >> $CH 2>&1 || { echo "tests FAILED: the short check is not launched $(date)" >> $CH; exit 1; }
wait_gate; $M/src/run_model.sh $M/src/s8_short.py before > $L/build4_before.log 2>&1; echo "before rc=$? $(date)" >> $CH
test -f $M/reports/BUILD4_PREREG.md || { echo "no prereg: not launching" >> $CH; exit 1; }
wait_gate; echo "short check start $(date) gpu: $(gpu) load: $(cat /proc/loadavg) $(nas)" >> $CH
$M/src/run_model.sh $M/src/s8_short.py run > $L/build4_short.log 2>&1; echo "short rc=$? $(date)" >> $CH
echo "build4 chain end $(date)" >> $CH
