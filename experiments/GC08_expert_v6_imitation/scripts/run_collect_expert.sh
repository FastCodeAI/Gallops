#!/usr/bin/env bash
# Expert v6 drives all 150 training episodes through the real engine + renderer (lossless frames), the
# model's 32 candidates dumped beside; then lane references and v6 labels -> labelled_expert_v6.
# Resumable: one .done per episode in state_expert. Each attempt waits for MIN_FREE_GB.
#   setsid bash run_collect_expert.sh > run_collect_expert.out 2>&1 < /dev/null &
set -uo pipefail
OPS={paddock_runs}/ops/dj_collect
R={runs}
G={repos}/gallops-expert6
PY={repos}/paddock/.venv/bin/python
NCAPPY={ncap_env}/bin/python
LOG=$OPS/run_collect_expert.log
MIN_FREE_GB=${MIN_FREE_GB:-20}
export COLLECT_PADDOCK_SRC={repos}/paddock-dj-ncapdump-camfix
export PYTHONPATH=$COLLECT_PADDOCK_SRC/src NCAP_LOSSLESS_FRAMES=1 TMPDIR={paddock_runs}/tmp
log() { echo "$(TZ=America/Los_Angeles date '+%F %H:%M %Z') $*" >> $LOG; }
cd $OPS || exit 1
stalls=0
for attempt in $(seq 1 12); do
  n=$(ls $OPS/state_expert/*.done 2>/dev/null | wc -l)
  [ "$n" -ge 150 ] && break
  [ $stalls -ge 4 ] && { log "gave up: 4 attempts without progress ($n of 150)"; exit 1; }
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt $((MIN_FREE_GB * 1024)) ]; do sleep 60; done
  log "attempt $attempt ($n of 150 done)"
  PADDOCK_CONFIG=$OPS/paddock.collect.expert.toml setsid taskset -c 10-15 nice $PY collect_expert.py --tag expert \
    --scenarios $R/scenarios_gt --current $OPS/expert_current.txt >> $OPS/collect_expert.log 2>&1 < /dev/null &
  pid=$!
  setsid nohup $OPS/group_pause_guard.sh $pid $OPS/group_pause_guard_expert.log > /dev/null 2>&1 < /dev/null &
  wait $pid
  m=$(ls $OPS/state_expert/*.done 2>/dev/null | wc -l)
  log "attempt $attempt ended ($m of 150 done)"
  if [ "$m" -gt "$n" ]; then stalls=0; else stalls=$((stalls + 1)); sleep 120; fi
done
export GALLOPS_CONFIG=$G/gallops.toml PYTHONPATH=$G/src OMP_NUM_THREADS=1
log "lane references"
taskset -c 0-1 $NCAPPY -m gallops.expert.lanes --collect $R/collect_expert >> $OPS/label_expert.log 2>&1 || { log "lanes failed"; exit 1; }
log "labels (expert v6)"
taskset -c 0-1 {repos}/gallops/.venv/bin/python -m gallops.expert.label --expert v6 \
  --collect $R/collect_expert --dumps $R/collect_dumps_expert --out $R/labelled_expert_v6 \
  --drivable $R/drivable --packs $R/scene_packs >> $OPS/label_expert.log 2>&1 || { log "label failed"; exit 1; }
log "done: $(ls $R/labelled_expert_v6/*.jsonl | wc -l) labelled episodes"
