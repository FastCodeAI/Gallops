#!/usr/bin/env bash
# Cache v2: batch 1 (the vendor model mixes ticks within a batch; batch 1 reproduces what it drove
# to 5e-5 m, batch 4 drifted up to 1.0 m), vocab = 32 swerves + 32 far-ahead probes. Labels in 2 shards.
#   setsid bash run_cache_v2.sh > run_cache_v2.out 2>&1 < /dev/null &
set -uo pipefail
OUT={dj_cache}/v2
G={gallops}
R={runs}
D={this_folder}/ops/dj_cache
DJ={conda_envs}/drive-jepa/bin/python
GPY={gallops}/.venv/bin/python
LOG=$D/run_cache_v2.log
log() { echo "$(date '+%F %T %Z') $*" >> $LOG; }
export GALLOPS_CONFIG=$G/gallops.toml PYTHONPATH=$G/src PYTHONUNBUFFERED=1
export TMPDIR={paddock_runs}/tmp
mkdir -p $OUT; cp $D/vocab_v1_probes.py $OUT/vocab.py
log "start v2: gallops $(git -C $G rev-parse --short HEAD) (+uncommitted --only), batch 1"
labels() {
  OMP_NUM_THREADS=1 taskset -c 2-3 nice $GPY -m gallops.train.scorer_labels --cache $OUT --drivable $R/drivable --only round0_v5 >> $D/labels_v2a.log 2>&1 &
  local a=$!
  OMP_NUM_THREADS=1 taskset -c 6-7 nice $GPY -m gallops.train.scorer_labels --cache $OUT --drivable $R/drivable --only gt round1_s2 --seed 1 >> $D/labels_v2b.log 2>&1 &
  local b=$!
  wait $a; local ra=$?; wait $b; local rb=$?
  [ "$ra" = 0 ] && [ "$rb" = 0 ]
}
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for attempt in $(seq 1 200); do
  until [ "$(free_mib)" -ge 7000 ]; do labels; sleep 120; done
  log "attempt $attempt: $(free_mib) MiB free"
  OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True taskset -c 4-5 $DJ -m gallops.train.scorer_cache \
    --source $R/labelled_round0_v5 round0_v5 --source $R/labelled_gt gt --source $R/labelled_round1_s2 round1_s2 \
    --vocab $OUT/vocab.py --out $OUT --batch 1 >> $D/cache_v2.log 2>&1 &
  CACHE=$!
  while kill -0 $CACHE 2>/dev/null; do labels || { log "labels failed"; kill $CACHE; exit 1; }; sleep 240; done
  wait $CACHE; rc=$?
  log "cache attempt $attempt exit $rc: $(ls $OUT/*/*.npz 2>/dev/null | grep -vc labels) files"
  [ "$rc" = 0 ] && break
  tail -50 $D/cache_v2.log | grep -q "OutOfMemoryError" || { log "cache failed, not memory; stopping"; exit 1; }
  sleep 300
done
labels || { log "labels failed"; exit 1; }
log "v2 done: $(ls $OUT/*/*.labels.npz | wc -l) label files"
