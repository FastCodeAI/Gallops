#!/usr/bin/env bash
# Scorer cache -> labels -> leave-one-scene-out folds, for the sub-score scorer.
#   setsid bash run_cache.sh <vocab.py> <out dir> > run_cache.out 2>&1 < /dev/null &
# Resumable: the cache and the labels skip files that already exist.
set -uo pipefail
VOCAB=$1; OUT=$2
G={gallops}
R={runs}
D={this_folder}/ops/dj_cache
DJ={conda_envs}/drive-jepa/bin/python
GPY={gallops}/.venv/bin/python
CKPT={data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt
LOG=$D/run_cache.log
log() { echo "$(date '+%F %T %Z') $*" >> $LOG; }
export GALLOPS_CONFIG=$G/gallops.toml PYTHONPATH=$G/src PYTHONUNBUFFERED=1
export TMPDIR={paddock_runs}/tmp
mkdir -p $OUT; cp $VOCAB $OUT/vocab.py
log "start: gallops $(git -C $G rev-parse --short HEAD), vocab $VOCAB -> $OUT"

label_pass() {
  OMP_NUM_THREADS=1 taskset -c 2-3 nice $GPY -m gallops.train.scorer_labels --cache $OUT --drivable $R/drivable >> $D/labels.log 2>&1
}

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
# The GPU is shared and often within 2 GB of full: wait for room, and retry after an
# out-of-memory exit. The cache skips episodes already written, so a retry resumes.
NEED_MIB=7000
for attempt in $(seq 1 200); do
  until [ "$(free_mib)" -ge $NEED_MIB ]; do label_pass; sleep 120; done
  log "attempt $attempt: $(free_mib) MiB free, starting cache"
  OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True taskset -c 4-5 $DJ -m gallops.train.scorer_cache \
    --source $R/labelled_round0_v5 round0_v5 --source $R/labelled_gt gt --source $R/labelled_round1_s2 round1_s2 \
    --vocab $OUT/vocab.py --out $OUT --batch 4 >> $D/cache.log 2>&1 &
  CACHE=$!
  while kill -0 $CACHE 2>/dev/null; do
    label_pass || { log "labels failed during cache; see labels.log"; kill $CACHE; exit 1; }
    sleep 300
  done
  wait $CACHE; rc=$?
  log "cache attempt $attempt exit $rc: $(ls $OUT/*/*.npz 2>/dev/null | grep -v labels | wc -l) episode files"
  [ "$rc" = 0 ] && break
  if ! tail -50 $D/cache.log | grep -q "OutOfMemoryError"; then log "cache failed for a reason other than memory; stopping"; exit 1; fi
  sleep 300
done
[ "$rc" = 0 ] || { log "cache never completed"; exit 1; }
label_pass || { log "labels failed; see labels.log"; exit 1; }
log "labels done: $(ls $OUT/*/*.labels.npz | wc -l) files"

OMP_NUM_THREADS=2 taskset -c 4-5 $DJ -m gallops.train.scorer_folds --cache $OUT \
  --labelled round0_v5 $R/labelled_round0_v5 --labelled gt $R/labelled_gt --labelled round1_s2 $R/labelled_round1_s2 \
  --checkpoint $CKPT --out $OUT/folds.json --epochs 20 >> $D/folds.log 2>&1
log "folds exit $?: $OUT/folds.json"
