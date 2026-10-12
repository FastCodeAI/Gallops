#!/usr/bin/env bash
# Real nuScenes train scenes -> episodes -> scorer cache v3_nusc (batch 1, swerves + probes) -> labels.
#   setsid bash run_nusc.sh > run_nusc.out 2>&1 < /dev/null &
set -uo pipefail
G={gallops}
R={runs}
D={this_folder}/ops/dj_cache
OUT={dj_cache}/v3_nusc
NUSC={data.nuscenes}
NC={ncap_envs}/ncap/bin/python
DJ={conda_envs}/drive-jepa/bin/python
GPY={gallops}/.venv/bin/python
LOG=$D/run_nusc.log
log() { echo "$(date '+%F %T %Z') $*" >> $LOG; }
export GALLOPS_CONFIG=$G/gallops.toml PYTHONPATH=$G/src PYTHONUNBUFFERED=1 GALLOPS_SCENE_CALIB=$R/calib_nusc
export TMPDIR={paddock_runs}/tmp
log "start: gallops $(git -C $G rev-parse --short HEAD) + uncommitted nuscenes_real/calib cache"
convert() {
  OMP_NUM_THREADS=1 taskset -c $2 nice $NC -m gallops.data.nuscenes_real --nuscenes $NUSC --collect $R/collect_nusc \
    --labelled $R/labelled_nusc --labelled-holdout $R/labelled_nusc_holdout --drivable $R/drivable_nusc --calib $R/calib_nusc \
    --val-scenes {paddock}/src/paddock/adapters/benchmarks/nuscenes/val_scenes.json \
    --shard $1 >> $D/nusc_convert_${1/\//of}.log 2>&1
}
convert 0/2 2 & A=$!; convert 1/2 3 & B=$!; wait $A; ra=$?; wait $B; rb=$?
log "convert exit $ra $rb: $(ls $R/labelled_nusc | wc -l) train + $(ls $R/labelled_nusc_holdout | wc -l) holdout scenes"
[ "$ra" = 0 ] && [ "$rb" = 0 ] || exit 1
mkdir -p $OUT; cp $D/vocab_v1_probes.py $OUT/vocab.py
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for attempt in $(seq 1 200); do
  until [ "$(free_mib)" -ge 7000 ]; do sleep 120; done
  log "cache attempt $attempt: $(free_mib) MiB free"
  OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True taskset -c 4-5 $DJ -m gallops.train.scorer_cache \
    --source $R/labelled_nusc nusc --source $R/labelled_nusc_holdout nusc_holdout --vocab $OUT/vocab.py --out $OUT --batch 1 >> $D/cache_nusc.log 2>&1
  rc=$?; log "cache attempt $attempt exit $rc: $(ls $OUT/*/*.npz 2>/dev/null | grep -vc labels) files"
  [ "$rc" = 0 ] && break
  tail -50 $D/cache_nusc.log | grep -q "OutOfMemoryError" || { log "cache failed, not memory; stopping"; exit 1; }
  sleep 300
done
OMP_NUM_THREADS=1 taskset -c 2 nice $GPY -m gallops.train.scorer_labels --cache $OUT --drivable $R/drivable_nusc --only nusc >> $D/labels_nusc_a.log 2>&1 & A=$!
OMP_NUM_THREADS=1 taskset -c 3 nice $GPY -m gallops.train.scorer_labels --cache $OUT --drivable $R/drivable_nusc --only nusc_holdout --seed 1 >> $D/labels_nusc_b.log 2>&1 & B=$!
wait $A; ra=$?; wait $B; rb=$?
log "labels exit $ra $rb: $(ls $OUT/*/*.labels.npz | wc -l) label files"
[ "$ra" = 0 ] && [ "$rb" = 0 ] && log "NUSC DONE"
