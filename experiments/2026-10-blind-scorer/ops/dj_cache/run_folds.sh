#!/usr/bin/env bash
cd {gallops}
export GALLOPS_CONFIG=$PWD/gallops.toml PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D={this_folder}/ops/dj_cache
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for attempt in $(seq 1 200); do
  until [ "$(free_mib)" -ge 4000 ]; do sleep 120; done
  echo "$(date '+%F %T %Z') folds attempt $attempt, $(free_mib) MiB free" >> $D/run_cache.log
taskset -c 4-5 {conda_envs}/drive-jepa/bin/python -m gallops.train.scorer_folds --cache {dj_cache}/v1 \
  --labelled round0_v5 {runs}/labelled_round0_v5 --labelled gt {runs}/labelled_gt --labelled round1_s2 {runs}/labelled_round1_s2 \
  --checkpoint {data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt \
  --out {dj_cache}/v1/folds.json --epochs 20 >> {this_folder}/ops/dj_cache/folds2.log 2>&1
  rc=$?
  [ "$rc" = 0 ] && break
  tail -30 $D/folds2.log | grep -q 'out of memory' || break
  sleep 300
done
echo "$(date '+%F %T %Z') folds rerun exit $rc" >> $D/run_cache.log
