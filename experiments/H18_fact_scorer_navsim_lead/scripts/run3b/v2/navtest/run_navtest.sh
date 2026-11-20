#!/bin/bash
# navtest EPDMS for run 3B v2 (20 Nov; Navvrat: "evaluate the model .. asap"): navsim_v2 run_pdm_score.py, the procedure of
# run 3A's 87.30 (eval_run3a_ep13_tuned, 25 Oct): train_test_split=navtest, metric_cache_v2, 4 process workers, no
# momentum-aware selection. Agent run3b_agent.Run3BV2Agent (run3b/v2/navtest), config from hydra/agent via searchpath.
# Data: navtest_root (Naren's maps + navsim_logs, local sensor blobs since 10 Nov). Usage: run_navtest.sh <name> <ckpt> [extra overrides]
set -u
NAME=$1; CK=$2; shift 2
DK={data.drivejepa_tree}/navsim_v2
N={data.paddock_dump}/navsim
export NAVSIM_DEVKIT_ROOT=$DK NAVSIM_EXP_ROOT={data.paddock_data}/run3b/navtest OPENSCENE_DATA_ROOT=$N/navtest_root NUPLAN_MAPS_ROOT=$N/navtest_root/maps \
  NUPLAN_MAP_VERSION=nuplan-maps-v1.0 PYTHONPATH=$DK:{runs}/run3b/v2/navtest OMP_NUM_THREADS=2 \
  WAJEPA_CODE={data.wajepa_code} \
  WAJEPA_WEIGHTS={data.paddock_dump}/weights/WA-JEPA/model_state_dict.pt
cd $DK
echo "START $(date '+%F %T') $NAME $CK"
{runs}/mmcv-sm120/.venv/bin/python -u navsim/planning/script/run_pdm_score.py \
  'hydra.searchpath=[pkg://navsim.planning.script.config.common,file://{runs}/run3b/v2/navtest/hydra]' \
  train_test_split=navtest agent=run3b_v2_agent agent.checkpoint_path=$CK \
  metric_cache_path=$N/metric_cache_v2 experiment_name=$NAME \
  worker=single_machine_thread_pool worker.max_workers=4 worker.use_process_pool=true "$@"
echo "EXIT $? $(date '+%F %T')"
