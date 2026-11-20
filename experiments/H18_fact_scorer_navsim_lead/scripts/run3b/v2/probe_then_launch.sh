#!/bin/bash
# 19 Nov: eval-path check (held-out report at step 3, 32 + 32 samples), then the 48 h run if the report was written.
cd {runs}/run3b
export TORCHINDUCTOR_CACHE_DIR={runs}/run3b/v2/inductor_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CAP_GB=50 RUN_CWD={runs}/run3b/v2
./capped_pss.sh train3.py --name v2_probe_eval2_19nov --max-steps-probe 5 --smoke-step 3 --eval-n 32 > logs/v2_probe_eval2_19nov.log 2>&1
R={data.paddock_data_nohome}/run3b/runs_v2/v2_probe_eval2_19nov; R={data.home}${R#/home}
if ls $R/eval_step000003_smoke.json >/dev/null 2>&1 && ! ls $R/*FAILED* >/dev/null 2>&1; then
  echo "$(date +%T) eval ok; launching run3b_v2b_19nov"
  ./capped_pss.sh train3.py --name run3b_v2b_19nov --hours 48 > logs/run3b_v2b_19nov.log 2>&1 &
  echo launched
else
  echo "$(date +%T) eval check FAILED or missing; not launching"; ls $R
fi
