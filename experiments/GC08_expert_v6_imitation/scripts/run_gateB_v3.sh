#!/usr/bin/env bash
# Gate B again at T=3.5 after: only clear swerves (tight-pass fallback), two-frame visibility, plus v2's
# measurement fixes; one process per scene, then merged into gateB_v3_T3.5.json.
#   setsid bash run_gateB_v3.sh > run_gateB_v3.out 2>&1 < /dev/null &
set -uo pipefail
O={paddock_runs}/ops/dj_expert
R={runs}
PY={ncap_env}/bin/python
cd {repos}/neuro-ncap || exit 1
export PYTHONPATH={repos}/gallops-expert6/src:{repos}/neuro-ncap OMP_NUM_THREADS=1
cores=(0-1 2-3 4-5 6-7 8-9); i=0; parts=()
for S in 0053 0058 0064 0070 0075; do
  taskset -c ${cores[$i]} nice $PY -m gallops.expert.rollout expert --collect $R/collect_r0cam --packs $R/scene_packs \
    --scenarios $R/scenarios_gt --drivable $R/drivable --trigger-s 3.5 --scene $S \
    --out $O/gateB_v3_T3.5_$S.json > $O/gateB_v3_T3.5_$S.log 2>&1 &
  parts+=($O/gateB_v3_T3.5_$S.json); i=$((i + 1))
done
wait
$PY -m gallops.expert.rollout merge "${parts[@]}" --out $O/gateB_v3_T3.5.json > $O/gateB_v3_merge.log 2>&1
echo "$(TZ=America/Los_Angeles date '+%F %H:%M %Z') done"
