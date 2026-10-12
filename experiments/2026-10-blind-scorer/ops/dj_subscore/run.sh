#!/usr/bin/env bash
# drivejepa_subscore on neuroNCAP. Smoke: 0103 x 3 scenarios x 3 runs. The full 20-pair x 10-run sweep follows
# only if stationary-0103 and side-0103 have no collision (released: 0/10 and 0/10 there). Detached, resumable.
#   setsid bash run.sh > run.out 2>&1 < /dev/null &
set -uo pipefail
W={paddock}
D={this_folder}/ops/dj_subscore
RAW={paddock_runs}/neuroncap/drivejepa_subscore
PY={paddock}/.venv/bin/python
SCENES=0099,0101,0103,0106,0108,0110,0278,0331,0346,0783,0796,0921,0923,0966
LOG=$D/run.log
NEED_MIB=21000
stamp() { TZ=America/Los_Angeles date '+%F %H:%M %Z'; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0; }
wait_gpu() { while [ "$(free_mib)" -lt "$NEED_MIB" ]; do echo "$(stamp) $(free_mib) MiB free, need $NEED_MIB; waiting" >> $LOG; sleep 120; done; }
cd $W || exit 1
export PYTHONUNBUFFERED=1 PADDOCK_NCAP_MANAGE_RENDERER=true PADDOCK_BASE_PORT=9970 PYTHONPATH=$W/src PADDOCK_CONFIG=$W/paddock.toml
paddock() { taskset -c 0-3 nice $PY -m paddock.cli "$@"; }

if [ ! -f $D/smoke.ok ]; then
  wait_gpu
  echo "=== $(stamp) smoke: 0103 x 3 runs" >> $LOG
  paddock run --bench neuroncap --model drivejepa_subscore --scene 0103 --runs 3 > $D/smoke.out 2>&1; rc=$?
  grep -E "ncap_score|->|FAILED" $D/smoke.out >> $LOG; echo "$(stamp) smoke exit $rc" >> $LOG
  [ $rc -eq 0 ] || { echo "$(stamp) smoke failed; stopping" >> $LOG; exit 1; }
  if grep -qE "^\s+(stationary|side)-0103: collision_rate 0," $D/smoke.out && [ "$(grep -cE '^\s+(stationary|side)-0103: collision_rate 0,' $D/smoke.out)" -eq 2 ]; then
    touch $D/smoke.ok
  else
    echo "$(stamp) SMOKE GATE: stationary or side collided; not starting the sweep" >> $LOG; exit 0
  fi
fi
before=$(ls -d $RAW/2026* 2>/dev/null | wc -l)
wait_gpu
echo "=== $(stamp) full sweep: $SCENES x 10 runs" >> $LOG
paddock run --bench neuroncap --model drivejepa_subscore --scene "$SCENES" --runs 10 >> $LOG 2>&1; rc=$?
id=$(ls -dt $RAW/2026* | head -1 | xargs basename)
for attempt in 1 2 3; do
  [ $rc -eq 0 ] && break
  [ "$(ls -d $RAW/2026* | wc -l)" -gt "$before" ] || { echo "$(stamp) no run dir was made; not resuming" >> $LOG; break; }
  wait_gpu; echo "$(stamp) resuming $id (attempt $attempt)" >> $LOG
  paddock run --resume "$id" --bench neuroncap --model drivejepa_subscore --scene "$SCENES" --runs 10 >> $LOG 2>&1; rc=$?
done
echo "=== $(stamp) DONE rc=$rc run=$id" >> $LOG
