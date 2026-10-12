#!/usr/bin/env bash
# drivejepa_subscore, relative rule (margin 0.2) + history guard: 0103 x 3 scenarios x 3 runs. Smoke only.
#   setsid bash smoke_relative.sh > smoke_relative.out 2>&1 < /dev/null &
set -uo pipefail
W={paddock}
D={this_folder}/ops/dj_subscore
PY={paddock}/.venv/bin/python
LOG=$D/smoke_relative.log
stamp() { TZ=America/Los_Angeles date '+%F %H:%M %Z'; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0; }
cd $W || exit 1
export PYTHONUNBUFFERED=1 PADDOCK_NCAP_MANAGE_RENDERER=true PADDOCK_BASE_PORT=9970 PYTHONPATH=$W/src PADDOCK_CONFIG=$W/paddock.toml
# The GPU is shared; a renderer that runs out of memory mid-run fails the scenario (HTTP 500). Wait for
# room, run, and retry the whole smoke if any scenario failed, up to 12 times.
for attempt in $(seq 1 12); do
  while [ "$(free_mib)" -lt 21000 ]; do echo "$(stamp) $(free_mib) MiB free; waiting" >> $LOG; sleep 120; done
  echo "=== $(stamp) relative-rule smoke attempt $attempt: 0103 x 3 runs" >> $LOG
  taskset -c 0-3 nice $PY -m paddock.cli run --bench neuroncap --model drivejepa_subscore --scene 0103 --runs 3 > $D/smoke_relative_attempt$attempt.out 2>&1
  rc=$?
  grep -E "ncap_score|FAILED|->" $D/smoke_relative_attempt$attempt.out >> $LOG
  if [ $rc -eq 0 ] && ! grep -q FAILED $D/smoke_relative_attempt$attempt.out; then echo "=== $(stamp) DONE ok (attempt $attempt)" >> $LOG; exit 0; fi
  echo "$(stamp) attempt $attempt had failures; retrying after a wait" >> $LOG; sleep 300
done
echo "=== $(stamp) DONE giving up" >> $LOG
