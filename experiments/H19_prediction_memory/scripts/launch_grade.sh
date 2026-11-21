#!/bin/bash
# H15 step 2: grade the 32 released proposals at every graded moment, one batch of logs at a time.
#
#   bash launch_grade.sh val          # batch 1: the 214 val logs (the report set)
#   bash launch_grade.sh selection    # batch 2: the 100 selection logs
#   bash launch_grade.sh rest         # batch 3: the other 878 train logs
#   bash launch_grade.sh val selection   # several, one after the other, in one detached job
#
# Starts detached (setsid nohup) and returns. Resumable per log: re-running a batch continues it.
# Cores 0-3 at nice 19, 4 worker processes (Sid D17). Every location comes from the site config
# (gallops.toml): the cache root ({data.navtrain.cache}), the Drive-JEPA interpreter ([python]).
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 val|selection|rest [...]"; exit 2; }
for b in "$@"; do case $b in val|selection|rest) ;; *) echo "unknown batch: $b"; exit 2 ;; esac; done
WT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
site() { "$WT/.venv/bin/python" -c "from gallops.config import SiteConfig; c = SiteConfig.load(); print($1)"; }
cd "$WT"
CACHE=$(site 'c.resolve("{data.navtrain.cache}")')
E=$(site 'c.python["drivejepa"]')
OUT=$CACHE/grades
LOGDIR=$CACHE/logs
TMP=$(dirname "$CACHE")/tmp
SEL=$WT/experiments/H15_prediction_memory/selection_logs.txt
mkdir -p "$LOGDIR" "$TMP"
STAMP=$(TZ=America/Los_Angeles date '+%Y%m%dT%H%M%S%Z')
LOG=$LOGDIR/grade_$(IFS=-; echo "$*")_$STAMP.log

# Refuse a second copy.
if pgrep -f "^[^ ]*/bin/python -m gallops.memory.grade build" >/dev/null; then
  echo "a grade build is already running:"; pgrep -af "gallops.memory.grade build"; exit 1
fi
# The code that runs must be committed, so manifest.json's git_sha names it.
if [ -n "$(git -C "$WT" status --porcelain -- src tests)" ]; then
  echo "uncommitted changes under src/ or tests/ in $WT; commit first"; exit 1
fi
# Memory gate on shakaal: 5-min load < 32 and available RAM > 40 GB.
LOAD5=$(awk '{print $2}' /proc/loadavg)
AVAIL_GB=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if awk -v l="$LOAD5" 'BEGIN{exit !(l >= 32)}' || [ "$AVAIL_GB" -le 40 ]; then
  echo "memory gate closed: 5-min load $LOAD5, available ${AVAIL_GB} GB (need < 32 and > 40)"; exit 1
fi

RUNS=""
for b in "$@"; do
  RUNS="$RUNS taskset -c 0-3 nice -n 19 $E -m gallops.memory.grade build --batch $b --workers 4 --out $OUT --selection $SEL || exit \$?;"
done
setsid nohup env TMPDIR="$TMP" PYTHONPATH="$WT/src" TZ=America/Los_Angeles \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  bash -c "echo \"start \$(date '+%F %T %Z') commit \$(git rev-parse --short HEAD) batches $*\"; \
           ( $RUNS ); echo \"exit \$? at \$(date '+%F %T %Z')\"" \
  > "$LOG" 2>&1 < /dev/null &
disown
echo "launched batches: $*; log $LOG"
echo "status: bash $(dirname "$0")/status_grade.sh"
