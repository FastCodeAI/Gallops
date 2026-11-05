#!/bin/bash
# Detached driver: UNSEEN then SEEN. Each launch resumes from the shards already on the NAS (finished shards are skipped, never redone).
# Waits on the start gate (load_gate.sh) before every launch; retries a set that exits without finishing.
# Start with: setsid nohup ./run_s6_all.sh > /dev/null 2>&1 &
cd "$(dirname "$0")"
for S in unseen seen; do
  until grep -q '^DONE' dump_$S.log 2>/dev/null && [ "$(tail -1 dump_$S.log)" = "exit 0" ]; do
    ./load_gate.sh; echo "=== launch $(date -Is)" >> dump_$S.log
    ./run_s6_dump.sh --set $S --threads 6 --pool 2 --loaders 1 >> dump_$S.log 2>&1; rc=$?; echo "exit $rc" >> dump_$S.log
    [ $rc = 0 ] || sleep 120
  done
done
