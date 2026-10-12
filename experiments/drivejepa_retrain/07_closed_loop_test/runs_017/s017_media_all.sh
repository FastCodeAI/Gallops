#!/usr/bin/env bash
# GENERATOR-ONLY TRIAL media driver (pattern of 07_closed_loop_test/s5_media_all.sh, without its one-scene proof): for every finished 0.1.7 run not yet processed,
# menus (s017_menus.py, model on CPU, one worker) -> overlay video (s017_videos.py). Everything is written to the NAS. Loops until logs/sweep.log says SWEEP DONE and nothing is pending.
set -uo pipefail
BASE={runs}/paddock_sid/Alpasim; T=$BASE/analysis_3.0/Retrain-data/07_closed_loop_test/runs_017; VENV=$BASE/alpasim/.venv/bin/python; N={runs}/drivejepa_retrain/07_closed_loop_test_017
export TMPDIR=$N/tmp PYTHONDONTWRITEBYTECODE=1
log() { echo "[$(date -Is)] $*"; }
cd "$BASE"
# Memory gate (memory note shakaal-swap-gate): start a worker only when the 5-minute load is below 32 AND available RAM is above 40 GB.
gate() {
  local n=0
  while :; do
    l5=$(awk '{print int($2)}' /proc/loadavg); ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$l5" -lt 32 ] && [ "$ma" -gt 40 ]; then log "gate open for $1: load5 $l5, available RAM $ma GB"; return 0; fi
    [ $((n % 10)) = 0 ] && log "gate closed for $1: load5 $l5 (need < 32), available RAM $ma GB (need > 40) -- waiting"
    n=$((n+1)); sleep 120
  done
}
while :; do
  specs_m=(); specs_v=(); done_n=0; fin=0
  while read -r short scene lane; do
    run=$N/runs/drivejepa_017_$short; [ -f "$run/aggregate/metrics_results.parquet" ] || continue; fin=$((fin+1))
    [ -f "$N/menus/d017_$short.json" ] || specs_m+=("d017_$short=$run")
    [ -f "$N/videos/d017_$short.json" ] || specs_v+=("d017_$short=$run:017")
    [ -f "$N/menus/d017_$short.json" ] && [ -f "$N/videos/d017_$short.json" ] && done_n=$((done_n+1))
  done < "$T/tables/scenes_to_run.txt"
  log "finished runs $fin, fully processed $done_n, menus pending ${#specs_m[@]}, videos pending ${#specs_v[@]}"
  if [ ${#specs_m[@]} -gt 0 ]; then gate menus; CUDA_VISIBLE_DEVICES="" nice -n 10 "$VENV" -B "$T/s017_menus.py" "${specs_m[@]}" 2>&1 | grep -v -i --line-buffered warn; fi
  ready=(); for s in "${specs_v[@]}"; do l=${s%%=*}; [ -f "$N/menus/$l.json" ] && ready+=("$s"); done
  if [ ${#ready[@]} -gt 0 ]; then gate videos; nice -n 10 "$VENV" -B "$T/s017_videos.py" "${ready[@]}" 2>&1 | grep -v -i --line-buffered warn; fi
  if grep -q "SWEEP DONE" "$T/logs/sweep.log" 2>/dev/null && [ ${#specs_m[@]} -eq 0 ] && [ ${#specs_v[@]} -eq 0 ]; then log "MEDIA DONE: $done_n scenes"; break; fi
  [ ${#specs_m[@]} -gt 0 ] || [ ${#ready[@]} -gt 0 ] || sleep 120
done
