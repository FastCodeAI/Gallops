#!/usr/bin/env bash
# S5 TRIAL media driver, resumable and safe to run while the sweep is going: for every finished 0.1.6 run not yet processed,
# menus (s5_menus.py, model on CPU, ONE worker, all pending scenes in one invocation = one model load) -> overlay video (s5_videos.py)
# -> standard replays (s5_replays.sh). Loops until logs/sweep.log says SWEEP DONE and nothing is pending. Run in tmux s5_media.
set -uo pipefail
BASE={runs}/paddock_sid/Alpasim; T=$BASE/analysis_3.0/Retrain-data/07_closed_loop_test; VENV=$BASE/alpasim/.venv/bin/python
log() { echo "[$(date -Is)] $*"; }
cd "$BASE"
# Swap gate (Sid, 2026-10-07; memory note {site-host}-swap-gate): start a worker only when the 5-minute load is below 32 AND free swap is
# above 15 GB -- nice does not protect the S5 sweep running beside this. MEDIA_FORCE=1 skips the gate. Waiting is logged every 20 min.
gate() {
  [ "${MEDIA_FORCE:-0}" = 1 ] && return 0; local n=0
  while :; do
    l5=$(awk '{print int($2)}' /proc/loadavg); sw=$(awk '/SwapFree/{print int($2/1048576)}' /proc/meminfo)
    if [ "$l5" -lt 32 ] && [ "$sw" -gt 15 ]; then log "gate open for $1: load5 $l5, free swap $sw GB"; return 0; fi
    [ $((n % 10)) = 0 ] && log "gate closed for $1: load5 $l5 (need < 32), free swap $sw GB (need > 15) -- waiting"
    n=$((n+1)); sleep 120
  done
}
while :; do
  specs_m=(); specs_v=(); done_n=0; fin=0
  while read -r short scene lane; do
    run=$BASE/runs/drivejepa/drivejepa_016_$short; [ -f "$run/aggregate/metrics_results.parquet" ] || continue; fin=$((fin+1))
    [ -f "$T/menus/d016_$short.json" ] || specs_m+=("d016_$short=$run")
    [ -f "$T/videos/d016_$short.json" ] || specs_v+=("d016_$short=$run:016")
    [ -f "$T/menus/d016_$short.json" ] && [ -f "$T/videos/d016_$short.json" ] && done_n=$((done_n+1))
  done < "$T/tables/scenes_to_run.txt"
  log "finished runs $fin, fully processed $done_n, menus pending ${#specs_m[@]}, videos pending ${#specs_v[@]}"
  if [ ${#specs_m[@]} -gt 0 ]; then gate menus; CUDA_VISIBLE_DEVICES="" nice -n 10 "$VENV" "$T/s5_menus.py" "${specs_m[@]}" 2>&1 | grep -v -i --line-buffered warn; fi
  ready=(); for s in "${specs_v[@]}"; do l=${s%%=*}; [ -f "$T/menus/$l.json" ] && ready+=("$s"); done
  if [ ${#ready[@]} -gt 0 ]; then gate videos; nice -n 10 "$VENV" "$T/s5_videos.py" "${ready[@]}" 2>&1 | grep -v -i --line-buffered warn; fi
  "$T/s5_replays.sh"
  # one-time proof of the whole chain on the first scene that has menus + video: provisional row -> page -> headless check
  if [ ! -f "$T/logs/one_scene_proof.log" ]; then
    one=$(ls "$T"/videos/d016_*.json 2>/dev/null | head -1 | sed 's/.*d016_\(.*\)\.json/\1/')
    if [ -n "$one" ]; then
      { "$VENV" "$T/s5_proof_one.py" "$one" 2>&1 | grep -v -i warn; SC=/tmp/{session}/{session-id}/scratchpad
        PLAYWRIGHT_BROWSERS_PATH=$SC/pw-browsers "$SC/pw/bin/python" "$T/s5_check_review.py" "$T/review_standin_one" 2>&1; cat "$T/menus/d016_$one.json"; } > "$T/logs/one_scene_proof.log.tmp" 2>&1
      mv "$T/logs/one_scene_proof.log.tmp" "$T/logs/one_scene_proof.log"; log "one-scene proof written: logs/one_scene_proof.log"
    fi
  fi
  if grep -q "SWEEP DONE" "$T/logs/sweep.log" 2>/dev/null && [ ${#specs_m[@]} -eq 0 ] && [ ${#specs_v[@]} -eq 0 ]; then log "MEDIA DONE: $done_n scenes"; break; fi
  [ ${#specs_m[@]} -gt 0 ] || [ ${#ready[@]} -gt 0 ] || sleep 120
done
