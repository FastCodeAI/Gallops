#!/bin/bash
# Approach 4 closed loop (Sid's terms 2026-11-20; PREREG_APPROACH4.md section 2). One detached queue (tmux a4cl), CPU side pinned
# taskset -c 0-1 nice -n 19 (started that way). In order, each step only after the previous passed:
#   0. wait until approach 3's cl20ic queue has finished (never run alongside it) and no wizard of ours is running
#   1. image check, the 4 scenes not covered by cl20ic: p06 on 791d68ce 026d6a39 02e075b9 89f2eeb8 (runner = p06's, output root only);
#      then the compare on all 8 (the 4 reused ones already passed (a)-(c)). Any difference: STOP, drive nothing else.
#   2. wiring proof: step-0 snapshot through drivejepa_a4 on e3ed554f, bit-identical to p06 there, or STOP
#   3. arm A (p00.2) on the 12, preview table, then the 18; 4. arm B (p00.6) on all 30, same order; final tables + gate.
# Every scene: GPU >= 12 GB free (runner preflight, twice 60 s apart, waits without a cap) and the memory gate. One run per scene; a drive
# that ends without a rollout, or with a preflight/provenance exit, stops the queue (no re-run). Status: runs/STATUS.txt.
set -uo pipefail
MH={working_folder}
AS={paddock_sid}/Alpasim; SID={paddock_sid}
L="$AS/adapters/drivejepa_a4/launch"; CL={approach4_store}/cl
G="{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim"
S12F="$G/s10_018b_closed_loop/tables/scenes_12.txt"; S18F={working_notes}/memory/3_generator_scorer_memory_PAI/out/cl18_scenes_18.txt
EXT20="$SID/scene_sets/sim_scenes_ext20_26p01.csv"; P10="$AS/drafts/drivejepa-0.1.4/smoke_01d503d4/sim_scenes_pilot10_26p01.csv"
VENV="$AS/alpasim/.venv/bin/python"; REPORT="$MH/reports/CL_ARM_A.md"; ST="$CL/STATUS.txt"
mkdir -p "$CL"; exec >> "$CL/queue.log" 2>&1
log() { echo "[$(date -Is)] $*"; }
status() { echo "$(date '+%F %T') $*" > "$ST"; }
stop() { log "STOP: $*"; status "STOPPED: $*"; printf '\n## STOPPED (%s)\n\n%s\n' "$(date '+%F %T')" "$*" >> "$REPORT"; exit 4; }
gate() { while :; do l=$(awk '{print int($2)}' /proc/loadavg); r=$(free -g | awk '/^Mem:/{print $7}')
  [ -e "$CL/STOP" ] && { log "STOP file"; status "STOP file"; exit 0; }
  [ "$l" -lt 32 ] && [ "$r" -gt 40 ] && return; log "memory gate closed: load5 $l, RAM $r GB"; sleep 120; done; }
log "=== QUEUE START (affinity $(taskset -pc $$ | awk -F': ' '{print $2}'), nice $(ps -o ni= -p $$ | tr -d ' ')) ==="
[ "$(sha256sum "$EXT20" | cut -c1-8)" = 1b778280 ] && [ "$(sha256sum "$P10" | cut -c1-8)" = 94f50aa3 ] || stop "scene csv pins changed"
[ "$(sha256sum "$MH/PREREG_APPROACH4.md" | cut -d' ' -f1)" = f8f4b975edaf712fcc02342c151e21303337d1dfa0880f64385e0d9bbb0ba450 ] || stop "pre-registration sha256 changed"
[ "$(docker image inspect alpasim-base:0.134.0 --format '{{.Id}}')" = sha256:ab16bc78a4686dcd63c2b77a77fe585df3846528a61b552db6b71f9553983a44 ] || stop "alpasim-base is not ab16bc78"
[ "$(docker image inspect nvcr.io/nvidia/nre/nre-ga:26.04 --format '{{.Id}}')" = sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284 ] || stop "nre-ga is not 97f43e71"

# Sid 2026-11-20: approach 3's queues go first (cl20ic image check, then its 40-drive cl20 queue). Wait until neither queue process is
# alive and no wizard of ours runs, on two checks 5 min apart (a queue's tmux session can outlive its queue, so the process is what counts).
# match only a running queue script (a tmux server's title keeps its launch command line, so "cl20ic" matched long after its queue ended)
busy() { pgrep -u "$(id -un)" -f "^bash [^ ]*s20_(imagecheck_)?queue\.sh" >/dev/null || pgrep -u "$(id -un)" -f "^[^ ]*python[^ ]* [^ ]*alpasim_wizard" >/dev/null; }
while :; do
  while busy; do status "waiting for approach 3's cl20 queue to finish ($(grep -c '=== RUN' {working_notes}/memory/3_generator_scorer_memory_PAI/logs/cl20_queue.log 2>/dev/null) drive starts logged)"; sleep 60; done
  sleep 300; busy || break
done
log "approach 3's queues finished (cl20: $(tail -1 {working_notes}/memory/3_generator_scorer_memory_PAI/logs/cl20_queue.log | cut -c1-120)); image $(docker image inspect alpasim-base:0.134.0 --format '{{.Id}} {{.Created}}')"

drive() {  # $1 kind (ic|step0A|armA|armB), $2 short
  local kind=$1 short=$2 scene lane csv rd rc
  read -r _ scene lane < <(grep -h "^$short " "$S12F" "$S18F"); csv="$EXT20"; [ "$lane" = P10 ] && csv="$P10"
  if [ "$kind" = ic ]; then rd="$CL/imagecheck/runs/drivejepa_018bp06_$short"; else rd="$CL/runs/a4_${kind}_$short"; fi
  [ -e "$rd/rollouts" ] && { log "=== $kind $short already has rollouts: kept as is, not re-run ==="; return 0; }
  gate; status "$kind $short driving (started $(date +%T))"; log "=== DRIVE $kind $short ($lane) ==="
  if [ "$kind" = ic ]; then
    DRIVEJEPA_MOMENTUM=on EXT20_SCENES_SHA="$(sha256sum "$csv" | cut -d' ' -f1)" "$L/run_018bp06_imagecheck_a4.sh" dj "$short" "$scene" on \
      --scenes-csv "$csv" --gpu-floor-mib 12288 --wait-cap-s 864000 < /dev/null; rc=$?
  else
    EXT20_SCENES_SHA="$(sha256sum "$csv" | cut -d' ' -f1)" "$L/run_a4_scene.sh" dj "$short" "$scene" "$kind" \
      --scenes-csv "$csv" --gpu-floor-mib 12288 --wait-cap-s 864000 < /dev/null; rc=$?
  fi
  log "=== DRIVE $kind $short rc=$rc score $(python3 -c "import json;print(json.load(open('$rd/aggregate/results-summary.json'))['rollouts'][0]['score'])" 2>/dev/null) ==="
  { [ "$rc" = 0 ] || [ "$rc" = 9 ]; } && [ -f "$rd/aggregate/results-summary.json" ] || stop "$kind $short ended rc=$rc without a scored rollout (no re-run; see $rd)"
}

# 1. image check
for s in 791d68ce 026d6a39 02e075b9 89f2eeb8; do drive ic "$s"; done
"$VENV" "$MH/src/cl_imagecheck_cmp.py" > "$CL/imagecheck_cmp.txt" 2>&1
grep -q "IMAGE CHECK PASS" "$CL/imagecheck_cmp.txt" || stop "image check FAILED on the 8 scenes ($CL/imagecheck_cmp.txt, MH/out/cl_imagecheck.json): nothing else driven"
log "image check PASS on 8"; { echo; echo "## Image check: PASS on 8"; echo; echo '```'; cat "$CL/imagecheck_cmp.txt"; echo '```'; } >> "$REPORT"
# 2. wiring proof
drive step0A e3ed554f
"$VENV" "$MH/src/cl_wiring_cmp.py" > "$CL/wiring_cmp.txt" 2>&1
grep -q "WIRING PROOF PASS" "$CL/wiring_cmp.txt" || stop "wiring proof FAILED ($CL/wiring_cmp.txt): no arm driven"
log "wiring proof PASS"; { echo; echo "## Wiring proof: PASS"; echo; echo '```'; cat "$CL/wiring_cmp.txt"; echo '```'; } >> "$REPORT"
# 3. arm A, 4. arm B
for s in $(cut -d' ' -f1 "$S12F"); do drive armA "$s"; done
python3 "$MH/src/cl_scores.py" preview > /dev/null
for s in $(cut -d' ' -f1 "$S18F"); do drive armA "$s"; done
for s in $(cut -d' ' -f1 "$S12F") $(cut -d' ' -f1 "$S18F"); do drive armB "$s"; done
python3 "$MH/src/cl_scores.py" final > /dev/null
log "=== QUEUE DONE ==="; status "queue done: arms A and B on 30 (see reports/CL_ARM_A.md)"
