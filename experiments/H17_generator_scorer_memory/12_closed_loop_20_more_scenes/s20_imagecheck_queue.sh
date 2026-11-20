#!/bin/bash
# CL20 step 0b (reports/CL20_PREREG.md): simulator-image check. Copy of s12_queue.sh part 1, changed as pre-registered:
#   * output root {local_store}/runs/new40/imagecheck/ (NAS read-only), through the s20 image-check runners
#     (= each model's s20 runner with only OUTROOT changed);
#   * 6 drives: 0.1.8b pass 0.6 on e3ed554f 8023dce5 d2715023 04749bb9, carry on bae463d3 and 0245ff75;
#   * no image-id gate here (the image is being rebuilt); the first drive's docker compose pulls nre-ga:26.04 and builds alpasim-base:0.134.0;
#   * a root-disk watchdog: the rebuild took ~104 GiB on 2026-11-12 and the runners floor root only at 4 GiB. Refuses to start below
#     110 GiB free; during the queue, below 10 GiB free it writes STOP and sends TERM to the running drive (its trap tears down).
# Everything CPU-side under taskset -c 0-1, nice 19 (started that way). A drive with no rollout stops the queue. Compare: s20_imagecheck_cmp.py.
# Nothing after the compare: the 40 drives start only after PASS, by hand, once the queue's image pin is updated (pre-registered).
set -uo pipefail
M="{working_folder}"
G="{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim"
AS={paddock_sid}/Alpasim; SID={paddock_sid}
IC={local_store}/runs/new40/imagecheck
EXT20="$SID/scene_sets/sim_scenes_ext20_26p01.csv"; P10="$AS/drafts/drivejepa-0.1.4/smoke_01d503d4/sim_scenes_pilot10_26p01.csv"
S12F="$G/s10_018b_closed_loop/tables/scenes_12.txt"; S18F="$M/out/cl18_scenes_18.txt"
NRE_DIGEST=sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284
mkdir -p "$IC"; exec >> "$M/logs/cl20_imagecheck_queue.log" 2>&1
log() { echo "[$(date -Is)] $*"; }
free_gib() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
gate() { while :; do L=$(awk '{print int($2)}' /proc/loadavg); R=$(free -g | awk '/^Mem:/{print $7}'); [ -e "$IC/STOP" ] && { log "STOP file"; exit 0; }
  [ "$L" -lt 32 ] && [ "$R" -gt 40 ] && { log "gate open: load5 $L, RAM $R GB, gpu free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) MiB, root free $(free_gib) GiB"; return; }
  log "gate closed: load5 $L, RAM $R GB"; sleep 120; done; }
log "=== IMAGE-CHECK QUEUE START (affinity $(taskset -pc $$ | awk -F': ' '{print $2}'), nice $(ps -o ni= -p $$ | tr -d ' ')) ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader >/dev/null || { log "FATAL: nvidia-smi"; exit 3; }
[ "$(free_gib)" -ge 50 ] || { log "FATAL: root free $(free_gib) GiB < 50 (images built 2026-11-20 00:30; was 110 for the rebuild)"; exit 3; }
[ "$(sha256sum "$EXT20" | cut -c1-8)" = 1b778280 ] && [ "$(sha256sum "$P10" | cut -c1-8)" = 94f50aa3 ] || { log "FATAL: csv pins changed"; exit 3; }
grep -q "^## Simulator-image check" "$M/reports/CL20_PREREG.md" || { log "FATAL: no pre-registration"; exit 3; }
log "pre-registration sha256 $(sha256sum "$M/reports/CL20_PREREG.md" | cut -d' ' -f1)"
[ "$(diff "$AS/adapters/drivejepamemory/launch/run_memory_scene.sh" "$M/src/s20_run_memory_imagecheck.sh" | grep -c '^[<>]')" = 4 ] || { log "FATAL: carry runner differs from the frozen one beyond output root + model folder"; exit 3; }
[ "$(diff "$AS/drafts/drivejepa-0.1.8b-p06/run_018bp06_scene.sh" "$M/src/s20_run_018bp06_imagecheck.sh" | grep -c '^[<>]')" = 2 ] || { log "FATAL: 0.1.8b runner differs beyond the output root"; exit 3; }
( cd "$AS/adapters" && sha256sum -c --quiet drivejepamemory/launch/tables/build_memory.sha256 ) || { log "FATAL: memory adapter build freeze"; exit 3; }
T18="$AS/drafts/drivejepa-0.1.8b-p06"
grep -q '"gate_1_under_2cm": true' "$T18/tables/gate_018bp06.json" && grep -q '"pass": true' "$T18/tables/gate_018bp06.json" || { log "FATAL: 0.1.8b-p06 build gate"; exit 3; }
[ "$(sha256sum {retrain_store}/018b/release/drive_jepa_018b_p06.ckpt | cut -d' ' -f1)" = 3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66 ] || { log "FATAL: 0.1.8b-p06 checkpoint sha256"; exit 3; }
log "image before: $(docker image inspect alpasim-base:0.134.0 --format '{{.Id}}' 2>&1 | head -1); renderer before: $(docker image inspect nvcr.io/nvidia/nre/nre-ga:26.04 --format '{{.Id}}' 2>&1 | head -1)"

CUR=""; (  # root-disk watchdog
  while sleep 30; do f=$(free_gib); [ "$f" -lt 10 ] && { log "WATCHDOG: root free $f GiB < 10 -- STOP + TERM to the running drive"; touch "$IC/STOP"; [ -s "$IC/.current_pid" ] && kill -TERM "$(cat "$IC/.current_pid")" 2>/dev/null; exit 0; }; done ) &
WD=$!; trap 'kill $WD 2>/dev/null' EXIT

run() {  # $1 arm, $2 short
  local arm=$1 short=$2 scene lane csv RD
  read -r _ scene lane < <(grep -h "^$short " "$S12F" "$S18F"); csv="$EXT20"; [ "$lane" = P10 ] && csv="$P10"
  if [ "$arm" = carry ]; then RD="$IC/runs/memory_carry_$short"; else RD="$IC/018bp06/runs/drivejepa_018bp06_$short"; fi
  if [ -e "$RD/rollouts" ]; then log "=== IMAGECHECK $arm $short already has rollouts -- skipped ==="; return; fi
  gate; log "=== IMAGECHECK $arm $short ($lane) ==="
  if [ "$arm" = carry ]; then
    EXT20_SCENES_SHA="$(sha256sum "$csv" | cut -d' ' -f1)" "$M/src/s20_run_memory_imagecheck.sh" dj "$short" "$scene" carry --scenes-csv "$csv" --gpu-floor-mib 8000 < /dev/null &
  else
    DRIVEJEPA_MOMENTUM=on EXT20_SCENES_SHA="$(sha256sum "$csv" | cut -d' ' -f1)" "$M/src/s20_run_018bp06_imagecheck.sh" dj "$short" "$scene" on --scenes-csv "$csv" --gpu-floor-mib 8000 < /dev/null &
  fi
  echo $! > "$IC/.current_pid"; wait $!; rc=$?; : > "$IC/.current_pid"
  log "=== IMAGECHECK $arm $short rc=$rc (root free $(free_gib) GiB) ==="
  [ "$rc" = 0 ] || [ "$rc" = 9 ] || { log "STOP: image check $arm $short rc=$rc -- queue halted"; exit 4; }
  [ -e "$IC/STOP" ] && { log "STOP file"; exit 0; }
}
for s in e3ed554f 8023dce5 d2715023 04749bb9; do run 018bp06 "$s"; done
run carry bae463d3; run carry 0245ff75
d=$(docker image inspect nvcr.io/nvidia/nre/nre-ga:26.04 --format '{{join .RepoDigests " "}}' 2>&1)
log "image after: $(docker image inspect alpasim-base:0.134.0 --format '{{.Id}} {{.Created}}' 2>&1 | head -1); renderer: $d"
case "$d" in *"$NRE_DIGEST"*) log "renderer digest OK ($NRE_DIGEST)";; *) log "FAIL: renderer digest is not $NRE_DIGEST";; esac
"$AS/alpasim/.venv/bin/python" -B "$M/src/s20_imagecheck_cmp.py" > "$M/logs/cl20_imagecheck_cmp.log" 2>&1; log "image check compare: $(tail -1 "$M/logs/cl20_imagecheck_cmp.log")"
log "IMAGE-CHECK QUEUE DONE"
