#!/bin/bash
# 18-scene closed loop (reports/CL18_MEMORY_PREREG.md): the 18 scenes of the 30-scene set not among the 12 diagnosis scenes, each with
#   carry   the frozen memory adapter (0.1.8b-p06-memory5, mode carry) via s18_run_memory_scene.sh (output root and model folder changed only)
#   0.1.8b  its own 0.1.8b-p06 adapter via s18_run_018bp06_scene.sh (output root changed only)
# in that order per scene. Everything CPU-side pinned to cores 0-1 at nice 19 (taskset is inherited by the wizard, driver and model worker;
# the simulator containers are started by the docker daemon and are not pinned). Each step waits for the machine gate. Resumable: a run whose
# folder holds rollouts is skipped. A run with no rollout stops the queue (no retry loop). STOP file {experiment_store}/runs/cl18/STOP.
set -uo pipefail
M="{working_folder}"
AS={paddock_sid}/Alpasim; SID={paddock_sid}
CL18={experiment_store}/runs/cl18
EXT20="$SID/scene_sets/sim_scenes_ext20_26p01.csv"; P10="$AS/drafts/drivejepa-0.1.4/smoke_01d503d4/sim_scenes_pilot10_26p01.csv"
S18F="$M/out/cl18_scenes_18.txt"; IMAGE=sha256:9c5909dafcd5574dca1c783e6abc29dfbdfcc9fc2860725dd80030475b8f8881
mkdir -p "$CL18"; exec >> "$M/logs/cl18_queue.log" 2>&1
log() { echo "[$(date -Is)] $*"; }
image_ok() { local id; id=$(docker image inspect alpasim-base:0.134.0 --format '{{.Id}}' 2>&1); [ "$id" = "$IMAGE" ] || { log "FATAL: simulator image is '$id', not $IMAGE -- refusing"; exit 3; }; }
gate() { while :; do L=$(awk '{print int($2)}' /proc/loadavg); R=$(free -g | awk '/^Mem:/{print $7}'); [ -e "$CL18/STOP" ] && { log "STOP file"; exit 0; }
  [ "$L" -lt 32 ] && [ "$R" -gt 40 ] && { log "gate open: load5 $L, RAM $R GB, gpu free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) MiB"; return; }
  log "gate closed: load5 $L, RAM $R GB"; sleep 120; done; }
log "=== QUEUE START (affinity $(taskset -pc $$ | awk -F': ' '{print $2}'), nice $(ps -o ni= -p $$ | tr -d ' ')) ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader >/dev/null || { log "FATAL: nvidia-smi"; exit 3; }
[ "$(sha256sum "$EXT20" | cut -c1-8)" = 1b778280 ] && [ "$(sha256sum "$P10" | cut -c1-8)" = 94f50aa3 ] || { log "FATAL: csv pins changed"; exit 3; }
[ "$(sha256sum "$S18F" | cut -d' ' -f1)" = 8a76391386cce356cbb185b156450d51f6ac26d243bc70073c0682d0b448e16c ] && [ "$(wc -l < "$S18F")" = 18 ] || { log "FATAL: scene list"; exit 3; }
grep -q "^Written 2026" "$M/reports/CL18_MEMORY_PREREG.md" || { log "FATAL: no pre-registration"; exit 3; }
log "pre-registration sha256 $(sha256sum "$M/reports/CL18_MEMORY_PREREG.md" | cut -d' ' -f1)"
[ "$(diff "$AS/adapters/drivejepamemory/launch/run_memory_scene.sh" "$M/src/s18_run_memory_scene.sh" | grep -c '^[<>]')" = 4 ] || { log "FATAL: carry runner differs from the frozen one beyond output root + model folder"; exit 3; }
[ "$(diff "$AS/drafts/drivejepa-0.1.8b-p06/run_018bp06_scene.sh" "$M/src/s18_run_018bp06_scene.sh" | grep -c '^[<>]')" = 2 ] || { log "FATAL: 0.1.8b runner differs beyond the output root"; exit 3; }
( cd "$AS/adapters" && sha256sum -c --quiet drivejepamemory/launch/tables/build_memory.sha256 ) || { log "FATAL: memory adapter build freeze"; exit 3; }
image_ok; log "image $IMAGE confirmed"
i=0
while read -r short scene lane; do
  csv="$EXT20"; [ "$lane" = P10 ] && csv="$P10"
  for arm in carry 018bp06; do
    i=$((i+1))
    if [ "$arm" = carry ]; then RD="$CL18/runs/memory_carry_$short"; else RD="$CL18/018bp06/runs/drivejepa_018bp06_$short"; fi
    if [ -e "$RD/rollouts" ]; then log "=== RUN $i: $arm $short already has rollouts -- skipped ==="; continue; fi
    gate; image_ok; log "=== RUN $i: $arm $short ==="
    if [ "$arm" = carry ]; then
      EXT20_SCENES_SHA="$(sha256sum "$csv" | cut -d' ' -f1)" "$M/src/s18_run_memory_scene.sh" dj "$short" "$scene" carry --scenes-csv "$csv" --gpu-floor-mib 8000 < /dev/null
    else
      DRIVEJEPA_MOMENTUM=on EXT20_SCENES_SHA="$(sha256sum "$csv" | cut -d' ' -f1)" "$M/src/s18_run_018bp06_scene.sh" dj "$short" "$scene" on --scenes-csv "$csv" --gpu-floor-mib 8000 < /dev/null
    fi
    rc=$?; log "=== RUN $i: $arm $short rc=$rc ==="
    if [ "$rc" != 0 ] && [ "$rc" != 9 ]; then log "STOP: run $i ($arm $short) ended rc=$rc -- queue halted, nothing else launched"; exit 5; fi
  done
done < "$S18F"
image_ok; log "post-queue: image unchanged; subnets used $(docker network ls --filter driver=bridge -q | wc -l)/31"
log "QUEUE DONE"
