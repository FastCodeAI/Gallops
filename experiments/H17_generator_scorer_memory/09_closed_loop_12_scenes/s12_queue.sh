#!/bin/bash
# Closed-loop check, part 4: the run queue. One detached process (tmux), sequential, each step behind the machine gate.
#   1. simulator-image check: 0.1.7 re-run on 4 scenes through the lane's own image-check runner (read-only, IMAGECHECK_ROOT under cl12);
#      the first of these runs rebuilds the missing simulator image. Compared by s12_imagecheck_cmp.py.
#   2. the 12 diagnosis scenes, carry then reset for each scene (so an interrupted queue leaves complete pairs).
# Resumable: a scene whose run folder already holds rollouts is skipped. A run that produced no rollout stops the queue (no retry loop).
# STOP file: {experiment_store}/runs/cl12/STOP. Log: logs/cl12_queue.log
set -uo pipefail
M="{working_folder}"
G="{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim"
AS={paddock_sid}/Alpasim; SID={paddock_sid}
CL12={experiment_store}/runs/cl12; RUNNER="$AS/adapters/drivejepamemory/launch/run_memory_scene.sh"
T17="$AS/analysis_3.0/Retrain-data/s5_trial/runs_017"
EXT20="$SID/scene_sets/sim_scenes_ext20_26p01.csv"; P10="$AS/drafts/drivejepa-0.1.4/smoke_01d503d4/sim_scenes_pilot10_26p01.csv"
S12F="$G/s10_018b_closed_loop/tables/scenes_12.txt"
exec >> "$M/logs/cl12_queue.log" 2>&1
log() { echo "[$(date -Is)] $*"; }
gate() { while :; do L=$(awk '{print int($2)}' /proc/loadavg); R=$(free -g | awk '/^Mem:/{print $7}'); [ -e "$CL12/STOP" ] && { log "STOP file"; exit 0; }
  [ "$L" -lt 32 ] && [ "$R" -gt 40 ] && { log "gate open: load5 $L, RAM $R GB, gpu free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) MiB"; return; }
  log "gate closed: load5 $L, RAM $R GB"; sleep 120; done; }
log "=== QUEUE START ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader >/dev/null || { log "FATAL: nvidia-smi"; exit 3; }
[ "$(sha256sum "$EXT20" | cut -c1-8)" = 1b778280 ] && [ "$(sha256sum "$P10" | cut -c1-8)" = 94f50aa3 ] || { log "FATAL: csv pins changed"; exit 3; }
[ "$(wc -l < "$S12F")" = 12 ] || { log "FATAL: scene list"; exit 3; }
for P in A B C; do grep -q '"pass": true' "$AS/adapters/drivejepamemory/launch/tables/cl12_proof_$P.json" || { log "FATAL: adapter proof $P not passed"; exit 3; }; done
grep -q "^Written 2026" "$M/reports/CL12_MEMORY_PREREG.md" || { log "FATAL: no pre-registration"; exit 3; }
log "pre-registration sha256 $(sha256sum "$M/reports/CL12_MEMORY_PREREG.md" | cut -d' ' -f1)"

# ---- 1. image check (Sid's rule of 2026-10-27, as s11_018c/logs/imagecheck_017.sh) ----
grep -q "READY" "$T17/logs/check_only.log" && grep -q '"gate_1_under_2cm": true' "$T17/tables/gate_017.json" || { log "FATAL: 0.1.7 build checks"; exit 3; }
[ "$(sha256sum {dataset_store}/s10_retrain/ckpt/drive_jepa_s10_gen.ckpt | cut -d' ' -f1)" = 29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916 ] || { log "FATAL: 0.1.7 checkpoint sha256"; exit 3; }
log "image before: $(docker image inspect alpasim-base:0.134.0 --format '{{.Id}} {{.Created}}' 2>&1 | head -1)"
for short in 04749bb9 acd22430 d2715023 b44a4a44; do
  read -r _ scene lane < <(grep "^$short " "$G/s4_closed_loop/tables/scenes_30.txt"); csv="$EXT20"; [ "$lane" = P10 ] && csv="$P10"
  if [ -e "$CL12/imagecheck_017/runs/drivejepa_017_$short/rollouts" ]; then log "IMAGECHECK $short already has rollouts -- skipped"; continue; fi
  gate; log "=== IMAGECHECK $short ($lane) ==="
  IMAGECHECK_ROOT="$CL12/imagecheck_017" DRIVEJEPA_MOMENTUM=on "$G/s4_closed_loop/src/run_017_imagecheck.sh" dj "$short" "$scene" on --scenes-csv "$csv" --gpu-floor-mib 8000 < /dev/null
  rc=$?; log "=== IMAGECHECK $short rc=$rc ==="
  [ "$rc" = 0 ] || [ "$rc" = 9 ] || { log "STOP: image check $short rc=$rc -- queue halted"; exit 4; }
done
log "image after: $(docker image inspect alpasim-base:0.134.0 --format '{{.Id}} {{.Created}}' 2>&1 | head -1); renderer: $(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedAt}}' | grep -i nre | head -3 | paste -sd';')"
"$AS/alpasim/.venv/bin/python" -B "$M/src/s12_imagecheck_cmp.py" > "$M/logs/cl12_imagecheck_cmp.log" 2>&1; log "image check compare: $(tail -1 "$M/logs/cl12_imagecheck_cmp.log")"

# ---- 2. the 12 scenes, carry and reset ----
i=0
while read -r short scene lane; do
  csv="$EXT20"; [ "$lane" = P10 ] && csv="$P10"
  for arm in carry reset; do
    i=$((i+1))
    if [ -e "$CL12/runs/memory_${arm}_${short}/rollouts" ]; then log "=== RUN $i: $arm $short already has rollouts -- skipped ==="; continue; fi
    gate; log "=== RUN $i: $arm $short ==="
    EXT20_SCENES_SHA="$(sha256sum "$csv" | cut -d' ' -f1)" "$RUNNER" dj "$short" "$scene" "$arm" --scenes-csv "$csv" --gpu-floor-mib 8000 < /dev/null
    rc=$?; log "=== RUN $i: $arm $short rc=$rc ==="
    if [ "$rc" != 0 ] && [ "$rc" != 9 ]; then log "STOP: run $i ($arm $short) ended rc=$rc -- queue halted, nothing else launched"; exit 5; fi
  done
done < "$S12F"
log "post-queue: containers of ours: $(docker ps -a --format '{{.Names}}' | grep -c "memory_" || true); subnets used $(docker network ls --filter driver=bridge -q | wc -l)/31"
log "QUEUE DONE"
