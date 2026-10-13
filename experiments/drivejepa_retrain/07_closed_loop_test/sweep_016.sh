#!/usr/bin/env bash
# S5 TRIAL: all 30 census scenes on drive-jepa 0.1.6, sequential, one run each, DRIVEJEPA_MOMENTUM=on. Pattern of baseline_014b/sweep_014b*.sh.
# Same scene csvs and pins as the 0.1.4b campaign; GPU floor 8000 MiB as its runs 11-30 (Sid's instruction of 2026-10-02).
# Resumable: a scene whose run dir already holds rollouts is skipped (the runner itself refuses to merge).
set -uo pipefail
T=experiments/drivejepa_retrain/07_closed_loop_test
SID={runs}/paddock_sid; RUNS=$SID/Alpasim/runs/drivejepa
EXT20="$SID/scene_sets/sim_scenes_ext20_26p01.csv"; P10="$SID/Alpasim/drafts/drivejepa-0.1.4/smoke_01d503d4/sim_scenes_pilot10_26p01.csv"
log() { echo "[$(date -Is)] $*"; }
grep -q "READY" "$T/logs/check_only.log" || { log "FATAL: --check-only not READY"; exit 3; }
grep -q '"ok": true' "$T/tables/step1_proof.json" || { log "FATAL: Step 1 proof not passed"; exit 3; }
[ -s "$T/tables/build_016.sha256" ] || { log "FATAL: build not frozen"; exit 3; }
[ "$(sha256sum "$EXT20" | cut -c1-8)" = 1b778280 ] && [ "$(sha256sum "$P10" | cut -c1-8)" = 94f50aa3 ] || { log "FATAL: csv pins changed"; exit 3; }
[ "$(sha256sum {runs}/drivejepa_retrain/06_first_training_run/candidate/drive_jepa_s4_pass09.ckpt | cut -d' ' -f1)" = 8a71c3c0d84a9758de46c7b46eaf18556a62ac4837feea7271040e7a0caacdd2 ] || { log "FATAL: candidate checkpoint sha256 mismatch"; exit 3; }
log "candidate checkpoint sha256 verified"
i=0
while read -r short scene lane; do
  i=$((i+1)); csv="$EXT20"; [ "$lane" = P10 ] && csv="$P10"
  if [ -e "$RUNS/drivejepa_016_${short}/rollouts" ]; then log "=== RUN $i: $short already has rollouts -- skipped ==="; continue; fi
  [ -e "$T/STOP" ] && { log "STOP file present -- standing down before run $i"; break; }
  log "=== RUN $i: $short mode=on ==="
  DRIVEJEPA_MOMENTUM=on "$T/run_016_scene.sh" dj "$short" "$scene" on --scenes-csv "$csv" --gpu-floor-mib 8000 < /dev/null
  rc=$?; log "=== RUN $i rc=$rc ==="
  # launch 1 (2026-10-07 23:32): the base image was missing, its rebuild failed on a full root disk, and the 29 scenes behind it were
  # all refused in one second. A run that is neither green (0) nor red-gated (9) now stops the queue instead of burning through it.
  if [ "$rc" != 0 ] && [ "$rc" != 9 ]; then log "STOP: run $i ($short) ended rc=$rc -- queue halted, nothing else launched"; break; fi
done < "$T/tables/scenes_to_run.txt"
log "post-sweep: containers of ours: $(docker ps -a --format '{{.Names}}' | grep -c 'drivejepa_016_' || true); networks of ours: $(docker network ls --format '{{.Name}}' | grep -c 'drivejepa_016_' || true); subnets used $(docker network ls --filter driver=bridge -q | wc -l)/31"
log "SWEEP DONE"
