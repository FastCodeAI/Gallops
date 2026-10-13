#!/usr/bin/env bash
# S5 TRIAL, GENERATOR ONLY: all 30 census scenes on drive-jepa 0.1.7 (copy of 07_closed_loop_test/sweep_016.sh; names, the NAS run root and the start checks changed), sequential, one run each, DRIVEJEPA_MOMENTUM=on. Pattern of baseline_014b/sweep_014b*.sh.
# Same scene csvs and pins as the 0.1.4b campaign; GPU floor 8000 MiB as its runs 11-30 (Sid's instruction of 2026-10-02).
# Resumable: a scene whose run dir already holds rollouts is skipped (the runner itself refuses to merge).
set -uo pipefail
T=experiments/drivejepa_retrain/07_closed_loop_test/runs_017
SID={runs}/paddock_sid; RUNS={runs}/drivejepa_retrain/07_closed_loop_test_017/runs
EXT20="$SID/scene_sets/sim_scenes_ext20_26p01.csv"; P10="$SID/Alpasim/drafts/drivejepa-0.1.4/smoke_01d503d4/sim_scenes_pilot10_26p01.csv"
log() { echo "[$(date -Is)] $*"; }
grep -q "READY" "$T/logs/check_only.log" || { log "FATAL: --check-only not READY"; exit 3; }
grep -q '"gate_1_under_2cm": true' "$T/tables/gate_017.json" || { log "FATAL: build gate not passed"; exit 3; }
[ -s "$T/tables/build_017.sha256" ] && [ -s "$T/tables/adapter_diff.sha256" ] || { log "FATAL: build not frozen"; exit 3; }
[ "$(sha256sum "$EXT20" | cut -c1-8)" = 1b778280 ] && [ "$(sha256sum "$P10" | cut -c1-8)" = 94f50aa3 ] || { log "FATAL: csv pins changed"; exit 3; }
[ "$(sha256sum {runs}/drivejepa_retrain/12_second_training_run/ckpt/drive_jepa_physicalai_generator.ckpt | cut -d' ' -f1)" = 29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916 ] || { log "FATAL: S10 generator checkpoint sha256 mismatch"; exit 3; }
( cd {runs}/drivejepa_retrain/12_second_training_run && sha256sum -c --quiet final_choice.sha256 ) || { log "FATAL: final_choice.json does not match its sha256"; exit 3; }
log "S10 generator checkpoint sha256 29d5f951... verified; final_choice.json verified; adapter diff sha256 $(cut -d' ' -f1 "$T/tables/adapter_diff.sha256")"
i=0
while read -r short scene lane; do
  i=$((i+1)); csv="$EXT20"; [ "$lane" = P10 ] && csv="$P10"
  if [ -e "$RUNS/drivejepa_017_${short}/rollouts" ]; then log "=== RUN $i: $short already has rollouts -- skipped ==="; continue; fi
  [ -e "$T/STOP" ] && { log "STOP file present -- standing down before run $i"; break; }
  log "=== RUN $i: $short mode=on ==="
  DRIVEJEPA_MOMENTUM=on "$T/run_017_scene.sh" dj "$short" "$scene" on --scenes-csv "$csv" --gpu-floor-mib 8000 < /dev/null
  rc=$?; log "=== RUN $i rc=$rc ==="
  # launch 1 (2026-10-07 23:32): the base image was missing, its rebuild failed on a full root disk, and the 29 scenes behind it were
  # all refused in one second. A run that is neither green (0) nor red-gated (9) now stops the queue instead of burning through it.
  if [ "$rc" != 0 ] && [ "$rc" != 9 ]; then log "STOP: run $i ($short) ended rc=$rc -- queue halted, nothing else launched"; break; fi
done < "$T/tables/scenes_to_run.txt"
log "post-sweep: containers of ours: $(docker ps -a --format '{{.Names}}' | grep -c 'drivejepa_017_' || true); networks of ours: $(docker network ls --format '{{.Name}}' | grep -c 'drivejepa_017_' || true); subnets used $(docker network ls --filter driver=bridge -q | wc -l)/31"
log "SWEEP DONE"
