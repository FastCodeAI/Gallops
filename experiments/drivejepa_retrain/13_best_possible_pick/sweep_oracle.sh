#!/usr/bin/env bash
# STEP 0 ORACLE PICK: the two stuck scenes on drivejepa017_oracle, sequential, one run each, momentum off (pattern of 07_closed_loop_test/runs_017/sweep_017.sh;
# same scene csv, pins and GPU floor 8000 MiB as the 0.1.7 trial). A scene whose run dir already holds rollouts is skipped (the runner refuses to merge).
# After the runs: readout (tables/facts.txt) and copies of the videos and step logs under NAS 13_best_possible_pick/{videos,steplog}.
# Start detached:  setsid nohup ./sweep_oracle.sh > logs/sweep.log 2>&1 < /dev/null &
set -uo pipefail
T=experiments/drivejepa_retrain/13_best_possible_pick
SID={runs}/paddock_sid; NAS={runs}/drivejepa_retrain/13_best_possible_pick; RUNS=$NAS/runs
VENV=$SID/Alpasim/alpasim/.venv/bin/python
EXT20="$SID/scene_sets/sim_scenes_ext20_26p01.csv"
log() { echo "[$(date -Is)] $*"; }
( cd "$T" && head -1 prereg.sha256 | sha256sum -c --quiet ) || { log "FATAL: PREREG.md does not match its hash"; exit 3; }
grep -q '"ok": true' "$T/tables/gate_offline.json" || { log "FATAL: offline gate not passed"; exit 3; }
[ -s "$T/tables/build_oracle.sha256" ] || { log "FATAL: oracle build not frozen"; exit 3; }
[ "$(sha256sum "$EXT20" | cut -c1-8)" = 1b778280 ] || { log "FATAL: csv pin changed"; exit 3; }
[ "$(sha256sum {runs}/drivejepa_retrain/12_second_training_run/ckpt/drive_jepa_physicalai_generator.ckpt | cut -d' ' -f1)" = 29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916 ] || { log "FATAL: S10 generator checkpoint sha256 mismatch"; exit 3; }
log "PREREG.md sha256 $(head -1 "$T/prereg.sha256" | cut -c1-16); checkpoint 29d5f951... verified; images: $(docker image inspect alpasim-base:0.134.0 nvcr.io/nvidia/nre/nre-ga:26.04 --format '{{index .RepoTags 0}}={{.Id}}' | paste -sd' ')"
i=0
while read -r short scene; do
  i=$((i+1))
  if [ -e "$RUNS/drivejepa_017oracle_${short}/rollouts" ]; then log "=== RUN $i: $short already has rollouts -- skipped ==="; continue; fi
  [ -e "$T/STOP" ] && { log "STOP file present -- standing down before run $i"; break; }
  log "=== RUN $i: $short mode=off (oracle) ==="
  DRIVEJEPA_MOMENTUM=off "$T/run_oracle_scene.sh" dj "$short" "$scene" off --scenes-csv "$EXT20" --gpu-floor-mib 8000 < /dev/null
  rc=$?; log "=== RUN $i rc=$rc ==="
  if [ "$rc" != 0 ] && [ "$rc" != 9 ]; then log "STOP: run $i ($short) ended rc=$rc -- queue halted, nothing else launched"; break; fi
  mkdir -p "$NAS/videos" "$NAS/steplog"
  cp -f "$RUNS/drivejepa_017oracle_${short}/oracle_steps.jsonl" "$NAS/steplog/${short}_oracle_steps.jsonl" 2>/dev/null
  cp -f "$RUNS/drivejepa_017oracle_${short}/replay/drivejepa_017oracle_${short}_annotated.mp4" "$NAS/videos/${short}_oracle_annotated.mp4" 2>/dev/null
  cp -f "$RUNS/drivejepa_017oracle_${short}"/rollouts/clipgt-*/*/*_camera_front_wide_120fov_default.mp4 "$NAS/videos/${short}_oracle_front_camera.mp4" 2>/dev/null
  ( cd "$T" && TMPDIR=$NAS/tmp "$VENV" -B oracle_readout.py > "logs/readout_after_${short}.log" 2>&1 ); log "readout after $short: rc=$? ($(grep -c ' | ' "$T/tables/facts.txt" 2>/dev/null) facts)"
done <<'SCENES'
d2715023 clipgt-d2715023-a903-4e67-9b80-e6c76dbbbd99
b44a4a44 clipgt-b44a4a44-0d00-4795-8a66-7acb0fdf46a2
SCENES
log "post-sweep: containers of ours: $(docker ps -a --format '{{.Names}}' | grep -c 'drivejepa_017oracle_' || true); networks of ours: $(docker network ls --format '{{.Name}}' | grep -c 'drivejepa_017oracle_' || true)"
log "SWEEP DONE"
