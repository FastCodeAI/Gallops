#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# S5 TRIAL, GENERATOR ONLY (07_closed_loop_test/runs_017): 30 census scenes on drive-jepa 0.1.7 = 0.1.6 + S10 generator weights + batch fix on (copy of 07_closed_loop_test/run_016_scene.sh; diff in runs_017/tables/runner_diff.txt; run dirs on the NAS, root-disk floor 15 -> 4 GiB because the run writes to the NAS) -- S5 TRIAL (Retrain-data/07_closed_loop_test): 30 census scenes on drive-jepa 0.1.6 = 0.1.4b + S4 pass-9 weights + bonnet mask + zeroed acceleration (copy of baseline_014b/run_014b_scene.sh; diff in 07_closed_loop_test/tables/runner_diff.txt) -- DRIVEJEPA_014B Stage B: 30-scene baseline on drive-jepa 0.1.4b (copy of s5/run_s5_scene.sh; diff in baseline_014b/tables/runner_diff.txt) -- S5: 30-scene campaign (copy of s5c/run_s5c_scene.sh; paths only) -- S5c: one scene on drive-jepa 0.1.5 + veto (copy of s5a/run_s5a_scene.sh; paths + veto provenance gate) (reranker on|off), shared 20 Hz referee.
# COPY of drafts/ext20/run_ext20_scene.sh; the diff is recorded in s5a/tables/runner_diff.txt.
# COPY of drafts/ext20/run_ext20_scene.sh; the diff is recorded in s5a/tables/runner_diff.txt.
#
#   run_s5a_scene.sh dj <short8> <full-clipgt-id> <on|off> [--scenes-csv PATH] [--gpu-floor-mib N] [--wait-cap-s S]
#
#   dj : drive-jepa 0.1.4 -- launch block VERBATIM from
#        drafts/drivejepa-0.1.4/pilot10/run_one_scene.sh (driver 9147, gates 1-6)
#   cv : constant-velocity 0.2.3 -- driver + config + gate1a VERBATIM from
#        drafts/cv_023_20hz_pilot10/run_cv_023_20hz_pilot10.sh (driver 9155),
#        scene selection switched from suite/first-10 to csv+scene_ids (the dj
#        template's own mechanism) because a suite cannot name 20 arbitrary scenes
#
# What this copy adds over the two templates, and only this (DRIVEJEPA_EXT20 C1):
#   * preflights INSIDE the run script: root disk >= 15 GiB, GPU free logged and
#     >= floor on two consecutive 60 s checks, free docker subnets logged and >= 1,
#     no other alpasim_wizard of ours running, port 9400 never in our block
#   * config-diff gate BEFORE launch: wizard dry-run into <bucket>/_preflight_dryrun/<run>,
#     resolved config vs the lane's analysis-2.0 template, scene keys only
#     (config_diff_gate.py); re-verified on the real resolved config after launch
#   * provenance gate before anything is spent (dj: package realpath +
#     ADAPTER_VERSION; cv: driver sha + version string, as the floor's gate1a)
#   * teardown: container logs captured, compose down, driver killed, ONLY networks
#     named after this run removed (deploy=local is host-networked: none expected)
#   * annotated replay for every scene incl. red ones (tools/make_replays.py), into
#     the run's own HDD dir, symlinked into analysis_2.0/drivejepa_014_ext20/
#   * gates never stop the campaign: verdicts go to $LOGDIR/ext20_gates.json and the
#     exit code tells the sweep (0 green, 9 red gates, 20 wizard failed, else preflight)
set -uo pipefail

LANE="${1:?dj}"; SHORT="${2:?short8}"; SCENE="${3:?clipgt id}"; MODE="${4:?on|off}"; shift 4
GPU_FLOOR_MIB=""; WAIT_CAP_S=10800; PREFLIGHT_ONLY=0; TEST_CSV=""; S5A_CSV=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-floor-mib) GPU_FLOOR_MIB="$2"; shift 2 ;;
    --wait-cap-s) WAIT_CAP_S="$2"; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;          # stop after the config-diff gate; launches nothing
    --test-scenes-csv) TEST_CSV="$2"; shift 2 ;;          # ONLY with --preflight-only: exercise the script on a cached scene
    --scenes-csv) S5A_CSV="$2"; shift 2 ;;                # S5a: the pinned csv that names this scene (ext20 or pilot10)
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
[ -z "$TEST_CSV" ] || [ "$PREFLIGHT_ONLY" = 1 ] || { echo "--test-scenes-csv needs --preflight-only"; exit 2; }

BASE={runs}/paddock_sid/Alpasim
SID={runs}/paddock_sid
HERE="$BASE/drafts/ext20"
VENV="$BASE/alpasim/.venv/bin/python"
BULK={runs}/alpasim_bulk
S5A={runs}/drivejepa_retrain/07_closed_loop_test_017/runs   # 0.1.7: run dirs on the NAS (Sid: outputs not on root), <S5A>/drivejepa_017_<scene>/
TRIAL=experiments/drivejepa_retrain/07_closed_loop_test/runs_017
export TMPDIR={runs}/drivejepa_retrain/07_closed_loop_test_017/tmp MPLCONFIGDIR={runs}/drivejepa_retrain/07_closed_loop_test_017/tmp/mpl
SCENES_CSV="${TEST_CSV:-${S5A_CSV:-$SID/scene_sets/sim_scenes_ext20_26p01.csv}}"
SCENES_CSV_SHA="${EXT20_SCENES_SHA:-}"          # the sweep pins it at campaign start
LANE_DIR="$TRIAL/lane"
NRE_PIN=26.1.112-59673f88
NRE_BAD=26.4.96-91b06fb8
POOL=31
EX_CONFIG=2; EX_PORT=3; EX_SUBNET=4; EX_DRIVER=5; EX_PROV=6; EX_BUSY=7; EX_DISK=8; EX_RED=9
log() { echo "[$(date -Is)] $*"; }

case "$LANE" in
  dj)
    DRAFT="$TRIAL/draft"                     # S5a: the 0.1.5 launcher/draft ...
    SEARCH_DRAFT="$BASE/drafts/drivejepa-0.1.4"               # ... under the UNCHANGED 0.1.4 wizard config (driver=drivejepa014, port 9147)
    LOGDIR="$S5A/drivejepa_017_${SHORT}"
    DRIVER_PORT=9147; DRIVER_CFG=drivejepa014
    [ -n "$GPU_FLOOR_MIB" ] || GPU_FLOOR_MIB=14394   # heaviest measured pilot10 floor, sweep_ten.sh:56
    NAME="drivejepa_017_${SHORT}"; VIDEO_TAG="drivejepa_017"; LINK_PREFIX="run_" ;;
  cv)
    DRAFT="$BASE/drafts/cv_023_20hz_pilot10"
    DRIVER_SRC="$BASE/drafts/cv_0.2.2"
    LOGDIR="$BASE/runs/baselines/cv_023_20hz_ext20/cv_023_20hz_ext20_$SHORT"
    DRIVER_PORT=9155; DRIVER_CFG=cv_023_20hz_pilot10; SEARCH_DRAFT="$DRAFT"
    [ -n "$GPU_FLOOR_MIB" ] || GPU_FLOOR_MIB=25600   # the floor's own GATE_GPU_MIB
    EXPECT_SHA=fd8356a0654422772eca5643a3783274773b0c7fde15ff770e18844fc82177d8
    NAME="cv_023_20hz_ext20_${SHORT}"; VIDEO_TAG="cv_023_20hz"; LINK_PREFIX="floor_run_" ;;
  *) echo "lane must be dj or cv"; exit "$EX_CONFIG" ;;
esac
CONSOLE="$LOGDIR/wizard_console.log"
GATES_JSON="$LOGDIR/ext20_gates.json"
PROJECT="$(basename "$LOGDIR")"   # compose project = basename(log_dir), as the templates; unique + greppable

# --- GATE 0: output path is the bucket, never runs/ top level; never merge -----
case "$LOGDIR" in "$S5A"/drivejepa_017_*) : ;;
  *) log "FATAL: $LOGDIR is not a bucket path"; exit "$EX_CONFIG" ;; esac
[ -e "$LOGDIR/rollouts" ] && { log "FATAL: $LOGDIR already holds rollouts -- refusing to merge"; exit "$EX_CONFIG"; }
if [ "$PREFLIGHT_ONLY" = 1 ]; then LOGDIR="$LOGDIR.preflight_only_$(date +%Y%m%dT%H%M%S)"; fi
mkdir -p "$LOGDIR" "$LANE_DIR/annotated_videos"
exec > >(tee -a "$LOGDIR/run_ext20_scene.log") 2>&1
log "=== START lane=$LANE scene=$SCENE logdir=$LOGDIR ==="

# --- GATE 0b: scene set frozen and pinned ---------------------------------------
[ -f "$SCENES_CSV" ] || { log "FATAL: $SCENES_CSV missing (Part B not frozen)"; exit "$EX_CONFIG"; }
got=$(sha256sum "$SCENES_CSV" | cut -d' ' -f1)
if [ -n "$SCENES_CSV_SHA" ]; then
  [ "$got" = "$SCENES_CSV_SHA" ] || { log "FATAL: scenes csv sha $got != campaign pin $SCENES_CSV_SHA"; exit "$EX_CONFIG"; }
fi
log "scenes csv sha256 $got"
bad=$(tail -n +2 "$SCENES_CSV" | grep -vc "$NRE_PIN")
[ "$bad" = "0" ] || { log "FATAL: $bad csv rows are not the pinned NRE build"; exit "$EX_CONFIG"; }
grep -q ",$SCENE," "$SCENES_CSV" || { log "FATAL: $SCENE not in the frozen csv"; exit "$EX_CONFIG"; }

source "$BASE/alpasim_env.sh"
UUID=$(grep ",$SCENE," "$SCENES_CSV" | cut -d, -f1)
if [ ! -f "$BULK/nre-artifacts/all-usdzs/$UUID.usdz" ] && [ -z "${HF_TOKEN:-}" ]; then
  log "FATAL: artifact $UUID not cached and HF_TOKEN unset (gated dataset) -- cannot fetch"; exit "$EX_CONFIG"
fi

# --- preflights (inside the run script; a wrapper-only guard is not a guard) ------
free_gib() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_subnets() { echo $(( POOL - $(docker network ls --filter driver=bridge -q | wc -l) )); }
ours_running() { pgrep -u "$(id -un)" -f "alpasim_wizard" | grep -v "^$$\$" | wc -l; }
# 0.1.7: the run dir is on the NAS. Root is nearly full with other users' files (9 GiB on 2026-10-10); the floor only keeps this run from starting on a root disk that cannot take docker's own logs.
F=$(free_gib); [ "$F" -ge 4 ] || { log "FATAL preflight: root free ${F} GiB < 4"; exit "$EX_DISK"; }
NF=$(df -BG --output=avail "$S5A" | tail -1 | tr -dc '0-9'); [ "$NF" -ge 50 ] || { log "FATAL preflight: NAS free ${NF} GiB < 50"; exit "$EX_DISK"; }; log "preflight: NAS free ${NF} GiB"
S=$(free_subnets); log "preflight: root free ${F} GiB; docker subnets ${S}/${POOL} free"
[ "$S" -ge 1 ] || { log "FATAL preflight: free subnets read ${S} -- STOP (Sid's rule)"; exit "$EX_SUBNET"; }
# 9400 is never ours: wizard block 9100-9111, drivers 9147/9155, prometheus 9101.
for p in 9100 9111 "$DRIVER_PORT"; do [ "$p" != 9400 ] || { log "FATAL: port 9400 in our block"; exit "$EX_PORT"; }; done
! ss -ltn 2>/dev/null | grep -q ":${DRIVER_PORT}\b" || { log "FATAL: ${DRIVER_PORT} already in use"; exit "$EX_PORT"; }
WAIT_T0=$(date +%s); STREAK=0
while :; do
  G=$(gpu_free); G=${G:-0}; O=$(ours_running)
  if [ "$O" = 0 ] && [ "$G" -ge "$GPU_FLOOR_MIB" ]; then
    STREAK=$((STREAK+1)); log "preflight ${STREAK}/2: gpu free ${G} MiB >= floor ${GPU_FLOOR_MIB}, no other wizard of ours"
    [ "$STREAK" -ge 2 ] && break
  else
    STREAK=0; log "WAIT: gpu free ${G} MiB (floor ${GPU_FLOOR_MIB}), other wizards of ours: ${O}"
  fi
  [ $(( $(date +%s) - WAIT_T0 )) -lt "$WAIT_CAP_S" ] || { log "BLOCKED: preflight not met in ${WAIT_CAP_S}s -- standing down"; exit "$EX_BUSY"; }
  sleep 60
done
if [ "$LANE" = cv ]; then  # the floor's own extra bars, verbatim values
  RM=$(free -g | awk '/^Mem:/{print $7}')
  a=$(awk '/^pswpin|^pswpout/{s+=$2} END{print s+0}' /proc/vmstat); sleep 2
  b=$(awk '/^pswpin|^pswpout/{s+=$2} END{print s+0}' /proc/vmstat); IO=$(( (b-a)*4/2 ))
  log "cv preflight: ram avail ${RM}/45 GiB, swap io ${IO}/2000 KB/s"
  [ "$RM" -ge 45 ] || { log "FATAL preflight: RAM avail ${RM} GiB < 45"; exit "$EX_BUSY"; }
  [ "$IO" -le 2000 ] || { log "FATAL preflight: swap I/O ${IO} KB/s > 2000 (thrashing)"; exit "$EX_BUSY"; }
fi
log "PRELAUNCH containers=$(docker ps -aq | wc -l) gpu_free_mib=$(gpu_free) load1=$(awk '{print $1}' /proc/loadavg) subnets_free=$(free_subnets)"

# --- provenance gate, before anything is spent -----------------------------------
if [ "$LANE" = dj ]; then
  PLANNER_REAL=$(readlink -f "$DRAFT/drivejepa/planner.py")
  case "$PLANNER_REAL" in "$TRIAL"/adapters/drivejepa017/planner.py) : ;;
    *) log "FATAL prov: draft planner resolves to $PLANNER_REAL, not runs_017/adapters/drivejepa017/"; exit "$EX_PROV" ;; esac
  AV=$("$VENV" -B -c "import sys; sys.path.insert(0,'$TRIAL/adapters'); from drivejepa017 import constants as K; print(K.ADAPTER_VERSION, K.BONNET_MASK_ROWS, K.ZERO_ACCELERATION, K.ZERO_MOMENTUM_ACCELERATION, K.BATCH_FIX, K.CHECKPOINT_SHA256)")
  [ "$AV" = "0.1.7 37 True True True 29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916" ] || { log "FATAL prov: ADAPTER_VERSION=$AV"; exit "$EX_PROV"; }
  MSHA=$(sha256sum "$TRIAL/adapters/drivejepa017/momentum.py" | cut -d' ' -f1); PSHA=$(sha256sum "$PLANNER_REAL" | cut -d' ' -f1)
  ( cd "$TRIAL/adapters" && sha256sum -c --quiet "$TRIAL/tables/build_017.sha256" ) || { log "FATAL prov: runs_017/adapters/drivejepa017 or the launcher differs from the build frozen after the gate (tables/build_017.sha256)"; exit "$EX_PROV"; }
  [ "$(sha256sum "$TRIAL/../adapter_0.1.7_as_built.diff" | cut -d' ' -f1)" = "$(cat "$TRIAL/tables/adapter_diff.sha256" | cut -d' ' -f1)" ] || { log "FATAL prov: adapter_0.1.7_as_built.diff changed since the freeze"; exit "$EX_PROV"; }
  grep -q '"gate_1_under_2cm": true' "$TRIAL/tables/gate_017.json" || { log "FATAL prov: build gate (plans under 2 cm) not passed"; exit "$EX_PROV"; }
  log "prov: adapter diff sha256 $(cut -d' ' -f1 "$TRIAL/tables/adapter_diff.sha256"), checkpoint sha256 ${AV##* }, batch_fix.py sha256 $(sha256sum "$TRIAL/adapters/drivejepa017/batch_fix.py" | cut -d' ' -f1)"
  log "prov OK: planner $PLANNER_REAL, build $AV, momentum.py sha ${MSHA:0:12}, planner.py sha ${PSHA:0:12}, mode $MODE, launcher sha $(sha256sum "$DRAFT/serve_drivejepa_0_1_7.py" | cut -c1-12)"
else
  GOT_SHA=$(sha256sum "$DRIVER_SRC/constant_velocity_driver.py" | cut -d' ' -f1)
  [ "$GOT_SHA" = "$EXPECT_SHA" ] || { log "FATAL gate1: driver sha $GOT_SHA != $EXPECT_SHA"; exit "$EX_PROV"; }
  grep -q 'DRIVER_VERSION_ID = "constant_velocity-driver-0.2.3"' "$DRIVER_SRC/constant_velocity_driver.py" || { log "FATAL gate1: driver does not declare 0.2.3"; exit "$EX_PROV"; }
  grep -q 'HANDOVER_MIN_DERIVED_SPEED_M_S = 1.0' "$DRIVER_SRC/constant_velocity_driver.py" || { log "FATAL gate1: no speed-form handover guard"; exit "$EX_PROV"; }
  grep -q 'if derived <= HANDOVER_MIN_DERIVED_SPEED_M_S' "$DRIVER_SRC/constant_velocity_driver.py" || { log "FATAL gate1: guard not on derived SPEED"; exit "$EX_PROV"; }
  grep -q '^HANDOVER_MIN_POSE_GAP_M = 0.5' "$DRIVER_SRC/constant_velocity_driver.py" && { log "FATAL gate1: old 0.5 m guard live"; exit "$EX_PROV"; }
  log "gate1a OK: serving $DRIVER_SRC/constant_velocity_driver.py sha ${GOT_SHA:0:12} (0.2.3, speed-form guard)"
fi

# --- the wizard command, per lane, VERBATIM apart from scene args and log_dir -----
wizard_cmd() {  # $1 = log dir, $2 = extra hydra overrides (dry_run)
  uv run alpasim_wizard \
    "hydra.searchpath=[file://$SEARCH_DRAFT/alpasim_configs,file://$BASE/adapters/alpasim_configs]" \
    deploy=local \
    topology=1gpu \
    driver="$DRIVER_CFG" \
    wizard.baseport=9100 \
    wizard.log_dir="$1" \
    scenes.scene_cache="$BULK/nre-artifacts" \
    "scenes.scenes_csv=[$SCENES_CSV]" \
    "scenes.scene_ids=[$SCENE]" \
    scenes.test_suite_id=null \
    scenes.limit_to_first_n=0 \
    runtime.enable_autoresume=true \
    eval.allow_aggregation_with_failed_rollouts=true \
    'eval.scorers.min_ade.time_deltas=[0.5,1.0,2.5]' ${2:-}
}

# --- config-diff gate: dry run, then compare the RESOLVED config to the template --
cd "$BASE/alpasim"
# OUTSIDE the run dir: the aggregator discovers "job directories" by scanning the
# run dir's subdirectories for a wizard config (eval/aggregation/main.py:231), and a
# dry-run dir inside it became the ONLY job dir -> "No metrics files found"
# (ext20 scene 1, 2026-09-22 21:47). Kept beside the bucket, never in a run dir.
DRY="$(dirname "$LOGDIR")/_preflight_dryrun/$PROJECT"; rm -rf "$DRY"; mkdir -p "$DRY"
log "config-diff gate: wizard dry-run into $DRY (also fetches/pins the artifact)"
wizard_cmd "$DRY" "wizard.dry_run=true" > "$DRY/wizard_dryrun_console.log" 2>&1
drc=$?
[ "$drc" = 0 ] || { log "FATAL: dry-run rc=$drc"; tail -20 "$DRY/wizard_dryrun_console.log"; exit "$EX_CONFIG"; }
"$VENV" -B "$HERE/config_diff_gate.py" "$LANE" "$DRY/wizard-config-loadable.yaml" --rundir "$DRY" \
    --json "$LOGDIR/config_diff_prelaunch.json" || { log "FATAL: config-diff gate REFUSED (pre-launch)"; exit "$EX_CONFIG"; }
n04=$(grep -c "$NRE_BAD" "$DRY/wizard_dryrun_console.log"); n26=$(grep -c "$NRE_PIN" "$DRY/wizard_dryrun_console.log")
log "dry-run artifacts: 26.01 x${n26}, 26.04 x${n04}"
[ "$n04" = 0 ] || { log "FATAL: 26.04 artifact resolved for this scene in the dry run"; exit "$EX_CONFIG"; }
[ -f "$BULK/nre-artifacts/all-usdzs/$UUID.usdz" ] || { log "FATAL: artifact $UUID not in cache after dry run"; exit "$EX_CONFIG"; }
if [ "$PREFLIGHT_ONLY" = 1 ]; then log "=== PREFLIGHT ONLY: gates passed, nothing launched (dir $LOGDIR) ==="; exit 0; fi

# --- teardown -------------------------------------------------------------------
DRIVER_PID=""; STACK_RELEASED=0; WIZ_PID=""; GPU_SAMPLER=""
release_stack() {
  [ "$STACK_RELEASED" = 1 ] && return 0; STACK_RELEASED=1
  log "teardown: capturing container logs BEFORE compose down"
  mkdir -p "$LOGDIR/container-logs"
  if [ -f "$LOGDIR/docker-compose.yaml" ]; then
    docker compose -f "$LOGDIR/docker-compose.yaml" logs --no-color > "$LOGDIR/container-logs/all.log" 2>&1 || true
    docker compose -f "$LOGDIR/docker-compose.yaml" down --remove-orphans >/dev/null 2>&1 || true
  fi
  if [ -n "$DRIVER_PID" ] && kill -0 "$DRIVER_PID" 2>/dev/null; then
    log "teardown: stopping driver pid $DRIVER_PID"; kill "$DRIVER_PID" 2>/dev/null || true; sleep 2; kill -9 "$DRIVER_PID" 2>/dev/null || true
  fi
  for n in $(docker network ls --format '{{.Name}}' | grep -F "$PROJECT" || true); do
    docker network rm "$n" >/dev/null 2>&1 && log "teardown: removed OUR network $n"
  done
  log "teardown: networks left for this run: $(docker network ls --format '{{.Name}}' | grep -cF "$PROJECT" || true); subnets free $(free_subnets)"
}
teardown() { rc=$?; trap - EXIT INT TERM; release_stack; [ -n "$GPU_SAMPLER" ] && kill "$GPU_SAMPLER" 2>/dev/null; log "teardown complete (rc=$rc)"; }
kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done; kill -TERM "$1" 2>/dev/null || true; }
on_signal() { log "signal: stopping wizard tree${WIZ_PID:+ (pid $WIZ_PID)}"; [ -n "$WIZ_PID" ] && kill_tree "$WIZ_PID"; exit 143; }
trap teardown EXIT; trap on_signal INT TERM
( while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | head -1)"; sleep 10; done ) > "$LOGDIR/gpu_samples.log" 2>/dev/null &
GPU_SAMPLER=$!

# --- driver up, per lane VERBATIM ----------------------------------------------
BIND_T0=$(date +%s)
if [ "$LANE" = dj ]; then
  DRIVEJEPA_BATCH_FIX=on DRIVEJEPA_WORKER_LOG="$LOGDIR/driver_worker.log" DRIVEJEPA_MOMENTUM="$MODE" "$VENV" "$DRAFT/serve_drivejepa_0_1_7.py" --host 0.0.0.0 --port "$DRIVER_PORT" --device cpu \
      --control-timestep-us 50000 > "$LOGDIR/driver.log" 2>&1 &
  DRIVER_PID=$!
  driver_up() { ss -ltn 2>/dev/null | grep -q ":${DRIVER_PORT}\b" && return 0; grep -q "driver listening on" "$LOGDIR/driver.log" 2>/dev/null; }
  for _ in $(seq 1 3600); do driver_up && break; kill -0 "$DRIVER_PID" 2>/dev/null || { log "FATAL: driver died while loading"; tail -30 "$LOGDIR/driver.log"; exit "$EX_DRIVER"; }; sleep 1; done
  bound=0; for _ in $(seq 1 10); do driver_up && { bound=1; break; }; sleep 1; done
  [ "$bound" = 1 ] || { log "FATAL: driver never bound ${DRIVER_PORT}"; exit "$EX_DRIVER"; }
  log "BIND_TIME_S=$(( $(date +%s) - BIND_T0 ))"
  grep -m1 "provenance:" "$LOGDIR/driver.log" || { log "FATAL: no provenance line"; exit "$EX_PROV"; }
  grep -q "adapter_version=0.1.7 " "$LOGDIR/driver.log" || { log "FATAL: wrong adapter_version"; exit "$EX_PROV"; }
  grep -q "runs_017/adapters/drivejepa017/planner.py" "$LOGDIR/driver.log" || { log "FATAL: planner not from the live 0.1.7 package"; exit "$EX_PROV"; }
  grep -q "drivejepa-0.1.7 | S10 generator (final_choice +2) with the pass-9 scorer, batch fix on (fix1+fix2+fix3), bonnet mask 37 rows, acceleration zeroed in the model input and in the §3.5 re-score | paper §3.5 momentum-aware selection (upstream calibrate_score)" "$LOGDIR/driver.log" || { log "FATAL: 0.1.7 provenance line missing"; exit "$EX_PROV"; }
  grep -q "drivejepa-0.1.7 batch_fix: installed on .* state {'fix1': True, 'fix2': True, 'fix3': True}" "$LOGDIR/driver_worker.log" || { log "FATAL: the worker did not report the batch fix installed and on"; exit "$EX_PROV"; }
  grep -m1 "drivejepa-0.1.7 batch_fix:" "$LOGDIR/driver_worker.log"
  if [ "$MODE" = off ]; then grep -q "switch=OFF" "$LOGDIR/driver.log" || { log "FATAL: kill switch not reported OFF"; exit "$EX_PROV"; }
  else grep -q "switch=on" "$LOGDIR/driver.log" || { log "FATAL: switch not reported on"; exit "$EX_PROV"; }; fi
  grep -m1 "drivejepa-0.1.7 | S10" "$LOGDIR/driver.log"
  grep -q "replan_period_us=500000" "$LOGDIR/driver.log" || { log "FATAL: divider period is not 500 ms"; exit "$EX_PROV"; }
else
  "$VENV" "$DRIVER_SRC/constant_velocity_driver.py" --host 0.0.0.0 --port "$DRIVER_PORT" > "$LOGDIR/cv_driver.log" 2>&1 &
  DRIVER_PID=$!
  driver_ready() { local l; l=$(ss -ltnp 2>/dev/null) || return 1; l=$(grep ":${DRIVER_PORT}\b" <<<"$l") || return 1; [[ "$l" == *"pid=${DRIVER_PID},"* ]]; }
  for _ in $(seq 1 60); do driver_ready && break; kill -0 "$DRIVER_PID" 2>/dev/null || { log "FATAL: driver died"; tail -30 "$LOGDIR/cv_driver.log"; exit "$EX_DRIVER"; }; sleep 1; done
  driver_ready || { log "FATAL: driver never bound ${DRIVER_PORT} (or another process holds it)"; exit "$EX_DRIVER"; }
  grep -q 'listening on' "$LOGDIR/cv_driver.log" || { log "FATAL: no listening line"; exit "$EX_DRIVER"; }
  log "BIND_TIME_S=$(( $(date +%s) - BIND_T0 ))"
fi
log "driver up on ${DRIVER_PORT} (pid $DRIVER_PID)"

# --- the run (background + wait so a signal is not deferred, METEOR_STEP4) ------
log "=== RUN START scene=$SCENE ==="
( wizard_cmd "$LOGDIR" 2>&1 | tee -a "$CONSOLE"; exit "${PIPESTATUS[0]}" ) &
WIZ_PID=$!
# post-write verification of the REAL resolved config; kill the wizard on mismatch
for _ in $(seq 1 600); do [ -f "$LOGDIR/wizard-config-loadable.yaml" ] && break; kill -0 "$WIZ_PID" 2>/dev/null || break; sleep 1; done
if [ -f "$LOGDIR/wizard-config-loadable.yaml" ]; then
  if "$VENV" -B "$HERE/config_diff_gate.py" "$LANE" "$LOGDIR/wizard-config-loadable.yaml" --json "$LOGDIR/config_diff_launch.json" > "$LOGDIR/config_diff_launch.txt" 2>&1; then
    log "config-diff gate (launch): PASS -- $(tail -1 "$LOGDIR/config_diff_launch.txt")"
  else
    log "FATAL: config-diff gate REFUSED on the live resolved config -- killing the wizard"; cat "$LOGDIR/config_diff_launch.txt"
    kill_tree "$WIZ_PID"; wait "$WIZ_PID" 2>/dev/null; exit "$EX_CONFIG"
  fi
fi
wait "$WIZ_PID"; rc=$?; WIZ_PID=""
log "=== RUN END rc=$rc ==="
release_stack

# --- gates (recorded, never fatal for the campaign) -------------------------------
"$VENV" -B - "$LOGDIR" "$LANE" "$rc" <<'PY'
import json, sys
d, lane, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
json.dump({"lane": lane, "wizard_rc": rc, "gates": {}, "ok": None}, open(f"{d}/ext20_gates.json", "w"), indent=1)
PY
gate() {  # $1 name, $2 ok(0/1), $3 detail  -> merged into ext20_gates.json
  "$VENV" -B - "$GATES_JSON" "$1" "$2" "$3" <<'PY'
import json, sys
p, name, ok, detail = sys.argv[1:5]
j = json.load(open(p)); j["gates"][name] = {"ok": ok == "0", "detail": detail}; json.dump(j, open(p, "w"), indent=1)
PY
  log "GATE $1: $([ "$2" = 0 ] && echo PASS || echo FAIL) -- $3"
}
gate "wizard_rc" "$([ "$rc" = 0 ] && echo 0 || echo 1)" "wizard exit $rc"
n26=$(grep -o "$NRE_PIN" "$CONSOLE" 2>/dev/null | wc -l); n04=$(grep -o "$NRE_BAD" "$CONSOLE" 2>/dev/null | wc -l)
gate "artifact_pin" "$([ -s "$CONSOLE" ] && [ "$n26" -gt 0 ] && [ "$n04" -eq 0 ] && echo 0 || echo 1)" "26.01 x${n26}, 26.04 x${n04}"
if [ "$LANE" = dj ]; then
  gate "push_cadence" "$(grep -q "push cadence ok" "$LOGDIR/driver.log" && echo 0 || echo 1)" "push-cadence check ran"
  gate "context_guards_silent" "$(grep -qE "ContextSpacingError|PushCadenceError" "$LOGDIR/driver.log" && echo 1 || echo 0)" "no ContextSpacingError/PushCadenceError"
  if [ "$rc" = 0 ]; then
    # S5a fix after the runs: the checker lives in the 0.1.4 draft (S5a's 4 runs recorded FAIL = file-not-found; re-run by hand, s5a/tables/smoke_gates_rerun.txt)
    "$VENV" -B "$BASE/drafts/drivejepa-0.1.4/smoke_01d503d4/check_smoke_gates.py" "$LOGDIR" > "$LOGDIR/smoke_gates.txt" 2>&1; g=$?
    gate "smoke_gates_2_4_5" "$g" "$(grep -E "^(FAIL|ALL|GATES)" "$LOGDIR/smoke_gates.txt" | paste -sd';' -)"
    "$VENV" -B "$TRIAL/dj_extra_gates_017.py" "$LOGDIR" --json "$LOGDIR/dj_extra_gates.json" > "$LOGDIR/dj_extra_gates.txt" 2>&1; g=$?
    gate "dj_extra_gates" "$g" "$(grep -E "^(FAIL|EXTRA)" "$LOGDIR/dj_extra_gates.txt" | paste -sd';' -)"
  fi
else
  if [ "$rc" = 0 ]; then
    "$VENV" -B "$HERE/cv_gates.py" "$LOGDIR" --json "$LOGDIR/cv_gates.json" > "$LOGDIR/cv_gates.txt" 2>&1; g=$?
    gate "cv_gates" "$g" "$(grep -E "^(FAIL|CV GATES)" "$LOGDIR/cv_gates.txt" | paste -sd';' -)"
  fi
fi

# --- annotated replay for EVERY scene, red ones included --------------------------
REPLAYS="$LOGDIR/replay"; mkdir -p "$REPLAYS"
if [ -d "$LOGDIR/controller" ] && ls "$LOGDIR"/controller/*.csv >/dev/null 2>&1; then
  if "$VENV" "$BASE/tools/make_replays.py" --run-dir "$LOGDIR" --out-dir "$REPLAYS" --name "$NAME" > "$LOGDIR/make_replays.log" 2>&1 \
     || "$VENV" "$BASE/tools/make_replays.py" --run-dir "$LOGDIR" --out-dir "$REPLAYS" --name "$NAME" --no-gt >> "$LOGDIR/make_replays.log" 2>&1; then
    ln -sfn "$REPLAYS/${NAME}_annotated.mp4" "$LANE_DIR/annotated_videos/${SHORT}_${VIDEO_TAG}.mp4"
    gate "replay" 0 "$LANE_DIR/annotated_videos/${SHORT}_${VIDEO_TAG}.mp4 -> $REPLAYS/${NAME}_annotated.mp4"
  else
    gate "replay" 1 "make_replays failed, see $LOGDIR/make_replays.log"
  fi
else
  gate "replay" 1 "no controller csv -- nothing to render"
fi
ln -sfn "$LOGDIR" "$LANE_DIR/${LINK_PREFIX}${SHORT}"

ALL_OK=$("$VENV" -B - "$GATES_JSON" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); ok = all(g["ok"] for g in j["gates"].values()); j["ok"] = ok
json.dump(j, open(sys.argv[1], "w"), indent=1); print("0" if ok else "1")
PY
)
if [ "$rc" != 0 ]; then log "=== HARD FAILURE scene=$SHORT lane=$LANE wizard rc=$rc ==="; exit 20; fi
if [ "$ALL_OK" = 0 ]; then log "=== ALL GATES GREEN scene=$SHORT lane=$LANE ==="; exit 0; fi
log "=== RED GATES scene=$SHORT lane=$LANE (recorded in $GATES_JSON) ==="; exit "$EX_RED"
