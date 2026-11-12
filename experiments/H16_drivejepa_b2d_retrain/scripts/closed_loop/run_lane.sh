#!/bin/bash
# Phase 2 Drive-JEPA (drivejepa_p2, epoch 18) on the 50 routes, 06 Nov; copied from run3a/b2d-run3a/run_lane.sh. Run 2 command-fix
# baseline lane script (run3a/b2d-run2/run_lane.sh), same worktree (rerank-native), native-50 settings (PADDOCK_INFER_HZ=2)
# and command fix (WAJEPA_DJ_FOLLOW=straight in the TOMLs); only the model, runs roots, ports and route timeout differ.
# Queue (queue.txt) = "<repeat> <route>": the 14 junction-turn routes (route turns > 45 deg within 30 m) for repeats a and b,
# then the other 36 of the 50 for a and b. Each repeat has its own runs root (lane<L>.rep<r>.toml).
# Before each route, under a lock: >= VRAM_MIN (12) GB free VRAM, >= 16 GB available RAM, CPU busy (us+sy+ni, 5 s sample)
# < CPU_MAX (80) %, no PAUSE, no other lane in start-up. Claims claims/<pfx>/<rep>_<route>.a<n>; a route whose run broke
# (route_state 'infra') is set aside with --move and taken again, max 3 attempts. Model outcomes stay.
#   run_lane.sh <lane>
L=$1; O={ops}/b2d_drivejepa/phase2/closed_loop; OPS={ops}; PFX=p2-b2d-ep18
W=$OPS/worktrees/rerank-native; V={workspace}/Paddock/.venv
LOGS=$O/logs; C=$OPS/claims/$PFX; QUEUE=$O/queue.txt; MAX_ATTEMPTS=3
mkdir -p $LOGS $C
runs_of() { echo "$(grep -m1 '^runs' $O/lane$L.rep$1.toml | cut -d'"' -f2)/bench2drive/drivejepa_p2"; }
free_vram() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print int(($1-$2)/1024)}'; }
free_ram() { free -g | awk 'NR==2{print $7}'; }
cpu_busy() { read -r _ a1 b1 c1 d1 e1 f1 g1 h1 _ < /proc/stat; sleep 5; read -r _ a2 b2 c2 d2 e2 f2 g2 h2 _ < /proc/stat
  echo $(( 100 * ((a2 - a1) + (b2 - b1) + (c2 - c1)) / ((a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2) - (a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1)) )); }
others_starting() { for f in $O/current_lane*; do [ -f "$f" ] || continue; n=${f##*current_lane}; [ "$n" = "$L" ] && continue
  [ -f $O/started_lane$n ] || { echo "lane $n"; return; }; done; }
# "<rep> <route> <attempt>" for a newly claimed item, WAIT if the only ones left run elsewhere, nothing if all are done.
pick() {
  local busy=0 rep r st n running a
  declare -A state
  for rep in a; do
    while read r st; do state[$rep$r]=$st; done < <(python3 $O/route_state.py $(runs_of $rep) $(awk -v p=$rep '$1==p{print $2}' $QUEUE | paste -sd,))
  done
  while read rep r; do
    [ "${state[$rep$r]}" = done ] && continue
    n=0; running=0
    for a in $C/${rep}_$r.a*; do [ -d "$a" ] || continue; n=$((n + 1)); [ -f "$a/finished" ] || running=1; done
    [ $running = 1 ] && { busy=1; continue; }
    [ $n -ge $MAX_ATTEMPTS ] && continue
    mkdir $C/${rep}_$r.a$((n + 1)) 2>/dev/null && { echo "$rep $r $((n + 1))"; return; }
    busy=1
  done < $QUEUE
  [ $busy = 1 ] && echo WAIT
}
exec 9>$O/.start.lock
while true; do
  flock 9
  until [ ! -f $O/PAUSE ] && [ "$(free_ram)" -ge 16 ] && [ "$(free_vram)" -ge ${VRAM_MIN:-12} ] && [ "$(cpu_busy)" -lt ${CPU_MAX:-80} ] \
      && [ -z "$(others_starting)" ]; do
    echo "lane $L waiting: ram $(free_ram) GB vram $(free_vram) GB cpu $(cpu_busy) % starting: $(others_starting) $(date -u +%T)"; sleep 60; done
  p=$(pick)
  if [ -z "$p" ] || [ "$p" = WAIT ]; then
    flock -u 9
    [ -z "$p" ] && break
    echo "lane $L: nothing to pick, routes still running elsewhere $(date -u +%T)"; sleep 300; continue
  fi
  set -- $p; rep=$1; r=$2; a=$3
  TOML=$O/lane$L.rep$rep.toml; ROOT=$(grep -m1 '^runs' $TOML | cut -d'"' -f2); RUNS=$ROOT/bench2drive/drivejepa_p2; CALLS=$ROOT/calls
  echo "$rep $r" > $O/current_lane$L; rm -f $O/started_lane$L
  flock -u 9
  echo "== $rep $r lane $L start $(date -u +%T) (attempt $a)"
  : > $LOGS/lane$L.log  # the watcher below must not see the previous route's "ready"
  # the lane counts as started 2 min after its node answered /alive (CARLA has loaded the map by then), or after 10 min
  ( for i in $(seq 1 60); do grep -q "drivejepa_p2 ready" $LOGS/lane$L.log 2>/dev/null && { sleep 120; break; }; sleep 10; done
    touch $O/started_lane$L ) 9>&- &
  PYTHONPATH=$W/src PADDOCK_INFER_HZ=2 PADDOCK_INFER_TIMEOUT_S=300 PADDOCK_CONFIG=$TOML PADDOCK_BASE_PORT=$((17600 + L * 100)) \
    nice -n 5 $V/bin/paddock run --model drivejepa_p2 --bench bench2drive --scene $r > $LOGS/lane$L.log 2>&1 9>&-
  rc=$?
  cp $LOGS/lane$L.log $LOGS/lane${L}_${rep}_$r.a$a.log 2>/dev/null
  res=$(python3 -c "
import json,glob,os
fs=sorted(glob.glob('$RUNS/*/$r-*/results.json'), key=os.path.getmtime)
r=json.load(open(fs[-1]))['_checkpoint']['records'] if fs else []
print(f\"{r[0]['status']}, driving_score {r[0]['scores']['score_composed']:.2f}, route_completion {r[0]['scores']['score_route']:.2f}\" if r and 'scores' in r[0] else 'no episodes completed')" 2>&1 | tail -n 1)
  st=$(python3 $O/route_state.py $RUNS $r --move | awk '{print $2}')
  mkdir -p $CALLS/$r.a$a && mv $CALLS/lane$L-rep$rep/calls_*.jsonl $CALLS/$r.a$a/ 2>/dev/null
  touch $C/${rep}_$r.a$a/finished; rm -f $O/current_lane$L
  echo "== $rep $r lane $L attempt $a exit $rc $(date -u +%T): $st -- $res"
done
echo "LANE_ALL_DONE $L $(date -u +%T)"
