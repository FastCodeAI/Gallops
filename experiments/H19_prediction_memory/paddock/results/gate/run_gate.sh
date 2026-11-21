#!/bin/bash
# H15 gate by hand (Sid D28-D30): quick tier (a) then (b), then closed tier (a) then (b); one gate run
# at a time, --keep-going so every subset of a tier is reported. Stops if a quick-tier subset does not
# finish (unfinished or INCOMPLETE). Guard before each: load < 32, GPU >= 12 GB free, available RAM
# > 9.8 GB for the quick tier (1.5 x the measured 6.52 GB peak; the quick tier runs in ONE lane so that
# peak applies) and > 40 GB for the closed tier (Sid, 6 Nov). Before the closed tier: released on AlpaSim
# gate scene 1acf2f5d with the present image (9ee8dfb8) against the baseline's 0.929 (Naren's rule); if it
# differs at 3 decimals the closed tier runs without AlpaSim. Re-runnable: the gate resumes.
set -u
W={paddock}; D={runs}
P={paddock_python}
A={weights}/drivejepa_memory/h15-r3-a-L4-s0/step_000001638
B={weights}/drivejepa_memory/h15-r3-b-L0-s0/step_000001638
guard() {  # $1 = RAM floor in GB
  for i in $(seq 720); do
    load=$(awk '{print $2}' /proc/loadavg); avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
    gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "$(date '+%F %T %Z') guard load=$load avail=${avail}G gpu=${gpu}MiB"
    if awk "BEGIN{exit !($load < 32 && $avail > $1 && $gpu >= 12288)}"; then return 0; fi
    sleep 60
  done
  echo "guard never opened"; exit 3
}
cd $W
gate() {  # tier arm dir ram cores... [--bench ...]
  local tier=$1 arm=$2 dir=$3 ram=$4; shift 4
  [ -f $D/gate_${tier}_${arm}.json ] && { echo "skip $tier $arm: done"; return 0; }
  guard $ram
  echo "$(date '+%F %T %Z') start $tier $arm $*"
  local env=(); [ -n "$dir" ] && env=(--env DRIVEJEPA_MEMORY=$dir)
  PYTHONUNBUFFERED=1 PADDOCK_CONFIG=$W/paddock.toml PYTHONPATH=$W/src $P -m tools.gate smoke --model drivejepa \
    "${env[@]}" --baseline {queue}/baselines/drivejepa.toml --tier $tier \
    --execute --keep-going --base-port 10400 --renderer-port 8950 "$@" \
    --json $D/gate_${tier}_${arm}.part.json > $D/gate_${tier}_${arm}.log 2>&1
  local rc=$?; echo "$(date '+%F %T %Z') end $tier $arm rc=$rc"
  if [ -f $D/gate_${tier}_${arm}.part.json ] && ! grep -q '"unfinished": \[\]' $D/gate_${tier}_${arm}.part.json; then
    echo "STOP: $tier $arm has unfinished subsets"; exit 4; fi
  if [ "$tier" = quick ] && grep -q INCOMPLETE $D/gate_${tier}_${arm}.log; then
    echo "STOP: $tier $arm has an INCOMPLETE subset"; exit 5; fi
  [ $rc -gt 1 ] && { echo "STOP: gate exit $rc"; exit 6; }
  mv $D/gate_${tier}_${arm}.part.json $D/gate_${tier}_${arm}.json
}
gate quick a $A 9.8 --cores 0-3
gate quick b $B 9.8 --cores 0-3
# Naren's image check: released Drive-JEPA on 1acf2f5d (the quick tier's AlpaSim scene), this image.
gate quick released-alpasim "" 40 --cores 0-3 --bench alpasim_quick --label h15-alpasim-image-check-released
cand=$(python3 -c "
import json
r = json.load(open('$D/gate_quick_released-alpasim.json'))
run = [s['run_id'] for s in r['steps'] if s['name'] == 'alpasim_quick'][0]
for line in open('$D/alpasim/drivejepa/' + run + '/samples.jsonl'):
    s = json.loads(line)
    if '1acf2f5d' in s['sample_id']:
        print(s['metrics']['scene_score'])
")
echo "image check: released on 1acf2f5d with alpasim-base 9ee8dfb8 scene_score $cand; baseline 0.9294890766651916 (reported 0.929)"
if [ -n "$cand" ] && [ "$(printf '%.3f' $cand)" = "0.929" ]; then
  echo "image check MATCH ($cand vs baseline 0.929): AlpaSim stays in the closed tier"; ALP=()
else
  echo "image check DIFFERS ($cand vs baseline 0.929): closed tier without AlpaSim"; ALP=(--bench neuroncap --bench bench2drive)
fi
gate closed a $A 40 --cores 0-3 --cores 4-7 --cores 8-11 "${ALP[@]}"
gate closed b $B 40 --cores 0-3 --cores 4-7 --cores 8-11 "${ALP[@]}"
echo DONE
