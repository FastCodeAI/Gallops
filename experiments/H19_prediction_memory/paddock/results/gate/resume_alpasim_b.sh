#!/bin/bash
# Complete (b)'s closed-tier AlpaSim run 20261106T040355-9f4cd1 in place (Sid, 6 Nov): `paddock run --resume`
# re-drives only scenes without a _complete marker (e121e37d, d2715023 failed to start). Same derived config and
# flags as the gate's launch. Guard: load < 32, available RAM > 40 GB, GPU >= 12 GB free (closed tier).
# Then re-runs the gate's closed (b) command, which finds the run finished and re-checks it.
set -u
W={paddock}; D={runs}
P={paddock_python}
B={weights}/drivejepa_memory/h15-r3-b-L0-s0/step_000001638
for i in $(seq 720); do
  load=$(awk '{print $2}' /proc/loadavg); avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "$(date '+%F %T %Z') guard load=$load avail=${avail}G gpu=${gpu}MiB"
  awk "BEGIN{exit !($load < 32 && $avail > 40 && $gpu >= 12288)}" && break
  [ $i = 720 ] && { echo "guard never opened"; exit 3; }
  sleep 60
done
cd $W
echo "$(date '+%F %T %Z') start resume alpasim (b)"
PYTHONUNBUFFERED=1 PYTHONPATH=$W/src PADDOCK_BASE_PORT=10500 PADDOCK_CONFIG=$D/gate/smoke/drivejepa-9fef30d0/paddock.toml \
  taskset -c 4-7 nice $P -m paddock.cli run --bench alpasim --model drivejepa --resume 20261106T040355-9f4cd1 \
  > $D/resume_alpasim_b.log 2>&1
echo "$(date '+%F %T %Z') end resume rc=$?"
echo "$(date '+%F %T %Z') start gate re-check closed b"
PYTHONUNBUFFERED=1 PADDOCK_CONFIG=$W/paddock.toml PYTHONPATH=$W/src $P -m tools.gate smoke --model drivejepa \
  --env DRIVEJEPA_MEMORY=$B --baseline {queue}/baselines/drivejepa.toml --tier closed \
  --execute --keep-going --base-port 10400 --renderer-port 8950 --cores 0-3 --cores 4-7 --cores 8-11 \
  --json $D/gate_closed_b_recheck.json > $D/gate_closed_b_recheck.log 2>&1
echo "$(date '+%F %T %Z') end gate re-check rc=$?"
echo DONE
