#!/bin/bash
# Phase 2 Drive-JEPA (drivejepa_p2, epoch 18) regression gate, 06 Nov: the first tier, as gate-oct27/run_openloop.sh. KITScenes, nuScenes, NAVSIM against the queue's released Drive-JEPA baselines.
set -u
G={ops}/b2d_drivejepa/phase2/gate
cd {ops}/worktrees/gate-oct27
export PADDOCK_CONFIG=$G/paddock.toml PYTHONPATH={ops}/worktrees/gate-oct27/src PADDOCK_INFER_TIMEOUT_S=600
PY={workspace}/Paddock/.venv/bin/python
echo "=== start $(date -u +%FT%TZ)"
nice -n 10 $PY -u -m tools.gate smoke --model drivejepa_p2 --baseline $G/baselines.toml --tier quick --bench kitscenes --bench nuscenes --bench navsim --execute --keep-going --cores 24-27 --base-port 10700 --label drivejepa_p2-openloop --json $G/verdict_openloop.json > $G/logs/openloop.log 2>&1
echo "=== exit $? $(date -u +%FT%TZ)"
echo OPENLOOP_DONE
