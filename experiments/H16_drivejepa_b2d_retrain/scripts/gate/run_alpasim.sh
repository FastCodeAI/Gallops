#!/bin/bash
# Phase 2 Drive-JEPA gate, AlpaSim (quick 1-scene check, then 10 scenes), after neuroNCAP ends. As gate-oct27/alp_one.sh: worktree gate-oct27-alp, under `sg docker`, lane cores 28-31.
set -u
G={ops}/b2d_drivejepa/phase2/gate
while kill -0 ${1:-0} 2>/dev/null; do sleep 60; done
echo "=== alpasim start $(date -u +%FT%TZ)"
sg docker -c "cd {ops}/worktrees/gate-oct27-alp && PADDOCK_CONFIG=$G/paddock.toml PYTHONPATH={ops}/worktrees/gate-oct27-alp/src PADDOCK_INFER_TIMEOUT_S=600 nice -n 10 {workspace}/Paddock/.venv/bin/python -u -m tools.gate smoke --model drivejepa_p2 --baseline $G/baselines.toml --tier all --bench alpasim --keep-going --execute --cores 28-31 --base-port 11000 --renderer-port 8970 --label drivejepa_p2-alpasim --json $G/verdict_alpasim.json > $G/logs/alpasim.log 2>&1"
echo "=== alpasim exit $? $(date -u +%FT%TZ)"
echo CLOSED_DONE
