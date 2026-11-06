#!/bin/bash
# Phase 2 Drive-JEPA gate, closed-loop tiers, 06 Nov (user: "run the regression gate in the meantime"): neuroNCAP (quick + closed), then AlpaSim, as gate-oct27's run_ncap.sh and alp_one.sh.
# Its own lane on cores 28-31. tools.gate waits for free GPU memory before each launch (smoke_need_gb), so M2 and the route lanes keep theirs.
set -u
G={ops}/b2d_drivejepa/phase2/gate
cd {ops}/worktrees/gate-oct27
export PADDOCK_CONFIG=$G/paddock.toml PYTHONPATH={ops}/worktrees/gate-oct27/src PADDOCK_INFER_TIMEOUT_S=600
PY={workspace}/Paddock/.venv/bin/python
echo "=== neuroncap start $(date -u +%FT%TZ)"
nice -n 10 $PY -u -m tools.gate smoke --model drivejepa_p2 --baseline $G/baselines.toml --tier all --bench neuroncap --keep-going --execute --cores 28-31 --base-port 10900 --renderer-port 8960 --label drivejepa_p2-ncap --json $G/verdict_ncap.json > $G/logs/ncap.log 2>&1
echo "=== neuroncap exit $? $(date -u +%FT%TZ)"
echo "=== alpasim start $(date -u +%FT%TZ)"
nice -n 10 $PY -u -m tools.gate smoke --model drivejepa_p2 --baseline $G/baselines.toml --tier all --bench alpasim --keep-going --execute --cores 28-31 --base-port 11000 --label drivejepa_p2-alpasim --json $G/verdict_alpasim.json > $G/logs/alpasim.log 2>&1
echo "=== alpasim exit $? $(date -u +%FT%TZ)"
echo CLOSED_DONE
