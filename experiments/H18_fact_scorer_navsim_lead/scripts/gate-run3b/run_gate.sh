#!/bin/bash
# Gate of 20 Nov (Navvrat: "can we run regression gate .... and also evaluate the model .. asap"): tools.gate smoke --tier all
# --keep-going for run3b_v2 (final checkpoint, step 23,859) against the queue's released Drive-JEPA baselines, from worktree
# gate-oct27 (origin/main 3daf4b9 + wajepa_dj / wajepa_dj_run3a / run3b_v2 adapters), site config gate-oct27/paddock.toml:
# the setup Run 3A's gate of 25-28 Oct ran under. Subsets: kitscenes, nuscenes, navsim, neuroncap_quick, neuroncap
# (bench2drive / alpasim not declared by run3b_v2's model.yaml).
set -u
C={runs}/gate-oct27
G={runs}/gate-run3b
cd {runs}/worktrees/gate-oct27
export PADDOCK_CONFIG=$C/paddock.toml PYTHONPATH={runs}/worktrees/gate-oct27/src PADDOCK_INFER_TIMEOUT_S=600
PY={data.paddock}/.venv/bin/python
m=run3b_v2
echo "=== $m start $(date -u +%FT%TZ)"
$PY -u -m tools.gate smoke --model $m --baseline $C/baselines_queue_drivejepa.toml --tier all --keep-going --execute \
  --cores 20-23 --cores 24-27 --cores 28-31 --base-port 10600 --renderer-port 8950 --json $G/verdict_$m.json > $G/logs/gate_$m.log 2>&1
echo "=== $m exit $? $(date -u +%FT%TZ)"
echo GATE_DONE
