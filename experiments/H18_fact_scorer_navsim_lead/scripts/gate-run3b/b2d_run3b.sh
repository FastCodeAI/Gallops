#!/bin/bash
# Run 3B v2 Bench2Drive gate (20 Nov; Navvrat: "did we add b2d regression gate or not .. in all run 2a and 3b"): the gate's
# bench2drive_quick (route 3717) then bench2drive (7 routes), from worktree gate-oct27 (paddock 3daf4b9 + ops adapters;
# run3b_v2 replans at 2 Hz, planning_hz 2.0), site config gate-oct27/paddock.toml, against the queue's released Drive-JEPA
# Bench2Drive run. RUN3B_SCORE_DOMAIN=lead: CARLA, scored as LEAD samples were in training (comfort is masked on all of
# CARLA in the spec). Starts after run 3B's HUGSIM lanes finish (one benchmark at a time; CPU at ~76 % at 12:28).
set -u
C={runs}/gate-oct27
G={runs}/gate-run3b
H={runs}/hugsim-eval/logs
until [ "$(grep -l LANE_DONE $H/run3b_v2_lane0.log $H/run3b_v2_lane1.log $H/run3b_v2_lane2.log 2>/dev/null | wc -l)" -ge 3 ]; do sleep 60; done
echo "HUGSIM lanes done $(date '+%F %T')"
cd {runs}/worktrees/gate-oct27
export PADDOCK_CONFIG=$C/paddock.toml PYTHONPATH={runs}/worktrees/gate-oct27/src PADDOCK_INFER_TIMEOUT_S=600
nice -n 10 {data.paddock}/.venv/bin/python -u -m tools.gate smoke \
  --model run3b_v2 --env RUN3B_SCORE_DOMAIN=lead --baseline $C/baselines_queue_drivejepa.toml --tier all \
  --bench bench2drive_quick --bench bench2drive --keep-going --execute \
  --cores 24-27 --base-port 10600 --renderer-port 8950 --label run3b_v2-b2d --json $G/verdict_run3b_v2_b2d.json > $G/logs/b2d_run3b_v2.log 2>&1
echo "b2d gate exit $? $(date '+%F %T')"
