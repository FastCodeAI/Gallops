#!/bin/bash
# Run 3B v2 AlpaSim gate (20 Nov; Navvrat: "what about alphasim .. why arent we using that"): quick 1-scene check, then the
# 10 scenes, from worktree gate-oct27-alp (origin/main 9819e36 + our adapters; run3b_v2 takes run 3A's wajepa AlpaSim driver
# profile), site config gate-oct27/paddock.toml, against the queue's drivejepa AlpaSim baseline. Run under `sg docker`.
set -u
C={runs}/gate-oct27
G={runs}/gate-run3b
cd {runs}/worktrees/gate-oct27-alp
export PADDOCK_CONFIG=$C/paddock.toml PYTHONPATH={runs}/worktrees/gate-oct27-alp/src PADDOCK_INFER_TIMEOUT_S=600
exec nice -n 10 {data.paddock}/.venv/bin/python -u -m tools.gate smoke \
  --model run3b_v2 --baseline $C/baselines_queue_drivejepa.toml --tier all --bench alpasim --execute \
  --cores 28-31 --base-port 10800 --renderer-port 8960 --label run3b_v2-alpasim --json $G/verdict_run3b_v2_alpasim.json
