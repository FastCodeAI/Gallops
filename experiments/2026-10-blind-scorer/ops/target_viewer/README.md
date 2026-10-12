# Scorer target viewer

Every tick of scorer caches v2 ({dj_cache}/v2, rendered) and v3_nusc (real nuScenes): front camera with all 64 plans projected (engine
FC_TRAJ projection), top-down ego-frame view (drivable map, lane centre, vehicles, target car at +1..+5 s,
probe points), and a sortable table of each plan's labels, six sub-score targets and held-out predictions
(5-seed probe+geometry average from the fold where that scene was held out).

Served at http://100.94.58.46:8932/ (Tailscale) and http://192.168.1.2:8932/ (office LAN). No login, read-only.
The URL hash remembers the episode (e.g. #round0_v5.frontal.0075.run_0). Each dropdown lists only what exists
under the ones to its left. The page is served no-store, so a reload always picks up edits to index.html.

Start:  cd ops/target_viewer && setsid nohup taskset -c 0-1 {paddock}/.venv/bin/python server.py --port 8932 > serve.log 2>&1 < /dev/null & echo $! > serve.pid
Stop:   kill $(cat ops/target_viewer/serve.pid)
