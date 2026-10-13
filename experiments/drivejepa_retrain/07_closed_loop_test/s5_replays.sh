#!/usr/bin/env bash
# S5 TRIAL: collect the standard replays of every finished 0.1.6 run into 07_closed_loop_test/replays/.
# The "paddock-sid wrapper" is Alpasim/tools/make_replays.py: one call writes <name>_annotated.mp4 (H.264, HUD) AND <name>_replay.html
# (self-contained interactive top-down replay) + <name>_trace.json. The runner (run_016_scene.sh) already calls it into <run>/replay/;
# this script only (re)generates what is missing (--no-video when the mp4 is there) and copies the html / links the mp4. Resumable.
set -uo pipefail
BASE={runs}/paddock_sid/Alpasim; T=$BASE/analysis_3.0/Retrain-data/07_closed_loop_test; VENV=$BASE/alpasim/.venv/bin/python; OUT=$T/replays; mkdir -p "$OUT"
while read -r short scene lane; do
  run=$BASE/runs/drivejepa/drivejepa_016_$short; name=drivejepa_016_$short
  [ -f "$run/aggregate/metrics_results.parquet" ] || continue
  if [ ! -f "$run/replay/${name}_replay.html" ]; then
    extra=""; [ -f "$run/replay/${name}_annotated.mp4" ] && extra="--no-video"
    "$VENV" "$BASE/tools/make_replays.py" --run-dir "$run" --out-dir "$run/replay" --name "$name" $extra >> "$T/logs/replays.log" 2>&1 \
      || "$VENV" "$BASE/tools/make_replays.py" --run-dir "$run" --out-dir "$run/replay" --name "$name" --no-gt $extra >> "$T/logs/replays.log" 2>&1 \
      || { echo "$short: make_replays FAILED (see logs/replays.log)"; continue; }
  fi
  cp -u "$run/replay/${name}_replay.html" "$OUT/${short}_replay.html"
  [ -f "$run/replay/${name}_annotated.mp4" ] && ln -sfn "$run/replay/${name}_annotated.mp4" "$OUT/${short}_annotated.mp4"
done < "$T/tables/scenes_to_run.txt"
echo "replays: $(ls "$OUT"/*_replay.html 2>/dev/null | wc -l) html, $(ls "$OUT"/*_annotated.mp4 2>/dev/null | wc -l) mp4 in $OUT"
