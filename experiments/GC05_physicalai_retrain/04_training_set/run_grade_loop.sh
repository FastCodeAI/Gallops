#!/bin/bash
# Detached grading loop: repeats passes until every nurec scene with a map has a rows file (maps are still arriving during the first passes).
cd "$(dirname "$0")"; N=${1:-10}
while true; do
  ./run_s3_grade.sh $N >> grade.log 2>&1
  g=$(ls grade_rows | wc -l); m=$(ls {runs}/drivejepa_retrain/04_training_set/scenes/*/DONE 2>/dev/null | wc -l)
  echo "pass done: graded $g, maps $m, $(date +%H:%M)" >> grade.log
  if ! pgrep -f s3_fetch_maps.py > /dev/null && [ "$g" -ge "$m" ]; then break; fi
  sleep 120
done
echo "GRADE LOOP FINISHED $(date)" >> grade.log
