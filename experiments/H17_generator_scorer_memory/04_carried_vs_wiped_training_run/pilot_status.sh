#!/bin/bash
# one status line per arm: step, pass, last check's pick grade, ETA (from the last 100-step progress line's step time and the pass plan)
R={runs}/generator_scorer_memory/runs/pilot_seed0
for ARM in carry wiped; do
  L=$R/$ARM/log.txt; S=$R/$ARM/status.json
  if [ ! -f "$L" ]; then echo "$ARM: not started"; continue; fi
  if grep -q "ENDED" "$L"; then echo "$ARM: $(grep ENDED "$L" | tail -1 | cut -c12-200)"; continue; fi
  prog=$(grep -E " pass [0-9]+ step [0-9]+/[0-9]+ " "$L" | tail -1)
  pass=$(echo "$prog" | sed -E 's/.* pass ([0-9]+) step ([0-9]+)\/([0-9]+).*/\1/'); step=$(echo "$prog" | sed -E 's/.* pass ([0-9]+) step ([0-9]+)\/([0-9]+).*/\2/'); n=$(echo "$prog" | sed -E 's/.* pass ([0-9]+) step ([0-9]+)\/([0-9]+).*/\3/')
  sps=$(echo "$prog" | sed -E 's/.* step ([0-9.]+)s wait.*/\1/')
  grade=$(python3 -c "import json;d=json.load(open('$S'));print(f\"{d['pick_grade']:.4f} (check pass {d['pass']})\")" 2>/dev/null || echo "none yet")
  if [ -n "$step" ] && [ -n "$sps" ]; then
    left=$(python3 -c "n=$n; s=$step; p=$pass; sps=$sps; steps_left=(n-s)+(2-p)*n; checks_left=(n-s)//930+1+(2-p)*5; print(f'{(steps_left*sps+checks_left*600)/3600:.1f} h to the 2-pass end (less if the stop rule fires)')")
    echo "$ARM: step $step/$n of pass $pass, last check pick grade $grade, ETA $left"
  else
    echo "$ARM: starting (no progress line yet), last check pick grade $grade"
  fi
done
