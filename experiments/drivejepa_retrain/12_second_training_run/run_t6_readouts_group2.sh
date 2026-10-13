#!/bin/bash
# S10 task 6: waits for P3 and P4 to end, then float32 + 16-bit SEEN held-out readouts of each (P4 with the wide range), one at a time.
G={runs}/drivejepa_retrain/12_second_training_run/gen; cd "$(dirname "$0")"
while kill -0 1114739 2>/dev/null || kill -0 1114742 2>/dev/null; do sleep 15; done
echo "$(date -Is) pilots ended" >> $G/t6_readout/group2.log
for trip in "P3:pilot_fix_l10:std" "P4:pilot_fixwide_l0:wide"; do
  IFS=: read arm run rng <<< "$trip"; mkdir -p $G/t6_readout/$arm
  if [ -f $G/$run/pass_01.pt ]; then
    ./run_s10_t6_readout.sh --mode readout --run t6_readout/$arm --init $G/$run/pass_01.pt --ref-range $rng --workers 2 --reserve-gb 0 >> $G/t6_readout/$arm/stdout.log 2>&1 < /dev/null
    echo "$(date -Is) $arm exit $?" >> $G/t6_readout/group2.log
  else echo "$(date -Is) $arm exit no-checkpoint" >> $G/t6_readout/group2.log; fi
done
