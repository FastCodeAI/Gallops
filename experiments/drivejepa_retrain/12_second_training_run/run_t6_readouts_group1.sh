#!/bin/bash
# S10 task 6: float32 + 16-bit SEEN held-out readouts of P0, P1, P2, one after the other (one GPU job at a time, 2 loader workers).
G={runs}/drivejepa_retrain/12_second_training_run/gen; cd "$(dirname "$0")"
for pair in P0:pilot_fix_l0 P1:pilot_fix_l0p1 P2:pilot_fix_l1; do
  arm=${pair%%:*}; run=${pair##*:}; mkdir -p $G/t6_readout/$arm
  ./run_s10_t6_readout.sh --mode readout --run t6_readout/$arm --init $G/$run/pass_01.pt --workers 2 --reserve-gb 0 >> $G/t6_readout/$arm/stdout.log 2>&1 < /dev/null
  echo "$(date -Is) $arm exit $?" >> $G/t6_readout/group1.log
done
