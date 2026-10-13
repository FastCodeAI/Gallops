#!/bin/bash
# S10 task 7: waits for final_P0 (python PID 3637372) to end, then float32 + 16-bit SEEN held-out readouts of its pass_01.pt (= pilot + 1) and
# pass_02.pt (= pilot + 2), one at a time, same settings as task 6 (run_t6_readouts_group2.sh). Appends to gen/t7_readout/wait.log.
G={runs}/drivejepa_retrain/12_second_training_run/gen; cd "$(dirname "$0")"; mkdir -p $G/t7_readout
PID=${1:-3637372}
while kill -0 $PID 2>/dev/null; do sleep 30; done
echo "$(date -Is) final run ended: $(tail -c 300 $G/final_P0/train.log | tail -1 | cut -c1-200)" >> $G/t7_readout/wait.log
for pair in "plus1:pass_01.pt" "plus2:pass_02.pt"; do
  IFS=: read arm ck <<< "$pair"; mkdir -p $G/t7_readout/$arm
  if [ -f $G/final_P0/$ck ]; then
    ./run_s10_t6_readout.sh --mode readout --run t7_readout/$arm --init $G/final_P0/$ck --workers 2 --reserve-gb 0 >> $G/t7_readout/$arm/stdout.log 2>&1 < /dev/null
    echo "$(date -Is) $arm exit $?" >> $G/t7_readout/wait.log
  else echo "$(date -Is) $arm exit no-checkpoint" >> $G/t7_readout/wait.log; fi
done
echo "$(date -Is) ALL-DONE" >> $G/t7_readout/wait.log
