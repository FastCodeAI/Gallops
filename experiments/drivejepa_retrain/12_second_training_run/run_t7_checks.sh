#!/bin/bash
# S10 task 7: the three 32-example reproduction checks (pass 9, batch 8, fix off) of the new dump scripts against the stored S6 / S9 arrays, one at a time.
R=$1; L={runs}/drivejepa_retrain/12_second_training_run/dump_gen/checks; P9={runs}/drivejepa_retrain/06_first_training_run/run/ckpt/pass_09.pt; cd "$(dirname "$0")"
for s in seen unseen; do NOGATE=1 ./run_s10_dump_gen.sh s10_dump_gen.py --set $s --ckpt $P9 --bs 8 --check $R --check-n 32 --threads 4 --pool 2 > $L/check_$s.log 2>&1 < /dev/null; echo "$(date -Is) $s exit $?" >> $L/checks.log; done
NOGATE=1 ./run_s10_dump_gen.sh s10_dump_pullout.py --ckpt $P9 --bs 8 --check $R --check-n 32 --threads 4 --pool 2 > $L/check_pullout.log 2>&1 < /dev/null; echo "$(date -Is) pullout exit $?" >> $L/checks.log
echo "$(date -Is) ALL-DONE" >> $L/checks.log
