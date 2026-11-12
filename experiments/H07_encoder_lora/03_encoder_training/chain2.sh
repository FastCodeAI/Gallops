#!/bin/bash
# Resume after the mask-equaliser crash (main step 1848): main from pass 3, then F.
cd "$(dirname "$0")"
set -e
./run.sh src/train.py main  2>&1 | tee -a logs/e_main.log;  test ${PIPESTATUS[0]} -eq 0
./run.sh src/f_final.py     2>&1 | tee -a logs/f_final.log; test ${PIPESTATUS[0]} -eq 0
./run.sh src/plots.py       2>&1 | tee -a logs/plots.log
echo CHAIN2 DONE
