#!/bin/bash
# Stage 2 pipeline after the cache build: Gate A -> C.2 -> warm-up -> main -> F.
# Each step must exit 0; any STOP (exit 2) or error halts the chain.
cd "$(dirname "$0")"
until [ -f {runs}/drivejepa_kitscenes/encoder_lora_kit/s2/frame_cache/index.json ] && ! pgrep -f "a_cache.py build" >/dev/null; do sleep 60; done
set -e
./run.sh src/a_cache.py gate      2>&1 | tee -a logs/a_gate.log;      test ${PIPESTATUS[0]} -eq 0
./run.sh src/c_reference.py fixed 2>&1 | tee -a logs/c2_fixed.log;    test ${PIPESTATUS[0]} -eq 0
./run.sh src/train.py warmup      2>&1 | tee -a logs/d_warmup.log;    test ${PIPESTATUS[0]} -eq 0
./run.sh src/train.py main        2>&1 | tee -a logs/e_main.log;      test ${PIPESTATUS[0]} -eq 0
./run.sh src/f_final.py           2>&1 | tee -a logs/f_final.log;     test ${PIPESTATUS[0]} -eq 0
echo CHAIN DONE
