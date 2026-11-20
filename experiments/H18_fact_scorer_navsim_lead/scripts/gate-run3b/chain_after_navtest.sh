#!/bin/bash
# 20 Nov: after run 3B's navtest exits (HDD), AlpaSim (10 scenes), then HUGSIM lanes 0 1 2 -- one benchmark at a time.
G={runs}/gate-run3b
L={runs}/run3b/v2/navtest/logs/navtest_final.log
until grep -q "^EXIT" $L; do sleep 30; done
echo "navtest done $(date '+%F %T'): $(grep '^EXIT' $L)"
sg docker -c "$G/alp_run3b.sh" > $G/logs/alpasim_run3b_v2.log 2>&1
echo "alpasim exit $? $(date '+%F %T')"
cd {runs}/hugsim-eval && ./launch_run3b_v2.sh lanes "0 1 2"
echo "HUGSIM lanes launched $(date '+%F %T')"
