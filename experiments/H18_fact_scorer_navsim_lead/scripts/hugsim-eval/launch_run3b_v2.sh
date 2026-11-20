#!/bin/bash
# Run 3B v2 on HUGSIM (20 Nov): run3b_v2 (final, step 23,859) from worktree rerank-native, same setup as the 22 Oct models (command only,
# 2 Hz, PR #57 controller, per-call records), rendering the four cameras it reads. Queue = the 420 runnable scenarios
# (queue_all minus the 16 HUGSIM release bugs). Usage: launch_run3a.sh smoke | launch_run3a.sh lanes "0 1 2"
D={runs}/hugsim-eval
P={data.paddock}/.venv/bin/python
RN={runs}/worktrees/rerank-native
R={data.p3_renesas}/hugsim-eval
CAMS=CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT,CAM_BACK
cd $D
# 20 Nov: lanes started by hand at Navvrat's ask before the chain reached them; the chain's later call stops here.
if [ "$1" = lanes ] && [ -e $R/run3b_v2/LANES_LAUNCHED ]; then echo "lanes already launched: $(cat $R/run3b_v2/LANES_LAUNCHED)"; exit 0; fi
if [ "$1" = smoke ]; then
  setsid nohup nice -n 10 $P run_eval.py --label smoke_run3b_v2 --model run3b_v2 --worktree $RN \
    --config $D/paddock-run3b-v2-lanesmoke.toml --port 9670 --infer_every 2 --lane 0 --queue $D/queue_smoke4.txt --limit 4 \
    --out $R/smoke_run3b_v2 --render_cams $CAMS --record_dir $R/smoke_run3b_v2/rec_lanesmoke > logs/smoke_run3b_v2.log 2>&1 < /dev/null &
  exit 0
fi
for L in $2; do
  setsid nohup nice -n 10 $P run_eval.py --label run3b_v2 --model run3b_v2 --worktree $RN \
    --config $D/paddock-run3b-v2-lane$L.toml --port $((9671 + L)) --infer_every 2 --lane $L --queue $D/queue_420.txt \
    --out $R/run3b_v2 --render_cams $CAMS --record_dir $R/run3b_v2/rec_lane$L > logs/run3b_v2_lane$L.log 2>&1 < /dev/null &
  sleep 90
done
