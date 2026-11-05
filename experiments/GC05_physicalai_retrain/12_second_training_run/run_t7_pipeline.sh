#!/bin/bash
# S10 task 7, steps 3-5 in order, unattended: (wait for the two readouts) -> final_choice.json (+ sha256, NAS copy) -> merged checkpoint + checks -> three dumps launched.
# UNSEEN is touched only by the last step, which runs only if final_choice.json and the merged-checkpoint record exist. Log: NAS/12_second_training_run/dump_gen/pipeline.log
R=$1; N={runs}/drivejepa_retrain/12_second_training_run; G=$N/gen; LOG=$N/dump_gen/pipeline.log; PY={runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python; cd "$(dirname "$0")"; S10=$(pwd)
say() { echo "$(date -Is) $*" >> $LOG; }
until grep -q "ALL-DONE" $G/t7_readout/wait.log 2>/dev/null; do pgrep -f run_t7_wait_readouts.sh > /dev/null || { sleep 60; grep -q "ALL-DONE" $G/t7_readout/wait.log 2>/dev/null || { say "FAILED waiter gone without ALL-DONE"; exit 1; }; }; sleep 60; done
say "readouts done: $(tr '\n' ';' < $G/t7_readout/wait.log | cut -c1-600)"
[ -f $S10/final_choice.json ] && { say "FAILED final_choice.json already exists"; exit 1; }
CUDA_VISIBLE_DEVICES="" $PY s10_final_choice.py $R > $N/dump_gen/final_choice_stdout.log 2>&1 || { say "FAILED choice script"; exit 1; }
cp $R/final_choice.json $S10/final_choice.json && (cd $S10 && sha256sum final_choice.json > final_choice.sha256) && cp $S10/final_choice.json $S10/final_choice.sha256 $N/ || { say "FAILED copying choice"; exit 1; }
CK=$($PY -c "import json;print(json.load(open('$S10/final_choice.json'))['chosen_checkpoint'] or '')"); say "choice written: $(cat $S10/final_choice.sha256) chosen checkpoint: $CK"
[ -n "$CK" ] && [ -f "$CK" ] || { say "FAILED no qualifying checkpoint"; exit 1; }
M=$N/ckpt/drive_jepa_physicalai_generator.ckpt
NOGATE=1 ./run_s10_dump_gen.sh s10_build_gen.py $CK $M $S10/gen_ckpt.json --choice $S10/final_choice.json > $N/dump_gen/build_gen_stdout.log 2>&1; say "build exit $?"
$PY -c "import json,sys;d=json.load(open('$S10/gen_ckpt.json'))['same_plans_check'];sys.exit(0 if d['plan_max_abs_diff_merged_vs_pass_file']==0 and d['logit_max_abs_diff_merged_vs_pass_file']==0 and d['all_finite'] else 1)" || { say "FAILED merged checkpoint record missing or plans differ"; exit 1; }
cp $S10/gen_ckpt.json $N/ckpt/; say "merged checkpoint ok: $(sha256sum $M)"
read l1 l5 l15 _ < /proc/loadavg; say "launching dumps; load $l1 $l5 $l15"
./launch_t7_dumps.sh $M 4 2 1 > $N/dump_gen/launch_stdout.log 2>&1; say "launched: $(tr '\n' ';' < $N/dump_gen/launch_stdout.log)"; say "PIPELINE-END"
