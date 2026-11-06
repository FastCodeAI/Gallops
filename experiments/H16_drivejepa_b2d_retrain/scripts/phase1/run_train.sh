#!/bin/bash
# Launcher, Sid's s9 run_train.sh with our paths: memory gate, the 30 s monitor alongside, the run, then DONE. ACTIVE marks a run in progress for watcher.sh.
# A Python exit of any kind writes DONE, so the watcher only restarts after the process vanished (reboot / kill).
#   run_train.sh <run name> [--resume <save dir>]
source {drivejepa_env}/env.sh
N=$1; shift
R={runs}/b2d_drivejepa/phase1/runs/$N; H={ops}/b2d_drivejepa/phase1; mkdir -p $R $H/logs
ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
echo "$(date -Is) train start $N $*, avail $ma GB" >> $H/logs/starts.txt
if [ -e $R/STOP ] || [ "$ma" -le 10 ]; then echo "$(date -Is) memory gate closed" >> $H/logs/starts.txt; exit 3; fi
rm -f $R/DONE
echo "$(date -Is) $*" > $R/ACTIVE
$H/monitor.sh $N &
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
cd $H/src && nice -n 10 $PY train_b2d.py --name $N "$@" >> $H/logs/$N.log 2>&1
rc=$?
echo "$(date -Is) rc $rc" > $R/DONE
rm -f $R/ACTIVE
echo "$(date -Is) train exit rc $rc" >> $H/logs/starts.txt
echo "P1_EXIT $rc $(date '+%F %T')" >> $H/logs/$N.log
