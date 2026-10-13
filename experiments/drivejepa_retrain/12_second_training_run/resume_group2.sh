#!/bin/bash
# S10 task 5: P3 and P4 were paused (SIGSTOP to their process groups 1114734 and 1114740) at launch + about 14 min because the 5-min load
# passed 30 with five pilots. This waits for P0, P1, P2 (python PIDs 1114725 1114729 1114735) to end, then lets P3 and P4 continue.
while kill -0 1114725 2>/dev/null || kill -0 1114729 2>/dev/null || kill -0 1114735 2>/dev/null; do sleep 20; done
echo "$(date -Is) group 1 ended; resuming process groups 1114734 1114740"; kill -CONT -1114734; kill -CONT -1114740; echo "$(date -Is) resumed"
