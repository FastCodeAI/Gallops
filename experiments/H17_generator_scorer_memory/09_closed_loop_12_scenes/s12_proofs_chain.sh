#!/bin/bash
# Closed-loop check, part 2: adapter proofs A, B, C in order, each behind the machine gate (5-minute load < 32, available RAM > 40 GB).
# Stops at the first proof that fails. Log: logs/cl12_proofs_chain.log
M="{working_folder}"; AS={runs}/paddock_sid/Alpasim
exec >> "$M/logs/cl12_proofs_chain.log" 2>&1
log() { echo "[$(date -Is)] $*"; }
gate() { while :; do L=$(awk '{print int($2)}' /proc/loadavg); R=$(free -g | awk '/^Mem:/{print $7}'); [ "$L" -lt 32 ] && [ "$R" -gt 40 ] && { log "gate open: load5 $L, RAM $R GB"; return; }; log "gate closed: load5 $L, RAM $R GB"; sleep 120; done; }
for P in ${@:-A B C}; do
  gate; log "=== PROOF $P start ==="
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 10 "$AS/alpasim/.venv/bin/python" -B "$M/src/s12_adapter_proofs.py" "$P" > "$M/logs/cl12_proof_$P.log" 2>&1; rc=$?
  log "=== PROOF $P rc=$rc: $(tail -1 "$M/logs/cl12_proof_$P.log") ==="
  grep -q "PROOF $P PASS" "$M/logs/cl12_proof_$P.log" || { log "STOP: proof $P did not pass"; break; }
done
log "CHAIN DONE"
