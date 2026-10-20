#!/usr/bin/env bash
# 10-16 second box, after queue_f2: the compiled repeat mask on A2 (t_a2c): a serving smoke test on card 3, then, if it
# served cleanly, its pair against A2 (t_a2 against t_a2c, c1, c16, both orders); then the scheduler loop on the
# highest priority stream on the Snake head (t_prio, an experiment): smoke test, then t_snake against t_prio, c1, c16,
# c64.
# usage (in the container): setsid bash -c 'echo $$ > queue_h.pgid; exec bash queue_h_c20261016.sh'
D=/data/c20261016
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_h.txt; }

log "queue start"
bash $IN/warm_c20261016.sh t_a2c 3 16-31,80-95 28000 $D/smoke_a2c
log "smoke a2c $(tail -n 1 $D/smoke_a2c/progress.txt) bench $(grep 'bench rc' $D/smoke_a2c/progress.txt) tracebacks $(grep -c Traceback $D/smoke_a2c/serve.log)"
cd /workspace/sglang-omni
if grep -q "bench rc 0" $D/smoke_a2c/progress.txt && ! grep -q Traceback $D/smoke_a2c/serve.log; then
  PAIR_CELLS=c1,c16 python3 $IN/serve_pair_c20261016.py /workspace/t_a2 /workspace/t_a2c $D/p_a2c > $D/p_a2c.log 2>&1
  log "p_a2c rc $?"
else
  log "p_a2c skipped"
fi
bash $IN/warm_c20261016.sh t_prio 3 16-31,80-95 29000 $D/smoke_prio
log "smoke prio $(grep 'bench rc' $D/smoke_prio/progress.txt) tracebacks $(grep -c Traceback $D/smoke_prio/serve.log)"
if grep -q "bench rc 0" $D/smoke_prio/progress.txt && ! grep -q Traceback $D/smoke_prio/serve.log; then
  PAIR_CELLS=c1,c16,c64 python3 $IN/serve_pair_c20261016.py /workspace/t_snake /workspace/t_prio $D/p_prio \
    > $D/p_prio.log 2>&1
  log "p_prio rc $?"
else
  log "p_prio skipped"
fi
log "queue done"
