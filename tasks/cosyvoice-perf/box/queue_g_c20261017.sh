#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: item 1 with only item 2's ranking (t_sr_1441d0b60, no hold), since p_sr showed the hold
# raising c64 inter chunk and playback underrun. Its tree warms its Inductor cache on card 3, then the pair against
# upstream main (t_main2): seeded c1 (identity), unseeded c1, c16, c64, both orders. No scoring: the ranking moves
# when windows run, not what they compute.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_g.pgid; exec bash queue_g_c20261017.sh'
D=/data/c20261017
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_g.txt; }

log "queue start"
bash $IN/warm_c20261017.sh t_sr_1441d0b60 3 16-31,80-95 22000 $D/smoke_2b
log "smoke 2b $(grep 'bench rc' $D/smoke_2b/progress.txt) tracebacks $(grep -c Traceback $D/smoke_2b/serve.log)"
cd /workspace/t_main2
if grep -q "bench rc 0" $D/smoke_2b/progress.txt && ! grep -q Traceback $D/smoke_2b/serve.log; then
  PAIR_CELLS=c1,c1u,c16,c64 python3 $IN/serve_pair_c20261017.py /workspace/t_main2 /workspace/t_sr_1441d0b60 \
    $D/p_2b > $D/p_2b.log 2>&1
  log "p_2b rc $?"
  python3 $IN/c1_identity_c20261017.py $D/p_2b c1 > $D/p_2b_identity.txt 2>&1
  log "p_2b identity $(tr '\n' ' ' < $D/p_2b_identity.txt)"
else
  log "p_2b skipped"
fi
log "queue done"
