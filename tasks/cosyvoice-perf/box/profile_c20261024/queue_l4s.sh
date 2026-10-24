#!/usr/bin/env bash
# 10-24, cards 0 and 1, after queue_l65r: does #3129's review fix cost c64 throughput? p_l4r read req/s +6.8 ; +4.5 %
# against main where p_l4c read +6.9 ; +7.7 %, and the fixed pool is 56960 frames against 60096 (the step measured at
# rows of 4096 + 750 tokens, a conv context charged per block, the padding block and the allocator's rounding). At
# about 14 blocks a stream the pool holds about 63 full streams against 67, near the 64 clients.
# Pair p_l4s: t_l4q (edc167791, the reviewed head) against t_l4r (9a51c1f30, the fixed head), c64, both orders.
# Decision rule, written before the run: a row moves when both orders agree in sign and their mean leaves p_aa2's c64
# band (gap p95 +4.0 ; +2.4 %, latency p99 +5.5 ; +4.1 %); a move against t_l4r gets a trace of its whole-history
# hops before any change.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261024/l4r/queue_s.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_l4s.sh'
D=/data/c20261024
P=$D/l4r
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $P/queue_s.txt; }
log "pair p_l4s start $(git -C /workspace/t_l4q log --oneline -1 | cut -c1-20) against $(git -C /workspace/t_l4r log --oneline -1 | cut -c1-20)"
(cd /workspace/sglang-omni && PAIR_CELLS=c64 python3 $IN/serve_pair_c20261021.py /workspace/t_l4q \
  /workspace/t_l4r $P/p_l4s > $P/p_l4s.log 2>&1)
log "pair p_l4s rc $?"
python3 $IN/pooled_inter_chunk_c20261021.py $P/p_l4s c64 t_l4q t_l4r > $P/p_l4s_pooled_c64.txt 2>&1
grep -E '^\| (a|b) \| c64' $P/p_l4s.log >> $P/queue_s.txt
log "queue l4s done"
