#!/usr/bin/env bash
# 10-18 lease, cards 2 and 3: the buffered path's spread, upstream main against itself (t_main3 in both slots),
# buffered b1 (256) and b16, both orders (aa_b), to read p_hgb's b1 latency tails (+3.6 / +1.8 % p95, +3.3 / +2.0 %
# p99) and b16 p99 (+3.3 / -5.9 %) against it. Waits for queue_b (its scoring holds card 2).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_c.pgid; exec bash queue_c_c20261018.sh'
D=/data/c20261018
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_c.txt; }

log "queue wait"
until grep -q "queue done" $D/queue_b.txt; do sleep 30; done
log "queue start"
cd /workspace/t_main3
PAIR_CELLS=b1,b16 python3 $IN/serve_pair_c20261018.py /workspace/t_main3 /workspace/t_main3 $D/aa_b > $D/aa_b.log 2>&1
log "aa_b rc $?"
log "queue done"
