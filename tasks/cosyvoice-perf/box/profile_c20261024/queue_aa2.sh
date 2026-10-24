#!/usr/bin/env bash
# 10-24, cards 0 and 1, after queue_l65.sh: the A/A band of the cells p_aac did not cover, main against main (t_mn,
# t_mn2, cd191df23) at c1f, c64, b1f, b16, both orders, default warmup, with each second's host load. L6's pair reads
# c64 first audio p99 +3.8 ; +2.0 %, c1f first audio p95 +1.4 ; +1.4 %, b16 latency p99 +3.4 ; +1.5 % and b1f req/s
# -0.8 ; -0.3 %, worse in both orders, beside gains under 8 %; this band reads them.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261024/l65/queue_aa2.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_aa2.sh'
D=/data/c20261024
P=$D/l65
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261024
CELLS=c1f,c64,b1f,b16
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
until grep -q 'queue l65 done' $P/queue.txt; do sleep 30; done
log "queue aa2 start t_mn $(git -C /workspace/t_mn rev-parse HEAD:sglang_omni | cut -c1-8) t_mn2 $(git -C /workspace/t_mn2 rev-parse HEAD:sglang_omni | cut -c1-8)"
python3 $IN/host_load_c20261022.py $P/p_aa2_load.jsonl &
sampler=$!
(cd /workspace/sglang-omni && PAIR_CELLS=$CELLS python3 $IN/serve_pair_c20261021.py /workspace/t_mn \
  /workspace/t_mn2 $P/p_aa2 > $P/p_aa2.log 2>&1)
log "pair p_aa2 rc $? token cap hits $(grep -E '\| at the token cap \|' $P/p_aa2.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
kill $sampler
python3 $IN/host_load_c20261022.py --summary $P/p_aa2_load.jsonl $P/p_aa2 $CELLS > $P/p_aa2_load.txt 2>&1
python3 $IN/pooled_inter_chunk_c20261021.py $P/p_aa2 c64 t_mn t_mn2 > $P/p_aa2_pooled_c64.txt 2>&1
python3 $HERE/latency_tail_by_audio.py $P/p_aa2 b16 c64 > $P/p_aa2_tails.txt 2>&1
log "queue aa2 done"
