#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1: is L3's c16 gap cost (about +0.9 % pooled over six pairs, run to run spread
# near 1 %) real, and which part of L3 carries it? Four interleaved rounds, each four cached pairs against t_l2c (c16p
# primes the reference cache, then c16 is measured, both orders): t_l3c (L3 as is), t_l3f (V1, no TF32 toggle),
# t_l3e (V2, no graphs), t_l2c (A/A). Each pair's gaps pooled; the per arm distributions decide.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_k.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_k_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $D/queue_k.txt; }
pair() {
  local second=$1 name=$2 second_label=$3
  (cd /workspace/sglang-omni && PAIR_CELLS=c16p,c16 python3 $IN/serve_pair_c20261021.py /workspace/t_l2c /workspace/$second \
    $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
  python3 $IN/pooled_inter_chunk_c20261021.py $D/$name c16 base $second_label > $D/pooled_${name}_c16.txt 2>&1
  log "pooled $name c16 $(grep '^pooled' $D/pooled_${name}_c16.txt | tr '\n' ' ')"
}
log "queue k start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
for round in 1 2 3 4; do
  pair t_l3c p_k${round}_l3 l3
  pair t_l3f p_k${round}_v1 v1
  pair t_l3e p_k${round}_v2 v2
  pair t_l2c p_k${round}_aa aa
done
log "queue k done"
