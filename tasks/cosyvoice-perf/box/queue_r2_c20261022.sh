#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1: the controls for the bisection (queue b). p_c1 read L3c's cached gaps p95
# +4.1 ; +4.3 %, the variants V1 (no TF32 toggle) -2.1 ; -1.8 % and V2 (eager) -3.7 ; -0.6 %. V1 ends in the same
# matmul precision state as L3c (setAllowTF32CuBLAS(False) writes IEEE in both), so the toggle's end state cannot
# separate them; these pairs test whether p_c1's reading repeats: p_c2 t_l2c against t_l3c again, p_c3 t_l2c against
# t_l2c (a cached A/A); c16p then c16, both orders, gaps pooled.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_r2.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_r2_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $D/queue_r2.txt; }
pair() {
  local first=$1 second=$2 name=$3 first_label=$4 second_label=$5
  (cd /workspace/sglang-omni && PAIR_CELLS=c16p,c16 python3 $IN/serve_pair_c20261021.py /workspace/$first /workspace/$second \
    $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
  python3 $IN/pooled_inter_chunk_c20261021.py $D/$name c16 $first_label $second_label > $D/pooled_${name}_c16.txt 2>&1
  log "pooled $name c16 $(grep '^pooled' $D/pooled_${name}_c16.txt | tr '\n' ' ')"
}
log "queue r2 start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
pair t_l2c t_l3c p_c2 t_l2c t_l3c
pair t_l2c t_l2c p_c3 first second
log "queue r2 done"
