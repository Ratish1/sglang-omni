#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1, after queue x: is p_r3's c16 gap rise (L3 against L2, pooled mean +2.5 %,
# p95 +4.7 %) beyond this pair's spread? The profiled pair (queue x) read gaps mean +0.4 %, p95 +3.1 %, every request
# state equal or better, and no tokenizer device, lock or GIL cost above ONNX Runtime's. p_y1: t_l2 against t_l2
# (A/A, one tree for both arms), c16 both orders; p_y2: t_l2 against t_l3 again, c16 both orders; gaps pooled per arm.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_y.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_y_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_y.txt; }
pair() {
  local first=$1 second=$2 name=$3 cells=$4
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
}
pooled() {
  local name=$1 cell=$2 first=$3 second=$4
  python3 $IN/pooled_inter_chunk_c20261021.py $D/$name $cell $first $second > $D/pooled_${name}_$cell.txt 2>&1
  log "pooled $name $cell $(grep '^pooled' $D/pooled_${name}_$cell.txt | tr '\n' ' ')"
}
log "queue y start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
pair t_l2 t_l2 p_y1 c16
pooled p_y1 c16 first second
pair t_l2 t_l3 p_y2 c16
pooled p_y2 c16 t_l2 t_l3
log "queue y done"
