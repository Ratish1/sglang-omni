#!/usr/bin/env bash
# 10-16 second box, after queue_c (the mixed step left the stack): V3 (C1 then D) on V1 on A3 (t_v3b) and on A3
# alone (t_v3a), with A1 (t_a1): their CosyVoice and model runner units on cards 2 and 3, then the pairs, c1, c16,
# c64, both orders: t_v1 against t_v3b (the stack's next item) and t_a3 against t_v3a (V3 without the step rule).
# usage (in the container): setsid bash -c 'echo $$ > queue_d.pgid; exec bash queue_d_c20261016.sh'
D=/data/c20261016
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_d.txt; }
units() {
  local tree=$1 card=$2 cpus=$3 out=$D/units_d/$1
  mkdir -p $out
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card taskset -c $cpus python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner > $out/units.log 2>&1)
  log "units $tree rc $?"
}

log "queue start"
units t_v3b 2 0-15,64-79 &
units t_v3a 3 16-31,80-95
wait
units t_a1 3 16-31,80-95

cd /workspace/sglang-omni
for pair in "t_v1 t_v3b p_v3b" "t_a3 t_v3a p_v3a"; do
  set -- $pair
  PAIR_CELLS=c1,c16,c64 python3 $IN/serve_pair_c20261016.py /workspace/$1 /workspace/$2 $D/$3 > $D/$3.log 2>&1
  log "$3 rc $?"
done
log "queue done"
