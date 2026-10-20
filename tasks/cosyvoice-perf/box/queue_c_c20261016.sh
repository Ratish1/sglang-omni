#!/usr/bin/env bash
# 10-16 second box, after queue_b: the CosyVoice and model runner units of the mixed step with each kind capped at
# the batch size (t_mix2, on A3) and V1 on it (t_v1m2), then the pairs, c1, c16, c64, both orders: t_a3 against
# t_mix2, t_mix2 against t_v1m2, and V1 alone (t_a3 against t_v1).
# usage (in the container): setsid bash -c 'echo $$ > queue_c.pgid; exec bash queue_c_c20261016.sh'
D=/data/c20261016
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_c.txt; }
units() {
  local tree=$1 card=$2 cpus=$3 out=$D/units_c/$1
  mkdir -p $out
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card taskset -c $cpus python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner > $out/units.log 2>&1)
  log "units $tree rc $?"
}

log "queue start"
units t_mix2 2 0-15,64-79 &
units t_v1m2 3 16-31,80-95
wait
units t_v1 3 16-31,80-95

cd /workspace/sglang-omni
for pair in "t_a3 t_mix2 p_mix2" "t_mix2 t_v1m2 p_v1m2" "t_a3 t_v1 p_v1"; do
  set -- $pair
  PAIR_CELLS=c1,c16,c64 python3 $IN/serve_pair_c20261016.py /workspace/$1 /workspace/$2 $D/$3 > $D/$3.log 2>&1
  log "$3 rc $?"
done
log "queue done"
