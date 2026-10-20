#!/usr/bin/env bash
# 10-16 second box queue: the shared Inductor cache warmed by main on card 2 while card 3 runs A3's units, then the
# one step both kinds pair (c1, c16, c64) and A3's pair (c1, c16), each both orders on cards 2 and 3.
# usage (in the container): setsid bash -c 'echo $$ > run.pgid; exec bash queue_b_c20261016.sh'
D=/data/c20261016
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_b.txt; }
units() {
  local tree=$1 out=$2
  shift 2
  mkdir -p $out
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider "$@" \
    > $out/units.log 2>&1)
  log "units $tree rc $?"
}

log "queue start"
bash $IN/warm_c20261016.sh t_main 2 0-15,64-79 21000 $D/warm_main &
WARM=$!
units t_mtest $D/units_a3/main_new_test tests/unit_test/model_runner/test_install_sampling_seeds.py
units t_a3 $D/units_a3/a3 tests/unit_test/model_runner tests/unit_test/fun_cosyvoice3 \
  tests/unit_test/qwen3_tts/test_sampling_kernels.py
units t_main $D/units_a3/main tests/unit_test/model_runner tests/unit_test/fun_cosyvoice3 \
  tests/unit_test/qwen3_tts/test_sampling_kernels.py
wait $WARM
log "warm done"

cd /workspace/sglang-omni
PAIR_CELLS=c1,c16,c64 python3 $IN/serve_pair_c20261016.py /workspace/t_main /workspace/t_i4 $D/p_i4b > $D/p_i4b.log 2>&1
log "p_i4b rc $?"
PAIR_CELLS=c1,c16 python3 $IN/serve_pair_c20261016.py /workspace/t_main /workspace/t_a3 $D/p_a3 > $D/p_a3.log 2>&1
log "p_a3 rc $?"
log "queue done"
