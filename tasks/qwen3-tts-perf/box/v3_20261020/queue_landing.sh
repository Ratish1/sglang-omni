#!/usr/bin/env bash
# The PR branches alone on their landing base, upstream main 4809648d4 (bcd0d8c68 plus docs, CI and a benchmark client
# change; none touch the items' files): V4 (perf/qwen3-tts-codec-arena-flat-row 68746040d) and P6
# (perf/qwen3-tts-predictor-attention-tensor-cores 649b7639a, its final kernel in one commit), each against main. The
# touched units on the new base, the predictor bench on main and P6, V4's pairs (seeded c1 that must be byte identical,
# then c16 and c64 in both card orders), P6's pairs (unseeded c1, c16, c64), then main against main at c16 in both
# orders for the noise floor. V3 was measured against bcd0d8c68, its own base.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
S=$T/tasks/omni_step_profiling_20260918/scripts
M=/workspace/m_4809648d4
V4=/workspace/v4m_68746040d
P6=/workspace/p6m_649b7639a
L=$D/queue.log
echo "queue landing start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) v4 $(git -C $V4 rev-parse HEAD:sglang_omni) p6 $(git -C $P6 rev-parse HEAD:sglang_omni)" >> $L
(cd $V4 && PYTHONPATH=$V4 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider -rfE > $D/units/v4m.log 2>&1)
echo "units v4m rc $? $(grep -E 'passed|failed' $D/units/v4m.log | tail -n 1) $(date +%T)" >> $L
(cd $P6 && PYTHONPATH=$P6 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts tests/unit_test/qwen3_omni/test_talker.py \
  tests/unit_test/qwen3_omni/test_predictor_kernels.py -q -p no:cacheprovider -rfE > $D/units/p6m.log 2>&1)
echo "units p6m rc $? $(grep -E 'passed|failed' $D/units/p6m.log | tail -n 1) $(date +%T)" >> $L
for TREE in $M $P6; do
  (cd $TREE && TRITON_CACHE_DIR=$D/triton_landing_$(basename $TREE) PYTHONPATH=$TREE timeout 900 python3 $S/predictor_layers_bench.py \
    > $D/predictor_bench_landing_$(basename $TREE).txt 2>&1)
  echo "predictor bench $(basename $TREE) rc $? $(grep -h '^bs' $D/predictor_bench_landing_$(basename $TREE).txt | sed -E 's/pair pass [a-z ]+, step replay median //; s/ \(per kernel.*//' | tr '\n' ' ') $(date +%T)" >> $L
done
run_pair() {
  python3 $IN/pair.py --label $1 --tree $M --b-tree $2 --concurrency $3 --a-card $4 --b-card $5 --port $6 ${7:-} \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) $(grep -h 'byte identical' $D/$1.out) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $D/pairs/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $D/pairs/$1/b/serve.log)" >> $L
}
run_pair v4m_c1_seed $V4 1 0 1 7000 "--seed 1234"
run_pair v4m_c16_a $V4 16 0 1 13000 --gpu-metrics
run_pair v4m_c16_b $V4 16 1 0 18500 --gpu-metrics
run_pair v4m_c64_a $V4 64 0 1 22000 --gpu-metrics
run_pair v4m_c64_b $V4 64 1 0 30000 --gpu-metrics
run_pair p6m_c1 $P6 1 0 1 7000
run_pair p6m_c16_a $P6 16 0 1 13000 --gpu-metrics
run_pair p6m_c16_b $P6 16 1 0 18500 --gpu-metrics
run_pair p6m_c64_a $P6 64 0 1 22000 --gpu-metrics
run_pair p6m_c64_b $P6 64 1 0 30000 --gpu-metrics
run_pair aa_c16_a $M 16 0 1 13000 --gpu-metrics
run_pair aa_c16_b $M 16 1 0 18500 --gpu-metrics
python3 $IN/pair_table.py v4m_c1_seed v4m_c16_a v4m_c16_b v4m_c64_a v4m_c64_b p6m_c1 p6m_c16_a p6m_c16_b p6m_c64_a \
  p6m_c64_b aa_c16_a aa_c16_b > $D/table_landing.txt 2>&1
echo "queue_landing_done $(date +%T)" >> $L
