#!/usr/bin/env bash
# P6 (perf/qwen3-tts-predictor-decode-attention cc09c00b2, stacked on P5 07b655c23: a Triton decode attention in place
# of SDPA in the fused predictor layer, releasing o_proj to launch under programmatic dependent launch) against P5: the
# predictor bench on both trees back to back (per kernel from a graph with PDL, and the step replay), the touched units
# (Qwen3-TTS and the Qwen3-Omni predictor suites, which share the module), an unseeded c1 pair (the attention changes
# bits, so c1 measures the per-frame latency the predictor bounds there), then c16 and c64 pairs in both card orders
# with each arm's SM Issue. Waits for queue_p5.sh.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
S=$T/tasks/omni_step_profiling_20260918/scripts
P5=/workspace/p5_07b655c23
P6=/workspace/p6_cc09c00b2
L=$D/queue.log
until grep -q queue_p5_done $L; do sleep 30; done
echo "queue p6 start $(date +%T) p5 $(git -C $P5 rev-parse HEAD:sglang_omni) p6 $(git -C $P6 rev-parse HEAD:sglang_omni)" >> $L
for TREE in $P5 $P6; do
  (cd $TREE && TRITON_CACHE_DIR=$D/triton_p6q_$(basename $TREE) PYTHONPATH=$TREE timeout 1800 python3 $S/predictor_layers_bench.py \
    > $D/predictor_bench_p6q_$(basename $TREE).txt 2>&1)
  echo "predictor bench $(basename $TREE) rc $? $(grep -h 'step replay' $D/predictor_bench_p6q_$(basename $TREE).txt | tr '\n' ' ') $(date +%T)" >> $L
done
(cd $P6 && PYTHONPATH=$P6 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts tests/unit_test/qwen3_omni/test_talker.py \
  tests/unit_test/qwen3_omni/test_predictor_kernels.py -q -p no:cacheprovider -rfE > $D/units/p6.log 2>&1)
echo "units p6 rc $? $(tail -n 1 $D/units/p6.log) $(date +%T)" >> $L
run_pair() {
  python3 $IN/pair.py --label $1 --tree $P5 --b-tree $P6 --concurrency $2 --a-card $3 --b-card $4 --port $5 ${6:-} \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $D/pairs/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $D/pairs/$1/b/serve.log)" >> $L
}
run_pair p6_c1 1 0 1 7000
run_pair p6_c16_a 16 0 1 13000 --gpu-metrics
run_pair p6_c16_b 16 1 0 18500 --gpu-metrics
run_pair p6_c64_a 64 0 1 22000 --gpu-metrics
run_pair p6_c64_b 64 1 0 30000 --gpu-metrics
python3 $IN/pair_table.py p6_c1 p6_c16_a p6_c16_b p6_c64_a p6_c64_b > $D/table_p6.txt 2>&1
echo "queue_p6_done $(date +%T)" >> $L
