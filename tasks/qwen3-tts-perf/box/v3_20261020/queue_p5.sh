#!/usr/bin/env bash
# P5 (perf/qwen3-tts-predictor-weight-prefetch 07b655c23, stacked on V4 a6214172d: each fused predictor layer kernel
# prefetches its weight lines into L2 before waiting on its predecessor) against V4: the predictor bench on both trees
# (per kernel from a graph with PDL), the prefetch present in the compiled PTX, the touched units (Qwen3-TTS and the
# Qwen3-Omni predictor suites, which share the module), a seeded c1 pair that must be byte identical (no arithmetic
# changes), then c16 and c64 pairs in both card orders with each arm's SM Issue. Waits for queue_v4.sh.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
S=$T/tasks/omni_step_profiling_20260918/scripts
V4=/workspace/v4_a6214172d
P5=/workspace/p5_07b655c23
L=$D/queue.log
until grep -q queue_v4_done $L; do sleep 30; done
echo "queue p5 start $(date +%T) v4 $(git -C $V4 rev-parse HEAD:sglang_omni) p5 $(git -C $P5 rev-parse HEAD:sglang_omni)" >> $L
for TREE in $V4 $P5; do
  (cd $TREE && TRITON_CACHE_DIR=$D/triton_$(basename $TREE) PYTHONPATH=$TREE timeout 1800 python3 $S/predictor_layers_bench.py \
    > $D/predictor_bench_$(basename $TREE).txt 2>&1)
  echo "predictor bench $(basename $TREE) rc $? prefetching ptx files $(grep -l 'prefetch.global.L2' -r $D/triton_$(basename $TREE) --include='*.ptx' 2>/dev/null | wc -l) $(date +%T)" >> $L
done
(cd $P5 && PYTHONPATH=$P5 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts tests/unit_test/qwen3_omni/test_talker.py \
  tests/unit_test/qwen3_omni/test_predictor_kernels.py -q -p no:cacheprovider -rfE > $D/units/p5.log 2>&1)
echo "units p5 rc $? $(date +%T)" >> $L
run_pair() {
  python3 $IN/pair.py --label $1 --tree $V4 --b-tree $P5 --concurrency $2 --a-card $3 --b-card $4 --port $5 ${6:-} \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) $(grep -h 'byte identical' $D/$1.out) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $D/pairs/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $D/pairs/$1/b/serve.log)" >> $L
}
run_pair p5_c1_seed 1 0 1 7000 "--seed 1234"
run_pair p5_c16_a 16 0 1 13000 --gpu-metrics
run_pair p5_c16_b 16 1 0 18500 --gpu-metrics
run_pair p5_c64_a 64 0 1 22000 --gpu-metrics
run_pair p5_c64_b 64 1 0 30000 --gpu-metrics
python3 $IN/pair_table.py p5_c1_seed p5_c16_a p5_c16_b p5_c64_a p5_c64_b > $D/table_p5.txt 2>&1
echo "queue_p5_done $(date +%T)" >> $L
