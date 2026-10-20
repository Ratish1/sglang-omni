#!/usr/bin/env bash
# V3 (perf/qwen3-tts-fused-decoder-transformer 6f6039ba6: the vocoder's decoder transformer on fused kernels, traced
# in pure torch under torch.compile) against main bcd0d8c68: the touched units, the transformer bench at the served
# shapes (eager-captured and compiled), then same-time pairs with each arm's SM Issue over its benchmark: c16 and c64
# in both card orders, c1, full corpus, WER and SIM; after each pair, the serve logs' graph capture and compile
# failures. Ports skip what the other container holds (16000s, 24000 to 25999) and omni's NCCL block (29500 to 29899).
# The first run on bb673b57a stopped at its first pair: that head's precompile could not trace SGLang's FusedOp
# dispatch, so its server ran without the warm codec runner.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
S=$T/tasks/omni_step_profiling_20260918/scripts
MAIN=/workspace/m_bcd0d8c68
V3=/workspace/v3_6f6039ba6
L=$D/queue.log
mkdir -p $D/units
echo "queue v3 start $(date +%T) marks $(git -C $T rev-parse HEAD) main $(git -C $MAIN rev-parse HEAD:sglang_omni) v3 $(git -C $V3 rev-parse HEAD:sglang_omni)" >> $L
(cd $V3 && PYTHONPATH=$V3 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider -rfE \
  > $D/units/v3.log 2>&1)
echo "units rc $? $(date +%T)" >> $L
(cd $V3 && PYTHONPATH=$V3 timeout 1800 python3 $S/vocoder_transformer_bench.py --widths 1 2 4 8 16 32 64 --rows 1 4 8 16 \
  > $D/bench_eager.txt 2>&1)
echo "bench eager rc $? $(date +%T)" >> $L
(cd $V3 && PYTHONPATH=$V3 timeout 1800 python3 $S/vocoder_transformer_bench.py --compiled --widths 1 2 8 --rows 1 8 \
  > $D/bench_compiled.txt 2>&1)
echo "bench compiled rc $? $(date +%T)" >> $L
run_pair() {
  python3 $IN/pair.py --label $1 --tree $MAIN --b-tree $V3 --concurrency $2 --a-card $3 --b-card $4 --port $5 \
    --gpu-metrics > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' /data/v3_20261020/pairs/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' /data/v3_20261020/pairs/$1/b/serve.log)" >> $L
}
run_pair v3_c16_a 16 0 1 14000
run_pair v3_c16_b 16 1 0 17000
run_pair v3_c64_a 64 0 1 19000
run_pair v3_c64_b 64 1 0 21000
run_pair v3_c1 1 0 1 27000
python3 $IN/pair_table.py v3_c16_a v3_c16_b v3_c64_a v3_c64_b v3_c1 > $D/table_v3.txt 2>&1
echo "queue_v3_done $(date +%T)" >> $L
