#!/usr/bin/env bash
# After the second-seed pair: what serving the vocoder without compile costs the first chunk, the vocoder bench (cold
# widths 8 and 9 included) on main compiled, the chain compiled and the chain without compile, twice each on card 3;
# alongside on card 2, the conv engine probe on the chain head (D9 step 1).
set -u
D=/data/chain_20261022
T=${T:-/workspace/t_marks}
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
M=/workspace/main_1b726ab64
C=${C:?the chain head tree}
L=$D/queue.log
until grep -q seed2_done $L; do sleep 20; done
echo "queue cold start $(date +%T) chain $(git -C $C rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
PYTHONPATH=$C python3 $HERE/conv_engine_probe.py --card 2 > $D/conv_engine_probe.txt 2>&1 &
probe=$!
for arm in "main $M" "chain $C" "chain-no-compile $C --no-compile" "chain-no-compile $C --no-compile" "chain $C" "main $M"; do
  set -- $arm
  echo "== bench $1" >> $D/bench_cold.log
  PYTHONPATH=$2 python3 $HERE/vocoder_decode_bench.py --card 3 ${3:-} 2>&1 | grep -E "^(cold|warm|tail) " >> $D/bench_cold.log
done
wait $probe
echo "conv probe rc $? $(date +%T)" >> $L
echo "queue_cold_done $(date +%T)" >> $L
