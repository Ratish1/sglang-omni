#!/usr/bin/env bash
# The vocoder's captured decodes, main against #3097 with the fp32 SnakeBeta, for #3097's body: vocoder_decode_bench.py
# main, fix, fix, main on card 3.
set -u
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
M=${M:?the main tree}
PF=${PF:?the fix tree}
L=$D/queue.log
OUT=$D/bench_main_snake_fp32.log
echo "queue bench main fix start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) fix $(git -C $PF rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
for arm in "main $M" "fix $PF" "fix $PF" "main $M"; do
  set -- $arm
  echo "== $1 $(date +%T)" >> $OUT
  (cd $2 && TORCHINDUCTOR_CACHE_DIR=$D/inductor_bench_mf_$1 PYTHONPATH=$2 python3 $HERE/vocoder_decode_bench.py --card 3 > $D/bench_mf_last.txt 2>&1)
  echo "rc $?" >> $OUT
  grep -E "^(cold|warm|tail) " $D/bench_mf_last.txt >> $OUT
done
echo "queue_bench_main_fix_done $(date +%T)" >> $L
