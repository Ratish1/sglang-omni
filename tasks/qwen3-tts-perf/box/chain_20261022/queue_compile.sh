#!/usr/bin/env bash
# Whether the vocoder's width-8 graphs still need torch.compile on the chain, and how far main's own compile switch
# moves quality: the vocoder bench of the chain head compiled and not (two runs each, card 3), then a seeded c1 pair of
# main against main with the vocoder's compile off (codes held equal; the D2 conv stack without compile decodes the bits
# main does without compile), both card orders, SNR, WER and SIM.
set -u
export PAIR_ROOT=/data/chain_20261022/pairs
D=/data/chain_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
M=/workspace/main_1b726ab64
C=${C:?the chain head tree}
L=$D/queue.log
echo "queue compile start $(date +%T) chain $(git -C $C rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
for flag in "" "--no-compile" "--no-compile" ""; do
  echo "== bench chain ${flag:-compiled}" >> $D/bench_compile.log
  PYTHONPATH=$C python3 $HERE/vocoder_decode_bench.py --card 3 $flag 2>&1 | grep -E "^(warm|tail) " >> $D/bench_compile.log
done
echo "bench compile done $(date +%T)" >> $L
for order in "a 2 3 7000" "b 3 2 7600"; do
  set -- $order
  python3 $IN/pair.py --label nocompile_c1_$1 --tree $M --b-tree $M --concurrency 1 --a-card $2 --b-card $3 --port $4 \
    --seed 1234 --b-args "--vocoder.factory.incremental_codec_compile false" > $D/nocompile_c1_$1.out 2>&1
  echo "nocompile_c1_$1 rc $? $(date +%T) $(grep 'byte identical' $D/nocompile_c1_$1.out)" >> $L
done
python3 $SEEDED/seeded_quality.py nocompile_c1_a nocompile_c1_b > $D/seeded_quality_nocompile.txt 2>&1
echo "queue_compile_done $(date +%T)" >> $L
