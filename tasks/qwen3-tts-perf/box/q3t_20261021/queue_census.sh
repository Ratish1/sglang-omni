#!/usr/bin/env bash
# Which vocoder runner modes and widths serve c16 and c64 (state_in_place_20261021/MAP.md 7, decisions 2 and 3): main
# against itself on cards 2 and 3 with the census probe, which reads only host metadata, after the PR queues.
set -u
export PAIR_ROOT=/data/q3t_20261021/pairs
D=/data/q3t_20261021
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
S=$T/tasks/omni_step_profiling_20260918/scripts
M=/workspace/main_f31cf9e9a
L=$D/queue.log
until grep -q queue_v4m2_done $L; do sleep 30; done
echo "queue census start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni)" >> $L
for C in 16 64; do
  python3 $IN/pair.py --label census_c$C --tree $M --b-tree $M --concurrency $C --a-card 2 --b-card 3 --port 7000 \
    --decode-census > $D/census_c$C.out 2>&1
  echo "census_c$C rc $? $(date +%T)" >> $L
done
python3 $S/decode_census.py $PAIR_ROOT/census_c16/a/digest $PAIR_ROOT/census_c16/b/digest \
  $PAIR_ROOT/census_c64/a/digest $PAIR_ROOT/census_c64/b/digest > $D/census.txt 2>&1
echo "queue_census_done $(date +%T)" >> $L
