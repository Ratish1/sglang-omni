#!/usr/bin/env bash
# S1 (repetition penalty on device slot state) against main bcd0d8c68: touched unit suites on both trees, c1 seeded
# byte identity, a c16 pair and c64 pairs in both card orders, then the table. Arm a is main in every pair.
set -u
D=/data/async_decode_20261020
IN=${T:-/workspace/t_marks}/tasks/qwen3-tts-perf/box/async_decode_20261020
MAIN=/workspace/m_bcd0d8c68
S1=/workspace/s1_7173ba3de
L=$D/queue.log
mkdir -p $D/units
echo "queue s1 start $(date +%T) main $(git -C $MAIN rev-parse HEAD:sglang_omni) s1 $(git -C $S1 rev-parse HEAD:sglang_omni)" >> $L
for TREE in $MAIN $S1; do
  NAME=$(basename $TREE)
  (cd $TREE && PYTHONPATH=$TREE timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts tests/unit_test/model_runner \
    tests/unit_test/scheduling -q -p no:cacheprovider -rfE > $D/units/$NAME.log 2>&1)
  echo "units $NAME rc $? $(date +%T)" >> $L
done
python3 $IN/pair.py --label s1_c1_seed --tree $MAIN --b-tree $S1 --concurrency 1 --a-card 0 --b-card 1 \
  --port 21000 --seed 1234 > $D/s1_c1_seed.out 2>&1
echo "s1_c1_seed rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s1_c16_a --tree $MAIN --b-tree $S1 --concurrency 16 --a-card 0 --b-card 1 \
  --port 24000 > $D/s1_c16_a.out 2>&1
echo "s1_c16_a rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s1_c64_a --tree $MAIN --b-tree $S1 --concurrency 64 --a-card 0 --b-card 1 \
  --port 27000 > $D/s1_c64_a.out 2>&1
echo "s1_c64_a rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s1_c64_b --tree $MAIN --b-tree $S1 --concurrency 64 --a-card 1 --b-card 0 \
  --port 30000 > $D/s1_c64_b.out 2>&1
echo "s1_c64_b rc $? $(date +%T)" >> $L
python3 $IN/pair_table.py s1_c1_seed s1_c16_a s1_c64_a s1_c64_b > $D/table_s1.txt 2>&1
echo "queue_s1_done $(date +%T)" >> $L
