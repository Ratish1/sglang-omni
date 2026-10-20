#!/usr/bin/env bash
# S2a (decode inputs from device slot state, replay from committed frames) against its base S1: touched units on both
# heads (S1's test doubles fixed in d1eef0380, same runtime tree as the S1 pairs), c1 seeded byte identity, forced
# retraction at c16 against the same tree, a c16 pair and c64 pairs in both card orders.
# Waits for queue_s1 to finish so the cards hold one run at a time.
set -u
D=/data/async_decode_20261020
IN=${T:-/workspace/t_marks}/tasks/qwen3-tts-perf/box/async_decode_20261020
S1=/workspace/s1_d1eef0380
S2A=/workspace/s2a_9db6227f1
L=$D/queue.log
until grep -q queue_s1_done $L; do sleep 30; done
echo "queue s2a start $(date +%T) s1 $(git -C $S1 rev-parse HEAD:sglang_omni) s2a $(git -C $S2A rev-parse HEAD:sglang_omni)" >> $L
for TREE in $S1 $S2A; do
  (cd $TREE && PYTHONPATH=$TREE timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts tests/unit_test/model_runner \
    tests/unit_test/scheduling -q -p no:cacheprovider -rfE > $D/units/$(basename $TREE).log 2>&1)
  echo "units $(basename $TREE) rc $? $(date +%T)" >> $L
done
python3 $IN/pair.py --label s2a_c1_seed --tree $S1 --b-tree $S2A --concurrency 1 --a-card 0 --b-card 1 \
  --port 33000 --seed 1234 > $D/s2a_c1_seed.out 2>&1
echo "s2a_c1_seed rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s2a_retract_c16 --tree $S2A --concurrency 16 --a-card 0 --b-card 1 --port 36000 \
  --b-env "SGLANG_TEST_RETRACT=1" > $D/s2a_retract_c16.out 2>&1
echo "s2a_retract_c16 rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s2a_c16_a --tree $S1 --b-tree $S2A --concurrency 16 --a-card 0 --b-card 1 \
  --port 39000 > $D/s2a_c16_a.out 2>&1
echo "s2a_c16_a rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s2a_c64_a --tree $S1 --b-tree $S2A --concurrency 64 --a-card 0 --b-card 1 \
  --port 42000 > $D/s2a_c64_a.out 2>&1
echo "s2a_c64_a rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s2a_c64_b --tree $S1 --b-tree $S2A --concurrency 64 --a-card 1 --b-card 0 \
  --port 45000 > $D/s2a_c64_b.out 2>&1
echo "s2a_c64_b rc $? $(date +%T)" >> $L
python3 $IN/pair_table.py s2a_c1_seed s2a_retract_c16 s2a_c16_a s2a_c64_a s2a_c64_b > $D/table_s2a.txt 2>&1
echo "queue_s2a_done $(date +%T)" >> $L
