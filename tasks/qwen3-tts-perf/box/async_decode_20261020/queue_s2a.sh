#!/usr/bin/env bash
# S2a (decode inputs from device slot state, replay from committed frames) against its base S1: touched units, c1
# seeded byte identity, forced retraction at c16 against the same tree, a c16 pair and c64 pairs in both card orders.
# Waits for queue_s1 to finish so the cards hold one run at a time.
set -u
D=/data/async_decode_20261020
IN=${T:-/workspace/t_marks}/tasks/qwen3-tts-perf/box/async_decode_20261020
S1=/workspace/s1_7173ba3de
S2A=/workspace/s2a_66b5971e8
L=$D/queue.log
until grep -q queue_s1_done $L; do sleep 30; done
echo "queue s2a start $(date +%T) s1 $(git -C $S1 rev-parse HEAD:sglang_omni) s2a $(git -C $S2A rev-parse HEAD:sglang_omni)" >> $L
(cd $S2A && PYTHONPATH=$S2A timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts tests/unit_test/model_runner \
  tests/unit_test/scheduling -q -p no:cacheprovider -rfE > $D/units/$(basename $S2A).log 2>&1)
echo "units $(basename $S2A) rc $? $(date +%T)" >> $L
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
