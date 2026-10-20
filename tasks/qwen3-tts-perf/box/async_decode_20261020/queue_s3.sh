#!/usr/bin/env bash
# S3 (decode lookahead) against its base S2a: touched units, c1 seeded byte identity, c16 and c64 pairs in both card
# orders, and a non-stream c16 pair. First reruns S2a's c64 card-order-a pair, whose arm b lost port 42200 to an
# ephemeral client socket. Ports stay below the box's ephemeral range (32768 to 60999). Waits for queue_s2a.
set -u
D=/data/async_decode_20261020
IN=${T:-/workspace/t_marks}/tasks/qwen3-tts-perf/box/async_decode_20261020
S1=/workspace/s1_d1eef0380
S2A=/workspace/s2a_9db6227f1
S3=/workspace/s3_ce54a94fb
L=$D/queue.log
until grep -q queue_s2a_done $L; do sleep 30; done
python3 $IN/pair.py --label s2a_c64_a2 --tree $S1 --b-tree $S2A --concurrency 64 --a-card 0 --b-card 1 \
  --port 10000 > $D/s2a_c64_a2.out 2>&1
echo "s2a_c64_a2 rc $? $(date +%T)" >> $L
echo "queue s3 start $(date +%T) s2a $(git -C $S2A rev-parse HEAD:sglang_omni) s3 $(git -C $S3 rev-parse HEAD:sglang_omni)" >> $L
(cd $S3 && PYTHONPATH=$S3 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts tests/unit_test/model_runner \
  tests/unit_test/scheduling -q -p no:cacheprovider -rfE > $D/units/$(basename $S3).log 2>&1)
echo "units $(basename $S3) rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s3_c1_seed --tree $S2A --b-tree $S3 --concurrency 1 --a-card 0 --b-card 1 \
  --port 12000 --seed 1234 > $D/s3_c1_seed.out 2>&1
echo "s3_c1_seed rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s3_c16_a --tree $S2A --b-tree $S3 --concurrency 16 --a-card 0 --b-card 1 \
  --port 14000 > $D/s3_c16_a.out 2>&1
echo "s3_c16_a rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s3_c16_b --tree $S2A --b-tree $S3 --concurrency 16 --a-card 1 --b-card 0 \
  --port 16000 > $D/s3_c16_b.out 2>&1
echo "s3_c16_b rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s3_c64_a --tree $S2A --b-tree $S3 --concurrency 64 --a-card 0 --b-card 1 \
  --port 18000 > $D/s3_c64_a.out 2>&1
echo "s3_c64_a rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s3_c64_b --tree $S2A --b-tree $S3 --concurrency 64 --a-card 1 --b-card 0 \
  --port 20000 > $D/s3_c64_b.out 2>&1
echo "s3_c64_b rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label s3_c16_nostream --tree $S2A --b-tree $S3 --concurrency 16 --a-card 0 --b-card 1 \
  --port 22000 --no-stream > $D/s3_c16_nostream.out 2>&1
echo "s3_c16_nostream rc $? $(date +%T)" >> $L
python3 $IN/pair_table.py s2a_c64_a2 s3_c1_seed s3_c16_a s3_c16_b s3_c64_a s3_c64_b s3_c16_nostream \
  > $D/table_s3.txt 2>&1
echo "queue_s3_done $(date +%T)" >> $L
