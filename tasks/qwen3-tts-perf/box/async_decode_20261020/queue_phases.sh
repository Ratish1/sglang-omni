#!/usr/bin/env bash
# The scheduler loop split in situ, no profiler: S2a and S3 each with the step phase probe (probe/qwen3-tts-step-phases-*
# on the fork), paired at c64 and c16 in both card orders with full scoring, then the pair and phase tables.
# Waits for queue_s3.sh.
set -u
D=/data/async_decode_20261020
IN=${T:-/workspace/t_marks}/tasks/qwen3-tts-perf/box/async_decode_20261020
S2A=/workspace/p_s2a_3a3639810
S3=/workspace/p_s3_86ed86b97
L=$D/queue.log
until grep -q queue_s3_done $L; do sleep 30; done
echo "queue phases start $(date +%T) s2a $(git -C $S2A rev-parse HEAD:sglang_omni) s3 $(git -C $S3 rev-parse HEAD:sglang_omni)" >> $L
python3 $IN/pair.py --label phases_c64_a --tree $S2A --b-tree $S3 --concurrency 64 --a-card 0 --b-card 1 \
  --port 24000 --step-phases > $D/phases_c64_a.out 2>&1
echo "phases_c64_a rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label phases_c64_b --tree $S2A --b-tree $S3 --concurrency 64 --a-card 1 --b-card 0 \
  --port 26000 --step-phases > $D/phases_c64_b.out 2>&1
echo "phases_c64_b rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label phases_c16_a --tree $S2A --b-tree $S3 --concurrency 16 --a-card 0 --b-card 1 \
  --port 28000 --step-phases > $D/phases_c16_a.out 2>&1
echo "phases_c16_a rc $? $(date +%T)" >> $L
python3 $IN/pair.py --label phases_c16_b --tree $S2A --b-tree $S3 --concurrency 16 --a-card 1 --b-card 0 \
  --port 30000 --step-phases > $D/phases_c16_b.out 2>&1
echo "phases_c16_b rc $? $(date +%T)" >> $L
python3 $IN/pair_table.py phases_c64_a phases_c64_b phases_c16_a phases_c16_b > $D/table_phases.txt 2>&1
python3 $IN/step_phases_table.py phases_c64_a phases_c64_b phases_c16_a phases_c16_b > $D/step_phases.txt 2>&1
echo "queue_phases_done $(date +%T)" >> $L
