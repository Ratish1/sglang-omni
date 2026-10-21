#!/usr/bin/env bash
# V3 against itself on cards 2 and 3, seeded c1 without the digest probe: the probe syncs the device on every decode,
# so its byte identity across the cards leaves a timing-dependent cause open. Ends with the marker the PR queues wait on.
set -u
export PAIR_ROOT=/data/q3t_20261021/pairs
D=/data/q3t_20261021
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
V3=/workspace/v3pr_98b0bbc14
L=$D/queue.log
echo "queue v3 identity start $(date +%T) v3 $(git -C $V3 rev-parse HEAD:sglang_omni)" >> $L
python3 $IN/pair.py --label v3_self_c1_seed --tree $V3 --b-tree $V3 --concurrency 1 --a-card 2 --b-card 3 \
  --port 7000 --seed 1234 > $D/v3_self_c1_seed.out 2>&1
echo "v3_self_c1_seed rc $? $(date +%T) $(grep -h 'byte identical' $D/v3_self_c1_seed.out)" >> $L
echo "queue_digest_done $(date +%T)" >> $L
