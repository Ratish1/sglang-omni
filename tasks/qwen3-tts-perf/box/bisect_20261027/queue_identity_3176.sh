#!/usr/bin/env bash
# #3176 against its base on non-streaming audio: SeedTTS EN on 1.7B Base, seeded (1234), c1, non-streaming, one
# arm at a time with serve_arm.py: the parent of #3176's merge, then the merge; byte identity by serve_arm.py --compare.
# Runs after queue_prompt_compare.sh. Card 0 only.
set -u
D=${D:?the results directory}
PROMPT=${PROMPT:?the prompt-compare directory, for its done line}
BASE=${BASE:?the tree at the parent of #3176's merge}
MERGED=${MERGED:?the tree at #3176's merge}
T=${T:?the kit tree}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
L=$D/queue.log
mkdir -p $D
until grep -q queue_prompt_compare_done $PROMPT/queue.log 2> /dev/null; do
  sleep 20
done
echo "queue identity 3176 start $(date +%T) base $(git -C $BASE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')merged $(git -C $MERGED rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
for spec in "base $BASE 26200" "merged $MERGED 26600"; do
  set -- $spec
  ARM_ROOT=$D/arms python3 $ARM --label c1s_$1 --tree $2 --card 0 --port $3 --concurrency 1 --seed 1234 --no-stream >> $L 2>&1
  echo "arm $1 rc $? $(date +%T)" >> $L
done
ARM_ROOT=$D/arms python3 $ARM --compare c1s_base c1s_merged >> $L 2>&1
echo "queue_identity_3176_done $(date +%T)" >> $L
