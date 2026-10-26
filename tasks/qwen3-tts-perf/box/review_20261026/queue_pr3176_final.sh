#!/usr/bin/env bash
# #3176 after its review fixes and the upstream merge (dev/pr3176-final): SeedTTS EN ICL c16, non-streaming and
# streaming, scored, so the merged vocoder serves both modes before the push. Card 1 only, after queue_seeded_c1.sh.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
P=${P:?the merged PR tree}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
export ARM_ROOT=$D/arms
CARD=1
L=$D/queue.log
until grep -q queue_seeded_c1_done $L; do
  sleep 30
done
echo "queue pr3176 final start $(date +%T) tree $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
python3 $ARM --label pr3176_c16_nonstream --tree $P --card $CARD --port 21000 --no-stream --concurrency 16 --score >> $L 2>&1
echo "arm pr3176_c16_nonstream rc $? $(date +%T)" >> $L
python3 $ARM --label pr3176_c16_stream --tree $P --card $CARD --port 21000 --concurrency 16 --score >> $L 2>&1
echo "arm pr3176_c16_stream rc $? $(date +%T)" >> $L
echo "queue_pr3176_final_done $(date +%T)" >> $L
