#!/usr/bin/env bash
# Byte identity at c1, where batches cannot differ between runs (seeded c16 matched main on itself in #3177's checks
# only at c1): seeded SeedTTS EN ICL non-streaming c1 (seed 1234), main, the two-frame minimum (A) and the frame
# ceilings (C), each compared with main. Card 1 only, after queue_icl3_bisect.sh.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
M=${M:?the main tree}
A=${A:?the two-frame minimum tree}
C=${C:?the frame ceilings tree}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
export ARM_ROOT=$D/arms
CARD=1
L=$D/queue.log
until grep -q queue_icl3_bisect_done $L; do
  sleep 30
done
echo "queue seeded c1 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
for arm in "main $M" "A $A" "C $C"; do
  set -- $arm
  python3 $ARM --label seeded_c1_$1 --tree $2 --card $CARD --port 21000 --no-stream --concurrency 1 --seed 1234 >> $L 2>&1
  echo "arm seeded_c1_$1 rc $? $(date +%T)" >> $L
done
for arm in A C; do
  echo "seeded c1 main against $arm: $(python3 $ARM --compare seeded_c1_main seeded_c1_$arm)" >> $L
done
echo "queue_seeded_c1_done $(date +%T)" >> $L
