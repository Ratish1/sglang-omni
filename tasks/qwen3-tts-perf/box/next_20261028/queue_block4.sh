#!/usr/bin/env bash
# Block 4 of 10-28: main on Pipeshift's bar on an H100, the like-for-like run against their published H100 numbers
# (10-27's was an H200): Nari Labs' client unmodified at its pinned commit, 1.7B CustomVoice at default settings, Ryan,
# English, Poisson arrivals at 1, 4, 8, 12, 20, 30, 40 req/s, 30 s warm-up and 120 s measured, seeds 0, 1 and 2 one
# after another on card 0 (Pipeshift ran one card per run), each on a fresh server. Waits for block 3.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BAR=$T/tasks/qwen3-tts-perf/box/bar_20261027
until grep -q "queue block3 done" $D/queue_3.log 2> /dev/null; do sleep 30; done
mkdir -p $D/bar
echo "queue block4 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $D/bar/queue.log
for seed in 0 1 2; do
  D=$D/bar TREE=/workspace/n28_main T=$T LABEL=main_s$seed SEED=$seed PORT=30000 bash $BAR/queue_baseline.sh
done
echo "queue block4 done $(date +%T)" >> $D/bar/queue.log
