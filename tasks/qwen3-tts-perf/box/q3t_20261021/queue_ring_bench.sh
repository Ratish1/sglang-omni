#!/usr/bin/env bash
# ring_step_bench.py over the served decode shapes (census: warm widths 1 to 8, width 8 the most, compiled widths
# 1, 2 and 8; reference-prefixed tails around 56 frames), both variants of a shape on one card, one process each,
# card 2 and card 3 taking alternate shapes, after V4's pairs.
set -u
D=/data/q3t_20261021
L=$D/queue.log
B=/workspace/t_marks/tasks/qwen3-tts-perf/box/q3t_20261021/ring_step_bench.py
V3=/workspace/v3pr_76b0b7aa2
until grep -q queue_v4m2_done $L; do sleep 30; done
echo "queue ring bench start $(date +%T)" >> $L
bench_card() {
  CARD=$1; shift
  for SHAPE in "$@"; do
    for VARIANT in gathered ring; do
      (cd $V3 && PYTHONPATH=$V3 timeout 900 python3 $B --card $CARD --variant $VARIANT $SHAPE) \
        >> $D/ring_bench_card$CARD.txt 2>&1
    done
  done
}
bench_card 2 "--rows 1 --width 8 --compiled" "--rows 16 --width 8 --compiled" "--rows 8 --width 4" \
  "--rows 1 --width 56 --first" &
bench_card 3 "--rows 8 --width 8 --compiled" "--rows 8 --width 2 --compiled" "--rows 8 --width 56 --first" &
wait
echo "queue_ring_bench_done $(date +%T)" >> $L
