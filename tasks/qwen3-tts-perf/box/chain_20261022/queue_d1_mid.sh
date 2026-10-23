#!/usr/bin/env bash
# After queue_d1_seedtts.sh: #3102's first chunk at reference lengths no served cell has, 300, 450 and 579 frames
# (24 to 46 s, which now pad into the 581-wide tail graph instead of chaining window graphs), with 1,100 (88 s) for
# reference; first_chunk_bench.py main, D1, D1, main on card 3.
set -u
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
M=${M:?the main tree}
P1=${P1:?the D1 tree}
L=$D/queue.log
OUT=$D/first_chunk_bench_d1_lengths.log
until grep -q queue_d1_seedtts_done $L; do sleep 20; done
echo "queue d1 mid start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
for arm in "main $M" "d1 $P1" "d1 $P1" "main $M"; do
  set -- $arm
  echo "== arm $1 $(date +%T)" >> $OUT
  (cd $2 && PYTHONPATH=$2 python3 $HERE/first_chunk_bench.py --card 3 --reference-frames 300 450 579 1100 >> $OUT 2>&1)
  echo "rc $?" >> $OUT
done
echo "queue_d1_mid_done $(date +%T)" >> $L
