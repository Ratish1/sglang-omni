#!/usr/bin/env bash
# After queue_kvip_after.sh: L3's paired first audio read 1.8 and 2.7 ms later with the rings (28 and 31 prompts).
# A long reference's first chunk is a tail decode of its reach plus one frame (about 576 frames), whose frames past
# the kept 71 went to one trash row: some 4,000 stores to one address a layer at batch 8. The tail bench at 64, 128
# and the reach width, card 3: main, the measured head's code (9180d5ec6, one trash row) and the spread trash
# (e6cbe020a, the trash slot's rows by position), in the order main, one, spread, spread, one, main.
set -u
D=/data/kvip1025
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
M=${M:?the main tree}
O=${O:?the one-trash-row tree}
S=${S:?the spread-trash tree}
L=$D/queue.log
until grep -q queue_kvip_after_done $L; do
  sleep 20
done
echo "queue kvip tail start $(date +%T) one $(git -C $O rev-parse HEAD) spread $(git -C $S rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
for arm in "main $M" "one $O" "spread $S" "spreadb $S" "oneb $O" "mainb $M"; do
  set -- $arm
  (cd $2 && PYTHONPATH=$2 timeout 1800 python3 $HERE/vocoder_decode_bench.py --card 3 --reach-tail --only-tail \
    > $D/tail_$1.txt 2>&1)
  echo "tail bench $1 rc $? $(grep -E 'tail width' $D/tail_$1.txt | awk '{print $3, $5, $8}' | tr '\n' ' ')$(date +%T)" >> $L
done
echo "queue_kvip_tail_done $(date +%T)" >> $L
