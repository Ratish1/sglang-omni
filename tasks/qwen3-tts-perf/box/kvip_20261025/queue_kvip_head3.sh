#!/usr/bin/env bash
# New lease (10-25 evening): the PR head ff72643e3 against what was measured. It differs from e6cbe020a only by the
# dropped ring clone, the FA3 op's explicit fake output and comments, so: the decode bench main, head, head, main (card
# 3; the compiled width-8 replay runs through the op's new fake), then a seeded c1 pair of e6cbe020a against the head
# (cards 2 and 3; every output must be byte identical), scored with 64 transcriptions in flight.
set -u
D=/data/kvip1026
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/chain_20261022
M=${M:?the main tree}
P=${P:?the measured tree}
H=${H:?the PR head tree}
L=$D/queue.log
mkdir -p $D
echo "queue kvip head3 start $(date +%T) main $(git -C $M rev-parse HEAD) measured $(git -C $P rev-parse HEAD) head $(git -C $H rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
for arm in "main $M" "head $H" "headb $H" "mainb $M"; do
  set -- $arm
  (cd $2 && PYTHONPATH=$2 timeout 1800 python3 $HERE/vocoder_decode_bench.py --card 3 > $D/bench_$1.txt 2>&1)
  echo "bench $1 rc $? $(grep -E 'warm width   8 batch 8|tail width  64 batch 8' $D/bench_$1.txt | awk '{print $1, $3, $5, $8}' | tr '\n' ' ')$(date +%T)" >> $L
done
python3 $K/v3_20261020/pair.py --label kvh_s_c1 --tree $P --b-tree $H --concurrency 1 --a-card 2 --b-card 3 --port 4400 \
  --seed 1234 > $D/kvh_s_c1.out 2>&1
echo "kvh_s_c1 rc $? $(date +%T) $(grep 'byte identical' $D/kvh_s_c1.out) asr concurrency $(grep -h -m1 -i 'ASR concurrency' $PAIR_ROOT/kvh_s_c1/b/score--transcribe-only.log)" >> $L
python3 $K/v3_20261020/pair_table.py kvh_s_c1 > $D/table_kvh.txt 2>&1
echo "queue_kvip_head3_done $(date +%T)" >> $L
