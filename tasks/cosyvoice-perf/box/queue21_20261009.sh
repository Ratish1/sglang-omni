#!/usr/bin/env bash
# Does replaying the whole history solve past the 7,680 frame top tier help in serving? #2631's code
# (finals6) in both arms, cards 0 and 1 only: the default flow_batch_admission_frames (ladder top
# 7,680) against 16000 (ladder top 15,872, every SeedTTS step held by one graph). That setting only
# gates buffered payload batching otherwise, so stream cells see the graph cap alone. Both arms carry
# the graph_hits probe. c16, c32, c64 in both orders, no scoring.
# usage (in the container): bash queue21_20261009.sh
R=/data/p20261009
cd $R
export ROOT=$R
H=$R/box/graph_hits
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-31,64-95 CPUS_1=32-63,96-127"
PROBE="PYTHONPATH=/workspace/finals6:$H OMNI_GRAPH_HITS=1"
BIG="--vocoder.factory.flow_batch_admission_frames 16000"
CELLS21="c16:16:::1 c32:32:::1 c64:64:::1"
echo "t1 $(date +%T)" >> $R/queue21.txt
env $PAIR CELLS="$CELLS21" SCORE_CELLS=none EXTRA_ENV_0="$PROBE" EXTRA_ENV_1="$PROBE" SERVE_ARGS_1="$BIG" \
  bash box/ab_pair.sh $R/t1 finals6 finals6 25000
echo "t2 $(date +%T)" >> $R/queue21.txt
env $PAIR CELLS="$CELLS21" SCORE_CELLS=none EXTRA_ENV_0="$PROBE" EXTRA_ENV_1="$PROBE" SERVE_ARGS_0="$BIG" \
  bash box/ab_pair.sh $R/t2 finals6 finals6 26000
echo "queue done $(date +%T)" >> $R/queue21.txt
