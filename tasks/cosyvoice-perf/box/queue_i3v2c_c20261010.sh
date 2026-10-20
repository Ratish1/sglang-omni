#!/usr/bin/env bash
# A same time repeat of pair i3v2a's card order (t_p12 on card 1, t_p123v2 on card 2) with both
# inductor caches warm: i3v2a's item 3 arm compiled cold and started its cells minutes late.
# usage (in the container): bash queue_i3v2c_c20261010.sh
R=/data/c20261010
until grep -q "queue done" $R/queue_i3v2.txt 2> /dev/null; do sleep 20; done
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1"
export SCORE_CELLS="c16"
echo "i3v2c $(date +%T) card1=t_p12 card2=t_p123v2" >> $R/queue_i3v2c.txt
env CARD_0=1 CARD_1=2 CPUS_0=32-47,96-111 CPUS_1=48-63,112-127 \
  bash $R/in/ab_pair.sh $R/i3v2c t_p12 t_p123v2 33000
echo "queue done $(date +%T)" >> $R/queue_i3v2c.txt
