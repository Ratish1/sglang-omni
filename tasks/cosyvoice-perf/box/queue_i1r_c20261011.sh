#!/usr/bin/env bash
# Item 1 repeats: is the c16 and c64 loss of float32 logits (pairs i1/a and i1/b, -0.9 / -1.3 % and
# -2.5 / -4.0 %) larger than a cell's spread on one server? Main (t_m2) against item 1 (t_i1), both card
# orders, stream c16 and stream c64 three times each per server, alternating, no scoring (WER and SIM were
# gated in i1/a and i1/b). i1_serve_split_c20261011.py then gives per cell the AR's running requests and
# steps per second and the vocoder's steps and rows, to test the AR and vocoder coupling across every run.
# usage (in the container): bash queue_i1r_c20261011.sh
set -u
R=/data/c20261011
IN=$R/in
export ROOT=$R
export CELLS="c16a:16:::1 c64a:64:::1 c16b:16:::1 c64b:64:::1 c16c:16:::1 c64c:64:::1"
export SCORE_CELLS="none"
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
mkdir -p $R/i1r
echo "i1r a $(date +%T) card0=t_m2 card1=t_i1" >> $R/queue_i1r.txt
env $PAIR bash $IN/ab_pair.sh $R/i1r/a t_m2 t_i1 37000
echo "i1r b $(date +%T) card0=t_i1 card1=t_m2" >> $R/queue_i1r.txt
env $PAIR bash $IN/ab_pair.sh $R/i1r/b t_i1 t_m2 37000
echo "queue done $(date +%T)" >> $R/queue_i1r.txt
