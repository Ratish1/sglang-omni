#!/usr/bin/env bash
# Item 5 (the AR prefill through SGLang's breakable graph) against the stack it sits on (t_stack
# against t_stack5), both card orders, cards 1 and 2 after queue_pairs_c20261010.sh. Same cells.
# usage (in the container): bash queue_i5_c20261010.sh
R=/data/c20261010
IN=$R/in
until grep -q "queue done" $R/queue_pairs.txt 2> /dev/null; do sleep 30; done
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1 c32:32:::1 c64:64:::1 b32:32:::0 L16:16:::1:$R/longform/meta.lst"
export SCORE_CELLS="c16"
PAIR="CARD_0=1 CARD_1=2 CPUS_0=32-47,96-111 CPUS_1=48-63,112-127"
echo "i5a $(date +%T) card1=t_stack card2=t_stack5" >> $R/queue_i5.txt
env $PAIR bash $IN/ab_pair.sh $R/i5a t_stack t_stack5 33000
echo "i5b $(date +%T) card1=t_stack5 card2=t_stack" >> $R/queue_i5.txt
env $PAIR bash $IN/ab_pair.sh $R/i5b t_stack5 t_stack 33000
echo "queue done $(date +%T)" >> $R/queue_i5.txt
