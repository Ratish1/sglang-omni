#!/usr/bin/env bash
# Same time pairs on cards 1 and 2 (NUMA node 1) after queue_mem_c20261010.sh: item 3 against its
# base (main with prerequisites 1 and 2, t_p12, against t_p123), then item 4 against item 3
# (t_p123 against t_stack), each in both card orders. Cells: seeded stream c1, stream c16, c32,
# c64, buffered c32, long form c16; WER and SIM of stream c16.
# usage (in the container): bash queue_pairs_c20261010.sh
R=/data/c20261010
IN=$R/in
until grep -q "queue done" $R/queue_mem.txt 2> /dev/null; do sleep 30; done
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1 c32:32:::1 c64:64:::1 b32:32:::0 L16:16:::1:$R/longform/meta.lst"
export SCORE_CELLS="c16"
PAIR="CARD_0=1 CARD_1=2 CPUS_0=32-47,96-111 CPUS_1=48-63,112-127"

pair() {
  local name=$1 tree0=$2 tree1=$3
  echo "$name $(date +%T) card1=$tree0 card2=$tree1" >> $R/queue_pairs.txt
  env $PAIR bash $IN/ab_pair.sh $R/$name $tree0 $tree1 33000
}

pair i3a t_p12 t_p123
pair i3b t_p123 t_p12
pair i4a t_p123 t_stack
pair i4b t_stack t_p123
echo "queue done $(date +%T)" >> $R/queue_pairs.txt
