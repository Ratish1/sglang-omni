#!/usr/bin/env bash
# Two A/Bs on current main 3304fffff (#2625 and #2626 in), each a same time pair in both card orders on the barrier
# runner: item 1 (32748a7a2, t_i1m) against main at c16 with WER and SIM, for its PR's RTF; then #2659 with main
# merged (c7e2b92a3, t_2659m) against main at seeded c1 and c16 with WER and SIM, Fun-CosyVoice3 only, plus the
# seeded c1 audio compared file by file. Trees are main 1ef7b8913 plus the patch to each head.
# usage (in the container): bash queue_n1_c20261012.sh
set -u
R=/data/c20261012
IN=$R/in
MAIN=1ef7b8913cd2a4a3c803f1c75fafad42537356ce
OUT=$R/n1
mkdir -p $OUT
add() {
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/$1 "$MAIN"
  git -C /workspace/$1 apply -3 $R/patches/m1ef7b8913_20261012/$2.patch
  git -C /workspace/$1 add -A
  git -C /workspace/$1 reset -q
  echo "$1 on $MAIN with $2 changed=$(git -C /workspace/$1 status --short | wc -l)" \
    | tee /workspace/$1/BRANCH_MARKER.txt >> $OUT/queue.txt
}
add t_mnow main_3304fffff
add t_2659m pr2659m_c7e2b92a3
for tree in t_mnow t_i1m t_2659m; do
  [ -d $R/inductor_$tree ] || cp -r $R/inductor_t_rf_new $R/inductor_$tree
done
export ROOT=$R
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
export SCORE_CELLS="c16"

export CELLS="c16:16:::1"
echo "i1 a $(date +%T) card0=t_mnow card1=t_i1m" >> $OUT/queue.txt
env $PAIR bash $IN/ab_pair_sync.sh $OUT/i1/a t_mnow t_i1m 38000
echo "i1 b $(date +%T) card0=t_i1m card1=t_mnow" >> $OUT/queue.txt
env $PAIR bash $IN/ab_pair_sync.sh $OUT/i1/b t_i1m t_mnow 38000

export CELLS="c1:1:64:1234:1 c16:16:::1"
echo "p59 a $(date +%T) card0=t_mnow card1=t_2659m" >> $OUT/queue.txt
env $PAIR bash $IN/ab_pair_sync.sh $OUT/p59/a t_mnow t_2659m 38000
echo "p59 b $(date +%T) card0=t_2659m card1=t_mnow" >> $OUT/queue.txt
env $PAIR bash $IN/ab_pair_sync.sh $OUT/p59/b t_2659m t_mnow 38000

for pair in a b; do
  main_dir=$(ls -d $OUT/p59/$pair/card*_t_mnow)
  pr_dir=$(ls -d $OUT/p59/$pair/card*_t_2659m)
  same=0; total=0
  for wav in $main_dir/c1/audio/*; do
    total=$((total + 1))
    cmp -s "$wav" "$pr_dir/c1/audio/$(basename "$wav")" && same=$((same + 1))
  done
  echo "p59 $pair seeded c1 audio identical: $same of $total" >> $OUT/queue.txt
done
echo "queue done $(date +%T)" >> $OUT/queue.txt
