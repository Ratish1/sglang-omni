#!/usr/bin/env bash
# Seeded stream c1 under nsys on card 0 for t_main then t_hw3, to read the AR's decode steps
# beside the vocoder's (step 1 slows the AR's decode at c1 by 3 to 4 % in hwa, hwb, hwfa).
# usage (in the container): bash hwq_card0_c20261010.sh
R=/data/c20261010h
for tree in t_main t_hw3; do
  bash $R/in/cosy_profile_c1_c20261010.sh $tree 0 0-15,64-79 34000 $R/prof2_c1_$tree
  echo "profile $tree $(date +%T)" >> $R/hwq2.txt
done
echo "done $(date +%T)" >> $R/hwq2.txt
