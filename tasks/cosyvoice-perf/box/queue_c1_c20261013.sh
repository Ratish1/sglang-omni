#!/usr/bin/env bash
# C1 served, before any repo edit: tree t_c1 is main 204ba29a5 unchanged, and its arm puts c1_patch (the proposed
# patch_sine_harmonics applied at import) first on PYTHONPATH.
# 1. the profile of main (t_m, card 0) and C1 (t_c1, card 1) at the same time (cosy_profile_run_c20261013.sh),
# 2. same time pairs in both card orders, seeded c1 and c16 with WER and SIM, then the seeded c1 audio compared.
# usage (in the container): bash queue_c1_c20261013.sh
set -u
R=/data/c20261013
IN=$R/in
MAIN=204ba29a5
OUT=$R/c1
PATCH=$IN/c1_patch
mkdir -p $OUT
[ -d /workspace/t_c1 ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_c1 "$MAIN"
echo "t_c1 on $MAIN with nothing changed=$(git -C /workspace/t_c1 status --short | wc -l), C1 by c1_patch on PYTHONPATH" \
  | tee /workspace/t_c1/BRANCH_MARKER.txt >> $OUT/queue.txt
[ -d $R/inductor_t_c1 ] || cp -r $R/inductor_t_m $R/inductor_t_c1

echo "profile $(date +%T)" >> $OUT/queue.txt
PEER=$OUT/prof/t_c1 bash $IN/cosy_profile_run_c20261013.sh t_m 0 0-15,64-79 43800 $OUT/prof/t_m &
PEER=$OUT/prof/t_m EXTRA_PATH=$PATCH bash $IN/cosy_profile_run_c20261013.sh t_c1 1 16-31,80-95 44000 $OUT/prof/t_c1 &
wait
grep -c "c1_patch applied" $OUT/prof/t_c1/serve.log >> $OUT/queue.txt
grep -c "c1_patch" $OUT/prof/t_m/serve.log >> $OUT/queue.txt

export ROOT=$R
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
export SCORE_CELLS="c16"
export CELLS="c1:1:64:1234:1 c16:16:::1"
echo "pa a $(date +%T) card0=t_m card1=t_c1" >> $OUT/queue.txt
env $PAIR EXTRA_ENV_1="PYTHONPATH=/workspace/t_c1:$PATCH" bash $IN/ab_pair_sync.sh $OUT/a t_m t_c1 44200
echo "pa b $(date +%T) card0=t_c1 card1=t_m" >> $OUT/queue.txt
env $PAIR EXTRA_ENV_0="PYTHONPATH=/workspace/t_c1:$PATCH" bash $IN/ab_pair_sync.sh $OUT/b t_c1 t_m 44600
for pair in a b; do
  echo "pa $pair c1_patch applied: $(grep -c 'c1_patch applied' $OUT/$pair/card*_t_c1/serve.log)," \
    "in main: $(grep -c 'c1_patch' $OUT/$pair/card*_t_m/serve.log)" >> $OUT/queue.txt
  main_dir=$(ls -d $OUT/$pair/card*_t_m)
  c1_dir=$(ls -d $OUT/$pair/card*_t_c1)
  same=0; total=0
  for wav in $main_dir/c1/audio/*; do
    total=$((total + 1))
    cmp -s "$wav" "$c1_dir/c1/audio/$(basename "$wav")" && same=$((same + 1))
  done
  echo "pa $pair seeded c1 audio identical: $same of $total" >> $OUT/queue.txt
done
echo "queue done $(date +%T)" >> $OUT/queue.txt
