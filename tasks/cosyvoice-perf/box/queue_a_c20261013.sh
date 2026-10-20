#!/usr/bin/env bash
# A (preprocessing on its own stream with host id keys, 83f484f98) on main 204ba29a5 (tree t_a) against main (t_m):
# 1. A's construction against main's on the real tables (a_identity_c20261013.py), card 0,
# 2. the Fun-CosyVoice3, Qwen3-TTS pipeline, preprocessing and staging unit tests on both trees,
# 3. same time pairs in both card orders, seeded c1 and c16 with WER and SIM, then the seeded c1 audio compared.
# usage (in the container): bash queue_a_c20261013.sh
set -u
R=/data/c20261013
IN=$R/in
MAIN=204ba29a5
OUT=$R/a
mkdir -p $OUT
git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_a "$MAIN"
git -C /workspace/t_a apply -3 $R/patches/m204ba29a5_20261013/a_83f484f98.patch
git -C /workspace/t_a add -A
git -C /workspace/t_a reset -q
echo "t_a on $MAIN with a_83f484f98 changed=$(git -C /workspace/t_a status --short | wc -l)" \
  | tee /workspace/t_a/BRANCH_MARKER.txt >> $OUT/queue.txt
[ -d $R/inductor_t_a ] || cp -r $R/inductor_t_m $R/inductor_t_a

(cd /workspace/t_a && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_a \
  python3 -u $IN/a_identity_c20261013.py $R/longform/meta.lst) > $OUT/identity.log 2>&1
echo "identity rc $? $(tail -3 $OUT/identity.log | tr '\n' ' ')" >> $OUT/queue.txt

TESTS="tests/unit_test/fun_cosyvoice3 tests/unit_test/qwen3_tts/test_pipeline.py tests/unit_test/preprocessing tests/unit_test/utils/test_cuda_staging.py"
for tree in t_m t_a; do
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/$tree \
    python3 -m pytest -q -p no:cacheprovider $TESTS) > $OUT/units_$tree.log 2>&1
  echo "units $tree rc $? $(grep -E 'passed|failed' $OUT/units_$tree.log | tail -1)" >> $OUT/queue.txt
done
grep -E "^FAILED" $OUT/units_t_m.log | sort > $OUT/failed_t_m.txt
grep -E "^FAILED" $OUT/units_t_a.log | sort > $OUT/failed_t_a.txt
echo "t_a failing where main passes: $(comm -13 $OUT/failed_t_m.txt $OUT/failed_t_a.txt | wc -l)" >> $OUT/queue.txt

export ROOT=$R
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
export SCORE_CELLS="c16"
export CELLS="c1:1:64:1234:1 c16:16:::1"
echo "pa a $(date +%T) card0=t_m card1=t_a" >> $OUT/queue.txt
env $PAIR bash $IN/ab_pair_sync.sh $OUT/a t_m t_a 42000
echo "pa b $(date +%T) card0=t_a card1=t_m" >> $OUT/queue.txt
env $PAIR bash $IN/ab_pair_sync.sh $OUT/b t_a t_m 42000
for pair in a b; do
  main_dir=$(ls -d $OUT/$pair/card*_t_m)
  a_dir=$(ls -d $OUT/$pair/card*_t_a)
  same=0; total=0
  for wav in $main_dir/c1/audio/*; do
    total=$((total + 1))
    cmp -s "$wav" "$a_dir/c1/audio/$(basename "$wav")" && same=$((same + 1))
  done
  echo "pa $pair seeded c1 audio identical: $same of $total" >> $OUT/queue.txt
done
echo "queue done $(date +%T)" >> $OUT/queue.txt
