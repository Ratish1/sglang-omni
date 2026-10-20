#!/usr/bin/env bash
# A v2 (afd99d535: CosyVoice only, stream passed into prepare, no shared helper) on main 204ba29a5 (tree t_a2)
# against main (t_m):
# 1. v2's construction against main's on the real tables (a2_identity_c20261013.py), card 0, while the Fun-CosyVoice3,
#    Qwen3-TTS pipeline, preprocessing and staging units run on t_a2, card 1 (main's run of the same set: a/units_t_m.log),
# 2. the race probe on both trees at the same time, c16,
# 3. same time pairs in both card orders, seeded c1 and c16 with WER and SIM, then the seeded c1 audio compared.
# Ports below the ephemeral range. usage (in the container): bash queue_a2_c20261013.sh
set -u
R=/data/c20261013
IN=$R/in
MAIN=204ba29a5
OUT=$R/a2
mkdir -p $OUT
git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_a2 "$MAIN"
git -C /workspace/t_a2 apply -3 $R/patches/m204ba29a5_20261013/a2_afd99d535.patch
git -C /workspace/t_a2 add -A
git -C /workspace/t_a2 reset -q
echo "t_a2 on $MAIN with a2_afd99d535 changed=$(git -C /workspace/t_a2 status --short | wc -l)" \
  | tee /workspace/t_a2/BRANCH_MARKER.txt >> $OUT/queue.txt
[ -d $R/inductor_t_a2 ] || cp -r $R/inductor_t_a $R/inductor_t_a2

(cd /workspace/t_a2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_a2 \
  python3 -u $IN/a2_identity_c20261013.py $R/longform/meta.lst) > $OUT/identity.log 2>&1 &
IDENTITY=$!
TESTS="tests/unit_test/fun_cosyvoice3 tests/unit_test/qwen3_tts/test_pipeline.py tests/unit_test/preprocessing tests/unit_test/utils/test_cuda_staging.py"
(cd /workspace/t_a2 && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/t_a2 \
  python3 -m pytest -q -p no:cacheprovider $TESTS) > $OUT/units_t_a2.log 2>&1
echo "units t_a2 rc $? $(grep -E 'passed|failed' $OUT/units_t_a2.log | tail -1)" >> $OUT/queue.txt
wait $IDENTITY
echo "identity rc $? $(tail -3 $OUT/identity.log | tr '\n' ' ')" >> $OUT/queue.txt
grep -E "^FAILED" $OUT/units_t_a2.log | sort > $OUT/failed_t_a2.txt
echo "t_a2 failing where main passes: $(comm -13 $R/a/failed_t_m.txt $OUT/failed_t_a2.txt | wc -l)" >> $OUT/queue.txt

export ROOT=$R
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
PROBE=$IN/race_probe
echo "race $(date +%T) card0=t_m card1=t_a2" >> $OUT/queue.txt
env $PAIR SCORE_CELLS="none" CELLS="c16:16:::1" \
  EXTRA_ENV_0="PYTHONPATH=$PROBE:/workspace/t_m RACE_PROBE=1" \
  EXTRA_ENV_1="PYTHONPATH=$PROBE:/workspace/t_a2 RACE_PROBE=1" \
  bash $IN/ab_pair_sync.sh $OUT/race t_m t_a2 27000
for arm in $OUT/race/card0_t_m $OUT/race/card1_t_a2; do
  echo "race $(basename $arm) patched=$(grep -c 'race_probe runner patched' $arm/serve.log)" \
    "mismatches=$(grep -c 'race_probe MISMATCH' $arm/serve.log) last: $(grep 'race_probe batch' $arm/serve.log | tail -1)" \
    >> $OUT/queue.txt
done

export SCORE_CELLS="c16"
export CELLS="c1:1:64:1234:1 c16:16:::1"
echo "pa a $(date +%T) card0=t_m card1=t_a2" >> $OUT/queue.txt
env $PAIR bash $IN/ab_pair_sync.sh $OUT/a t_m t_a2 25000
echo "pa b $(date +%T) card0=t_a2 card1=t_m" >> $OUT/queue.txt
env $PAIR bash $IN/ab_pair_sync.sh $OUT/b t_a2 t_m 26000
for pair in a b; do
  main_dir=$(ls -d $OUT/$pair/card*_t_m)
  a_dir=$(ls -d $OUT/$pair/card*_t_a2)
  same=0; total=0
  for wav in $main_dir/c1/audio/*; do
    total=$((total + 1))
    cmp -s "$wav" "$a_dir/c1/audio/$(basename "$wav")" && same=$((same + 1))
  done
  echo "pa $pair seeded c1 audio identical: $same of $total" >> $OUT/queue.txt
done
echo "queue done $(date +%T)" >> $OUT/queue.txt
