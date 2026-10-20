#!/usr/bin/env bash
# The trimmed comments and tests of #2625 (228ce6fab, fixed_pr2625_v2.patch) and #2626 (93703161e, stacked on
# #2625, fixed_pr2626_v7.patch) on main 1ef7b8913: the touched suites on each branch, then each new test case
# against the code before its fix (the CUDA staging test on main t_m1, the RAS test on #2625 alone).
# usage (in the container): bash queue_trim_c20261012.sh
set -u
R=/data/c20261012
MAIN=1ef7b8913cd2a4a3c803f1c75fafad42537356ce
OUT=$R/trim
mkdir -p $OUT
add() {
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/$1 "$MAIN"
  git -C /workspace/$1 apply -3 $R/patches/m1ef7b8913_20261012/$2.patch
  git -C /workspace/$1 add -A
  git -C /workspace/$1 reset -q
  echo "$1 on $MAIN with $2 changed=$(git -C /workspace/$1 status --short | wc -l)" \
    | tee /workspace/$1/BRANCH_MARKER.txt >> $OUT/queue.txt
}
add t_f2625g fixed_pr2625_v2
add t_f2626g fixed_pr2626_v7
cp -r $R/inductor_t_f2626 $R/inductor_t_f2626g 2>/dev/null
(cd /workspace/t_f2625g && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_f2625g python3 -m pytest \
  tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfE > $OUT/units_2625.log 2>&1) &
(cd /workspace/t_f2626g && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/t_f2626g \
  TORCHINDUCTOR_CACHE_DIR=$R/inductor_t_f2626g python3 -m pytest \
  tests/unit_test/fun_cosyvoice3 tests/unit_test/sampling tests/unit_test/model_runner -q -p no:cacheprovider -rfE \
  > $OUT/units_2626.log 2>&1) &
wait
echo "units 2625: $(grep -E '[0-9]+ (passed|failed)' $OUT/units_2625.log | tail -1)" >> $OUT/queue.txt
echo "units 2626: $(grep -E '[0-9]+ (passed|failed)' $OUT/units_2626.log | tail -1)" >> $OUT/queue.txt
cp /workspace/t_f2625g/tests/unit_test/fun_cosyvoice3/test_model_runner.py $OUT/test_staging_on_main.py
(cd /workspace/t_m1 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_m1 python3 -m pytest \
  $OUT/test_staging_on_main.py -q -p no:cacheprovider -k pinned_staging > $OUT/prefix_staging.log 2>&1)
echo "staging test on main (expect failed): $(grep -E '[0-9]+ (passed|failed)' $OUT/prefix_staging.log | tail -1)" \
  >> $OUT/queue.txt
cp /workspace/t_f2626g/tests/unit_test/fun_cosyvoice3/test_model_runner.py $OUT/test_ras_on_2625.py
(cd /workspace/t_f2625g && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_f2625g python3 -m pytest \
  $OUT/test_ras_on_2625.py -q -p no:cacheprovider -k ras_masks > $OUT/prefix_ras.log 2>&1)
echo "ras test on 2625 (expect failed): $(grep -E '[0-9]+ (passed|failed)' $OUT/prefix_ras.log | tail -1)" \
  >> $OUT/queue.txt
echo "queue done $(date +%T)" >> $OUT/queue.txt
