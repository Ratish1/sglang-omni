#!/usr/bin/env bash
# The #2626 redraw fold, checked on the box: the PR head a25a8c480 (t_rf_old), the moved test before
# the inline f0c3d51d6 (t_rf_test) and the fold c5be7dca0 (t_rf_new), each on main 1ef7b8913 by patch.
# 1. redraw_repeated_tokens on identical inputs in t_rf_old and t_rf_new, ids compared bit for bit,
# 2. the moved CUDA test on t_rf_test and t_rf_new, 3. the three suites on t_rf_new.
# usage (in the container): bash queue_rf_c20261012.sh
set -u
R=/data/c20261012
IN=$R/in
MAIN=1ef7b8913cd2a4a3c803f1c75fafad42537356ce
OUT=$R/rf
mkdir -p $OUT
add() {
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/$1 "$MAIN"
  git -C /workspace/$1 apply -3 $R/patches/m1ef7b8913_20261012/$2.patch
  git -C /workspace/$1 add -A
  git -C /workspace/$1 reset -q
  echo "$1 on $MAIN with $2 changed=$(git -C /workspace/$1 status --short | wc -l)" \
    | tee /workspace/$1/BRANCH_MARKER.txt >> $OUT/queue.txt
}
add t_rf_old rf_old_a25a8c480
add t_rf_test rf_test_f0c3d51d6
add t_rf_new rf_new_c5be7dca0
for tree in t_rf_old t_rf_new; do
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/$tree \
    TORCHINDUCTOR_CACHE_DIR=$R/inductor_$tree python3 -u $IN/ras_fold_identity_c20261012.py $OUT/$tree.pt) \
    > $OUT/identity_$tree.log 2>&1
  echo "identity $tree rc $?" >> $OUT/queue.txt
done
python3 $IN/ras_fold_identity_c20261012.py --compare $OUT/t_rf_old.pt $OUT/t_rf_new.pt >> $OUT/queue.txt 2>&1
for tree in t_rf_test t_rf_new; do
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/$tree \
    TORCHINDUCTOR_CACHE_DIR=$R/inductor_$tree python3 -m pytest tests/unit_test/fun_cosyvoice3/test_model_runner.py \
    -q -p no:cacheprovider -k "row_subset or ras_masks or pinned_staging" -rA > $OUT/moved_$tree.log 2>&1)
  echo "moved test on $tree: $(grep -cE '^PASSED' $OUT/moved_$tree.log) passed, $(grep -E '[0-9]+ (passed|failed)' $OUT/moved_$tree.log | tail -1)" \
    >> $OUT/queue.txt
done
(cd /workspace/t_rf_new && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/t_rf_new \
  TORCHINDUCTOR_CACHE_DIR=$R/inductor_t_rf_new python3 -m pytest \
  tests/unit_test/fun_cosyvoice3 tests/unit_test/sampling tests/unit_test/model_runner -q -p no:cacheprovider -rfE \
  > $OUT/units_new.log 2>&1)
echo "units t_rf_new: $(grep -E '[0-9]+ (passed|failed)' $OUT/units_new.log | tail -1)" >> $OUT/queue.txt
echo "queue done $(date +%T)" >> $OUT/queue.txt
