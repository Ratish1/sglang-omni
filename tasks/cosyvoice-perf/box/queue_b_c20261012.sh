#!/usr/bin/env bash
# Item B (Flow step inputs without a host wait, 7f612680f) checked on the box against main 870d77446:
# 1. the served vocoder over fixed streams in both trees, both cards, swapped (main against main and B against B
#    are the controls), outputs compared bit for bit, blocking calls counted by call site,
# 2. the Fun-CosyVoice3 unit suite on main and on B.
# usage (in the container): bash queue_b_c20261012.sh
set -u
R=/data/c20261012
IN=$R/in
MAIN=870d77446b3fae6e4f4f2a6964f7eae2f75e663a
OUT=$R/b
mkdir -p $OUT
git -C /workspace/sglang-omni fetch -q upstream main
add() {
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/$1 "$MAIN"
  if [ -n "$2" ]; then
    git -C /workspace/$1 apply -3 $R/patches/m870d77446_20261012/$2.patch
    git -C /workspace/$1 add -A
    git -C /workspace/$1 reset -q
  else
    :
  fi
  echo "$1 on $MAIN with ${2:-nothing} changed=$(git -C /workspace/$1 status --short | wc -l)" \
    | tee /workspace/$1/BRANCH_MARKER.txt >> $OUT/queue.txt
}
add t_bm ""
add t_b b_7f612680f
identity() {
  local tree=$1 card=$2 name=$3
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/$tree \
    TORCHINDUCTOR_CACHE_DIR=$R/inductor_$tree python3 -u $IN/flow_uploads_identity_c20261012.py $OUT/$name.pt) \
    > $OUT/identity_$name.log 2>&1
  echo "identity $name rc $?" >> $OUT/queue.txt
}
identity t_bm 0 main_card0 &
identity t_b 1 b_card1 &
wait
identity t_b 0 b_card0 &
identity t_bm 1 main_card1 &
wait
python3 $IN/flow_uploads_identity_c20261012.py --compare $OUT/main_card0.pt $OUT/main_card1.pt $OUT/b_card0.pt \
  >> $OUT/queue.txt 2>&1
python3 $IN/flow_uploads_identity_c20261012.py --compare $OUT/b_card1.pt $OUT/b_card0.pt $OUT/main_card1.pt \
  >> $OUT/queue.txt 2>&1
for tree in t_bm t_b; do
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/$tree \
    python3 -m pytest -q -p no:cacheprovider tests/unit_test/fun_cosyvoice3) > $OUT/suite_$tree.log 2>&1
  echo "suite $tree rc $? $(tail -1 $OUT/suite_$tree.log)" >> $OUT/queue.txt
done
echo done >> $OUT/queue.txt
