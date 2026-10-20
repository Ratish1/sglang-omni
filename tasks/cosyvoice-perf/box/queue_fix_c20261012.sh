#!/usr/bin/env bash
# Our fixes on top of #2625 and #2626 (each PR's head with upstream main 1ef7b8913 merged and our commit,
# patches/m1ef7b8913_20261012/fixed_pr<n>.patch), checked for regressions:
# 1. the whole unit suite on main (t_m1) and on each branch (t_f2625, t_f2626), failures compared,
# 2. the component tests (staged_ids_component on t_f2625, ras_component on t_f2626),
# 3. one same time A/B per PR against main with the barrier pair runner (ab_pair_sync.sh), both card orders:
#    seeded stream c1, stream c16 with WER and SIM, stream c64, long form c16.
# usage (in the container): bash queue_fix_c20261012.sh
set -u
R=/data/c20261012
IN=$R/in
MAIN=1ef7b8913cd2a4a3c803f1c75fafad42537356ce
PATCHES=$R/patches/m1ef7b8913_20261012
add() {
  local tree=$1 patch=${2:-}
  git -C /workspace/sglang-omni worktree add -q --detach "/workspace/$tree" "$MAIN"
  if [ -n "$patch" ]; then
    git -C "/workspace/$tree" apply -3 "$PATCHES/$patch.patch"
    git -C "/workspace/$tree" add -A
    git -C "/workspace/$tree" reset -q
  else
    :
  fi
  echo "$tree on $MAIN with ${patch:-nothing} changed=$(git -C "/workspace/$tree" status --short | wc -l)" \
    | tee "/workspace/$tree/BRANCH_MARKER.txt" >> $R/queue_fix.txt
}
add t_m1
add t_f2625 fixed_pr2625
add t_f2626 fixed_pr2626
mkdir -p $R/fix
units() {
  local tree=$1 card=$2
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/$tree timeout 3600 python3 -m pytest \
    tests/unit_test -q -p no:cacheprovider -rfE > $R/fix/units_$tree.log 2>&1)
  echo "units $tree: $(grep -E '[0-9]+ (passed|failed)' $R/fix/units_$tree.log | tail -1) $(date +%T)" >> $R/queue_fix.txt
}
units t_m1 0 &
units t_f2626 1 &
wait
units t_f2625 0
(cd /workspace/t_f2625 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_f2625 python3 -u $IN/staged_ids_component_c20261012.py) \
  > $R/fix/staged_ids.log 2>&1
echo "staged ids rc $? $(date +%T)" >> $R/queue_fix.txt
(cd /workspace/t_f2626 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_f2626 python3 -u $IN/ras_component_c20261012.py) \
  > $R/fix/ras_component.log 2>&1
echo "ras component rc $? $(date +%T)" >> $R/queue_fix.txt

export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1 c64:64:::1 L16:16:::1:$R/longform/meta.lst"
export SCORE_CELLS="c16"
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
for n in 2625 2626; do
  echo "f$n a $(date +%T) card0=t_m1 card1=t_f$n" >> $R/queue_fix.txt
  env $PAIR bash $IN/ab_pair_sync.sh $R/fix/p$n/a t_m1 t_f$n 38000
  echo "f$n b $(date +%T) card0=t_f$n card1=t_m1" >> $R/queue_fix.txt
  env $PAIR bash $IN/ab_pair_sync.sh $R/fix/p$n/b t_f$n t_m1 38000
done
echo "queue done $(date +%T)" >> $R/queue_fix.txt
