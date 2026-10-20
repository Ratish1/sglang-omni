#!/usr/bin/env bash
# The CI unit-test fix of #2631 (3af317c5f, main 8e33e31d7 merged) and #2632 (9f55f59ec), cards 0
# and 1 (the user holds 2 and 3). Trees finals9 and convk9 from patches/pr_ci_20261009 on the new
# main. Both CI lanes as test.yaml runs them: CPU with CUDA hidden, accelerator with the card. The
# empty ladder regression case against the pre-fix factory (prefix9: finals9 with the fix reversed),
# CUDA visible and hidden: it must fail there. Then one boot of each merged tree, stream c16.
# usage (in the container): bash queue23_20261009.sh
R=/data/p20261009
cd $R
export ROOT=$R
MAIN=8e33e31d77d96e43706cca1ad8ec77ff65664538
CPUS0=0-31,64-95
CPUS1=32-63,96-127
P=$R/patches_ci
F1=tests/unit_test/fun_cosyvoice3/test_vocoder.py::test_admission_budget_below_the_first_tier_keeps_finals_eager

git -C /workspace/sglang-omni fetch -q --depth 1 upstream $MAIN || { echo "fetch failed" >> $R/queue23.txt; exit 1; }
for tree in finals9 convk9 prefix9; do
  if [ -d /workspace/$tree ]; then
    echo "$tree exists" >> $R/queue23.txt
    exit 1
  else
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$tree $MAIN
  fi
done
git -C /workspace/finals9 apply $P/item1.patch
git -C /workspace/convk9 apply $P/item1.patch
git -C /workspace/convk9 apply $P/item2.patch
git -C /workspace/prefix9 apply $P/item1.patch
git -C /workspace/prefix9 apply -R $P/fix_stages.patch
for tree in finals9 convk9 prefix9; do
  echo "$tree changed=$(git -C /workspace/$tree status --short | wc -l)" >> $R/queue23.txt
done

units() {
  local tree=$1 card=$2 cpus=$3
  (
    cd /workspace/$tree
    export PYTHONPATH=/workspace/$tree
    python3 -c "import sglang_omni; print(sglang_omni.__file__)" > $R/units23/$tree.import.txt
    CUDA_VISIBLE_DEVICES=$card taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 \
      -q -p no:cacheprovider -m "accelerator and not benchmark" -rs > $R/units23/$tree.gpu.log 2>&1
    echo "gpu rc $?" >> $R/units23/$tree.done
    CUDA_VISIBLE_DEVICES="" taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 \
      -q -p no:cacheprovider -m "not benchmark and not accelerator" -rs > $R/units23/$tree.cpu.log 2>&1
    echo "cpu rc $?" >> $R/units23/$tree.done
  )
}

prefix_check() {
  (
    cd /workspace/prefix9
    export PYTHONPATH=/workspace/prefix9
    CUDA_VISIBLE_DEVICES=0 taskset -c $CPUS0 python3 -m pytest "$F1" -q -p no:cacheprovider \
      > $R/units23/prefix9.f1_cuda.log 2>&1
    echo "f1 cuda rc $?" >> $R/units23/prefix9.done
    CUDA_VISIBLE_DEVICES="" taskset -c $CPUS0 python3 -m pytest "$F1" -q -p no:cacheprovider \
      > $R/units23/prefix9.f1_hidden.log 2>&1
    echo "f1 hidden rc $?" >> $R/units23/prefix9.done
  )
}

mkdir -p $R/units23
echo "units23 $(date +%T)" >> $R/queue23.txt
units finals9 0 $CPUS0 &
units convk9 1 $CPUS1 &
wait
prefix_check
PAIR="CARD_0=0 CARD_1=1 CPUS_0=$CPUS0 CPUS_1=$CPUS1"
echo "b2 $(date +%T)" >> $R/queue23.txt
env $PAIR CELLS="c16:16:::1" SCORE_CELLS=none bash box/ab_pair.sh $R/b2 convk9 finals9 29000
echo "queue done $(date +%T)" >> $R/queue23.txt
