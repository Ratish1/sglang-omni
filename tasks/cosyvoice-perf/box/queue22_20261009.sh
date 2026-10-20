#!/usr/bin/env bash
# The SGLang custom op registration (#2631 c5997f892, #2632 58c369006) and the empty ladder fix
# (#2631 045cab72b), cards 0 and 1 (the user holds 2 and 3). Trees finals8 (#2631) and convk8
# (#2632) from patches/pr_review_20261009_ops. Unit suites of both trees, then one boot each:
# convk8 at the default budget (both graph sets captured through the new ops) and finals8 at
# flow_batch_admission_frames 3 (no whole history tier, finals eager), stream c16, no scoring.
# usage (in the container): bash queue22_20261009.sh
R=/data/p20261009
cd $R
export ROOT=$R
MAIN=a093c7a2542fc513d0d850ed090bf9d900e945b5
CPUS0=0-31,64-95
CPUS1=32-63,96-127

for tree in finals8 convk8; do
  if [ ! -d /workspace/$tree ]; then
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$tree $MAIN
  else
    echo "$tree exists" >> $R/queue22.txt
    exit 1
  fi
done
git -C /workspace/finals8 apply $R/patches_ops/item1.patch
git -C /workspace/convk8 apply $R/patches_ops/item1.patch
git -C /workspace/convk8 apply $R/patches_ops/item2.patch
for tree in finals8 convk8; do
  echo "$tree changed=$(git -C /workspace/$tree status --short | wc -l)" >> $R/queue22.txt
done

units() {
  local tree=$1 card=$2 cpus=$3
  (
    cd /workspace/$tree
    export PYTHONPATH=/workspace/$tree CUDA_VISIBLE_DEVICES=$card
    python3 -c "import sglang_omni; print(sglang_omni.__file__)" > $R/units22/$tree.import.txt
    taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider \
      -m "accelerator and not benchmark" -rs > $R/units22/$tree.gpu.log 2>&1
    echo "gpu rc $?" >> $R/units22/$tree.done
    taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider \
      -m "not benchmark and not accelerator" -rs > $R/units22/$tree.cpu.log 2>&1
    echo "cpu rc $?" >> $R/units22/$tree.done
  )
}

mkdir -p $R/units22
echo "units22 $(date +%T)" >> $R/queue22.txt
units finals8 0 $CPUS0 &
units convk8 1 $CPUS1 &
wait
PAIR="CARD_0=0 CARD_1=1 CPUS_0=$CPUS0 CPUS_1=$CPUS1"
echo "b1 $(date +%T)" >> $R/queue22.txt
env $PAIR CELLS="c16:16:::1" SCORE_CELLS=none \
  SERVE_ARGS_1="--vocoder.factory.flow_batch_admission_frames 3" \
  bash box/ab_pair.sh $R/b1 convk8 finals8 28000
echo "queue done $(date +%T)" >> $R/queue22.txt
