#!/usr/bin/env bash
# #2631's split replay alone, cards 0 and 1 (the user holds 2 and 3): unit suites of the trees with
# the RoPE fix, then #2631 before the split (finals3) against #2631 with the split and the RoPE fix
# (finals5) at c16, c32 and c64 in both orders. The RoPE fix leaves the default budget's angles bit
# identical, so this pair isolates the split. Quality of the split path was scored in q1 and q2.
# usage (in the container): bash queue18_20261009.sh
R=/data/p20261009
cd $R
export ROOT=$R
CPUS0=0-31,64-95
CPUS1=32-63,96-127

units() {
  local tree=$1 card=$2 cpus=$3
  (
    cd /workspace/$tree
    export PYTHONPATH=/workspace/$tree CUDA_VISIBLE_DEVICES=$card
    python3 -c "import sglang_omni; print(sglang_omni.__file__)" > $R/units18/$tree.import.txt
    taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider \
      -m "accelerator and not benchmark" -rs > $R/units18/$tree.gpu.log 2>&1
    echo "gpu rc $?" >> $R/units18/$tree.done
    taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider \
      -m "not benchmark and not accelerator" -rs > $R/units18/$tree.cpu.log 2>&1
    echo "cpu rc $?" >> $R/units18/$tree.done
  )
}

mkdir -p $R/units18
echo "units18 $(date +%T)" >> $R/queue18.txt
units finals5 0 $CPUS0 &
units convk5 1 $CPUS1 &
wait
PAIR="CARD_0=0 CARD_1=1 CPUS_0=$CPUS0 CPUS_1=$CPUS1"
CELLS18="c16:16:::1 c32:32:::1 c64:64:::1"
echo "s1 $(date +%T)" >> $R/queue18.txt
env $PAIR CELLS="$CELLS18" SCORE_CELLS=none bash box/ab_pair.sh $R/s1 finals3 finals5 26000
echo "s2 $(date +%T)" >> $R/queue18.txt
env $PAIR CELLS="$CELLS18" SCORE_CELLS=none bash box/ab_pair.sh $R/s2 finals5 finals3 27000
echo "queue done $(date +%T)" >> $R/queue18.txt
