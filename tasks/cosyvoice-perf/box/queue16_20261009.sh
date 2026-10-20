#!/usr/bin/env bash
# The split replay fix of #2631 (and #2632 merged on it) on cards 0 and 1 only (the user holds 2
# and 3). Card 0 is on NUMA node 0 and card 1 on node 1, so each arm gets its card's node: unit
# suites of both trees, a probed boot of both reading the solve graph hit rate per row, then main
# against #2631 at c1, c16, c32 and c64 in both orders, scored at c16 and c64.
# usage (in the container): bash queue16_20261009.sh
R=/data/p20261009
cd $R
export ROOT=$R
CPUS0=0-31,64-95
CPUS1=32-63,96-127
H=$R/box/graph_hits
STAGE4=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4

units() {
  local tree=$1 card=$2 cpus=$3
  (
    cd /workspace/$tree
    export PYTHONPATH=/workspace/$tree CUDA_VISIBLE_DEVICES=$card
    python3 -c "import sglang_omni; print(sglang_omni.__file__)" > $R/units16/$tree.import.txt
    taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider \
      -m "accelerator and not benchmark" -rs > $R/units16/$tree.gpu.log 2>&1
    echo "gpu rc $?" >> $R/units16/$tree.done
    taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider \
      -m "not benchmark and not accelerator" -rs > $R/units16/$tree.cpu.log 2>&1
    echo "cpu rc $?" >> $R/units16/$tree.done
  )
}

mkdir -p $R/units16
echo "units16 $(date +%T)" >> $R/queue16.txt
units finals4 0 $CPUS0 &
units convk4 1 $CPUS1 &
wait
echo "hits16 $(date +%T)" >> $R/queue16.txt
CELLS="c16:16:::1 c32:32:::1 c64:64:::1" EXTRA_ENV="PYTHONPATH=/workspace/finals4:$H OMNI_GRAPH_HITS=1" \
  bash $STAGE4/ab_boot.sh /workspace/finals4 $R/hits16/card0_finals4 0 29000 $CPUS0 $R/inductor_finals4 &
CELLS="c16:16:::1 c32:32:::1 c64:64:::1" EXTRA_ENV="PYTHONPATH=/workspace/convk4:$H OMNI_GRAPH_HITS=1" \
  bash $STAGE4/ab_boot.sh /workspace/convk4 $R/hits16/card1_convk4 1 29200 $CPUS1 $R/inductor_convk4 &
wait
PAIR="CARD_0=0 CARD_1=1 CPUS_0=$CPUS0 CPUS_1=$CPUS1"
CELLS16="c1:1:64:1234:1 c16:16:::1 c32:32:::1 c64:64:::1"
echo "q1 $(date +%T)" >> $R/queue16.txt
env $PAIR CELLS="$CELLS16" SCORE_CELLS="c16 c64" bash box/ab_pair.sh $R/q1 sglang-omni finals4 27000
echo "q2 $(date +%T)" >> $R/queue16.txt
env $PAIR CELLS="$CELLS16" SCORE_CELLS="c16 c64" bash box/ab_pair.sh $R/q2 finals4 sglang-omni 28000
echo "queue done $(date +%T)" >> $R/queue16.txt
