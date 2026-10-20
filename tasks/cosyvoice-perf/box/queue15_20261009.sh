#!/usr/bin/env bash
# The #2631 rework on the 10-09 lease, cards 1 and 2 (both NUMA 1, card 0 is on NUMA 0): unit
# suites of both PR trees, a warm-up boot of both reading the solve graph capture logs, then main
# against #2631 at c1 and c16 in both orders.
# usage (in the container): bash queue15_20261009.sh
R=/data/p20261009
cd $R
export ROOT=$R
PAIR="CARD_0=1 CARD_1=2 CPUS_0=32-47,96-111 CPUS_1=48-63,112-127"

units() {
  local tree=$1 card=$2 cpus=$3
  (
    cd /workspace/$tree
    export PYTHONPATH=/workspace/$tree CUDA_VISIBLE_DEVICES=$card
    python3 -c "import sglang_omni; print(sglang_omni.__file__)" > $R/units/$tree.import.txt
    taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider \
      -m "accelerator and not benchmark" -rs > $R/units/$tree.gpu.log 2>&1
    echo "gpu rc $?" >> $R/units/$tree.done
    taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider \
      -m "not benchmark and not accelerator" -rs > $R/units/$tree.cpu.log 2>&1
    echo "cpu rc $?" >> $R/units/$tree.done
  )
}

mkdir -p $R/units
echo "units $(date +%T)" >> $R/queue.txt
units finals3 1 32-47,96-111 &
units convk3 2 48-63,112-127 &
wait
echo "warm $(date +%T)" >> $R/queue.txt
env $PAIR CELLS="w16:16:64::1 wb16:16:64::0" SCORE_CELLS=none bash box/ab_pair.sh $R/warm finals3 convk3 26000
echo "p1 $(date +%T)" >> $R/queue.txt
env $PAIR CELLS="c1:1:64:1234:1 c16:16:::1" SCORE_CELLS=c16 bash box/ab_pair.sh $R/p1 sglang-omni finals3 27000
echo "p2 $(date +%T)" >> $R/queue.txt
env $PAIR CELLS="c1:1:64:1234:1 c16:16:::1" SCORE_CELLS=c16 bash box/ab_pair.sh $R/p2 finals3 sglang-omni 28000
echo "queue done $(date +%T)" >> $R/queue.txt
