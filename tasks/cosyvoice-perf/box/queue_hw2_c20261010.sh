#!/usr/bin/env bash
# Step 1 (HiFT over a window, the phase carried) on main ba456b68f. Trees t_main (main) and t_hw
# (main with hift_window.patch). Card 0: the Fun-CosyVoice3 unit suites of both trees, then the
# HiFT bench of both (timing, and accuracy against a float64 source). Cards 1 and 2 meanwhile:
# t_main against t_hw in both orders, seeded stream c1, stream c16 (WER and SIM), stream c64,
# long form c16.
# Second launch: the trees exist (t_hw holds the F0 alignment fix), so no tree setup here.
# usage (in the container): bash queue_hw2_c20261010.sh
R=/data/c20261010h
IN=$R/in
MAIN=ba456b68f516dd14e841fa150ba708c7801a2b65
mkdir -p $R/units $R/bench
units() {
  local tree=$1
  (
    cd /workspace/$tree
    export PYTHONPATH=/workspace/$tree
    python3 -c "import sglang_omni; print(sglang_omni.__file__)" > $R/units/$tree.import.txt
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q \
      -p no:cacheprovider -m "accelerator and not benchmark" -rs > $R/units/$tree.gpu.log 2>&1
    echo "gpu rc $? $(date +%T)" >> $R/units/$tree.done
    CUDA_VISIBLE_DEVICES="" taskset -c 0-15,64-79 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q \
      -p no:cacheprovider -m "not benchmark and not accelerator" -rs > $R/units/$tree.cpu.log 2>&1
    echo "cpu rc $? $(date +%T)" >> $R/units/$tree.done
  )
}

bench() {
  local tree=$1
  (
    cd /workspace/$tree
    export PYTHONPATH=/workspace/$tree
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 $IN/hift_window_bench_c20261010.py \
      --out $R/bench/$tree.json > $R/bench/$tree.log 2>&1
    echo "bench $tree rc $? $(date +%T)" >> $R/queue_hw2.txt
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 $IN/hift_window_bench_c20261010.py \
      --accuracy --out $R/bench/${tree}_acc.json > $R/bench/${tree}_acc.log 2>&1
    echo "accuracy $tree rc $? $(date +%T)" >> $R/queue_hw2.txt
  )
}

echo "card 0 start $(date +%T)" >> $R/queue_hw2.txt
(units t_hw; units t_main; bench t_main; bench t_hw; echo "card 0 done $(date +%T)" >> $R/queue_hw2.txt) &
CARD0=$!
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1 c64:64:::1 L16:16:::1:/data/c20261010/longform/meta.lst"
export SCORE_CELLS="c16"
PAIR="CARD_0=1 CARD_1=2 CPUS_0=32-47,96-111 CPUS_1=48-63,112-127"
echo "hwa $(date +%T) card1=t_main card2=t_hw" >> $R/queue_hw2.txt
env $PAIR bash $IN/ab_pair.sh $R/hwa t_main t_hw 33000
echo "hwb $(date +%T) card1=t_hw card2=t_main" >> $R/queue_hw2.txt
env $PAIR bash $IN/ab_pair.sh $R/hwb t_hw t_main 33000
wait $CARD0
echo "queue done $(date +%T)" >> $R/queue_hw2.txt
