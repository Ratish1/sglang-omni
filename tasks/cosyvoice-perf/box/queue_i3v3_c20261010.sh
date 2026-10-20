#!/usr/bin/env bash
# Item 3 with the largest admitted step measured under force_eager (i3v2 compiled extra PackedDiT
# graphs at the measurement's size, and stream c1 lost about 1 % in all three pairs). Trees t_i3v3
# and t_p123v3 with a fresh inductor cache; unit suites on card 0; one warm boot of t_p123v3 on card
# 3 (seeded c1, logs the eager measurement, warms the cache); then t_p12 against t_p123v3 on cards 1
# and 2 in both orders, seeded stream c1 and stream c16, WER and SIM of c16.
# usage (in the container): bash queue_i3v3_c20261010.sh
R=/data/c20261010
IN=$R/in
P=$IN/main_cf496151d_20261010
MAIN=cf496151dfa6eed47e7ed466e451a6e3eb1204d6
KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4

add() {
  local tree=$1 patch=$2
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/$tree $MAIN
  git -C /workspace/$tree apply $P/$patch.patch
  echo "$tree on $MAIN with $patch changed=$(git -C /workspace/$tree status --short | wc -l)" \
    | tee /workspace/$tree/BRANCH_MARKER.txt >> $R/queue_i3v3.txt
}
add t_i3v3 i3v3_prefix_pool_measured_reserve
add t_p123v3 stack_p1_p2_i3v3

units() {
  local tree=$1 card=$2 cpus=$3
  (
    cd /workspace/$tree
    export PYTHONPATH=/workspace/$tree
    python3 -c "import sglang_omni; print(sglang_omni.__file__)" > $R/units/$tree.import.txt
    CUDA_VISIBLE_DEVICES=$card taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q \
      -p no:cacheprovider -m "accelerator and not benchmark" -rs > $R/units/$tree.gpu.log 2>&1
    echo "gpu rc $? $(date +%T)" >> $R/units/$tree.done
    CUDA_VISIBLE_DEVICES="" taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q \
      -p no:cacheprovider -m "not benchmark and not accelerator" -rs > $R/units/$tree.cpu.log 2>&1
    echo "cpu rc $? $(date +%T)" >> $R/units/$tree.done
  )
}

echo "units and warm boot $(date +%T)" >> $R/queue_i3v3.txt
(units t_i3v3 0 0-15,64-79; units t_p123v3 0 0-15,64-79) &
UNITS=$!
CELLS="c1:1:64:1234:1" bash $KIT/ab_boot.sh /workspace/t_p123v3 $R/warm_v3/card3_p123v3 3 34200 \
  16-31,80-95 $R/inductor_t_p123v3
echo "fxgraph entries after the warm boot: $(find $R/inductor_t_p123v3 -path '*fxgraph*' -type f | wc -l)" \
  >> $R/queue_i3v3.txt
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1"
export SCORE_CELLS="c16"
PAIR="CARD_0=1 CARD_1=2 CPUS_0=32-47,96-111 CPUS_1=48-63,112-127"
echo "i3v3a $(date +%T) card1=t_p12 card2=t_p123v3" >> $R/queue_i3v3.txt
env $PAIR bash $IN/ab_pair.sh $R/i3v3a t_p12 t_p123v3 33000
echo "i3v3b $(date +%T) card1=t_p123v3 card2=t_p12" >> $R/queue_i3v3.txt
env $PAIR bash $IN/ab_pair.sh $R/i3v3b t_p123v3 t_p12 33000
echo "fxgraph entries after the pairs: $(find $R/inductor_t_p123v3 -path '*fxgraph*' -type f | wc -l)" \
  >> $R/queue_i3v3.txt
wait $UNITS
echo "queue done $(date +%T)" >> $R/queue_i3v3.txt
