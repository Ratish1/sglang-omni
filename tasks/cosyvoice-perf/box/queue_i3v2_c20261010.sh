#!/usr/bin/env bash
# Item 3 with the reserve measured from the largest admitted step: trees t_i3v2 (main and item 3)
# and t_p123v2 (main, prerequisites 1 and 2, item 3), their Fun-CosyVoice3 unit suites (cards 0 and
# 3), then t_p12 against t_p123v2 on cards 1 and 2 in both orders, seeded stream c1 and stream c16,
# WER and SIM of c16; and once, on card 3, the 32 row worst step that ran out of memory (w32b).
# usage (in the container): bash queue_i3v2_c20261010.sh
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
    | tee /workspace/$tree/BRANCH_MARKER.txt >> $R/queue_i3v2.txt
}
add t_i3v2 i3v2_prefix_pool_measured_reserve
add t_p123v2 stack_p1_p2_i3v2

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

worst32() {
  EXTRA_ENV="PYTHONPATH=/workspace/t_p123v2:$KIT/prefix_count OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1 \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
CUDA_COREDUMP_FILE=$R/cores/core_%h_%p_%t" \
    CELLS="W32:32:32:1234:1:$R/worst32/meta.lst" SERVE_ARGS="--vocoder.factory.max_batch_size 32" \
    bash $KIT/ab_boot.sh /workspace/t_p123v2 $R/w32v2/card3_p123v2 3 34200 16-31,80-95 $R/inductor_t_p123v2
}

echo "units $(date +%T)" >> $R/queue_i3v2.txt
(units t_i3v2 0 0-15,64-79) &
(units t_p123v2 3 16-31,80-95; worst32) &
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1"
export SCORE_CELLS="c16"
PAIR="CARD_0=1 CARD_1=2 CPUS_0=32-47,96-111 CPUS_1=48-63,112-127"
echo "i3v2a $(date +%T) card1=t_p12 card2=t_p123v2" >> $R/queue_i3v2.txt
env $PAIR bash $IN/ab_pair.sh $R/i3v2a t_p12 t_p123v2 33000
echo "i3v2b $(date +%T) card1=t_p123v2 card2=t_p12" >> $R/queue_i3v2.txt
env $PAIR bash $IN/ab_pair.sh $R/i3v2b t_p123v2 t_p12 33000
wait
echo "queue done $(date +%T)" >> $R/queue_i3v2.txt
