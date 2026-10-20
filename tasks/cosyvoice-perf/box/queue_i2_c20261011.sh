#!/usr/bin/env bash
# Item 2's design A/B (ARCH_AUDIT section R item 2, #2626), chained on item 1. Arm A t_i2a: main, item 1,
# #2626 as it is (omni's temperature 0.7, top k 20, top p 0.8, repetition penalty 1.21, then the repeated
# token redrawn from the full distribution at the request temperature with the candidate masked). Arm B
# t_i2b: arm A with i2b_cosyvoice_sampling_arm (CosyVoice3's sampling: temperature 1, top k 25, top p 0.8, no
# penalty, the redraw from the full distribution candidate included, as ras_sampling's random_sampling).
# Both card orders: seeded stream c1, stream c16 twice (the first scored), long form c16. Gate: WER, SIM and
# requests at the token cap. The inductor caches start from t_i1's (the Flow code is the same), so both arms
# boot warm.
# usage (in the container): bash queue_i2_c20261011.sh
set -u
R=/data/c20261011
IN=$R/in
MAIN=740576d7066e962dae60212370a5965225c20613
PATCHES=$R/patches/m740576d70_20261011
add() {
  local tree=$1
  shift
  git -C /workspace/sglang-omni worktree add -q --detach "/workspace/$tree" "$MAIN"
  for patch in "$@"; do
    git -C "/workspace/$tree" apply -3 "$PATCHES/$patch.patch"
    git -C "/workspace/$tree" add -A
  done
  git -C "/workspace/$tree" reset -q
  echo "$tree on $MAIN with $* changed=$(git -C "/workspace/$tree" status --short | wc -l)" \
    | tee "/workspace/$tree/BRANCH_MARKER.txt" >> $R/queue_i2.txt
  cp -r $R/inductor_t_i1 $R/inductor_$tree
}
add t_i2a i1_float32_speech_logits pr2626_ras
add t_i2b i1_float32_speech_logits pr2626_ras i2b_cosyvoice_sampling_arm
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1 c16r:16:::1 L16:16:::1:$R/longform/meta.lst"
export SCORE_CELLS="c16"
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
mkdir -p $R/i2
echo "i2a $(date +%T) card0=t_i2a card1=t_i2b" >> $R/queue_i2.txt
env $PAIR bash $IN/ab_pair.sh $R/i2/a t_i2a t_i2b 39000
echo "i2b $(date +%T) card0=t_i2b card1=t_i2a" >> $R/queue_i2.txt
env $PAIR bash $IN/ab_pair.sh $R/i2/b t_i2b t_i2a 39000
echo "queue done $(date +%T)" >> $R/queue_i2.txt
