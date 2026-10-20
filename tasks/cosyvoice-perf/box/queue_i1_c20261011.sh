#!/usr/bin/env bash
# Item 1 of ARCH_AUDIT_20261011.md section R: the speech head writes float32 logits (branch
# fix/cosyvoice3-float32-speech-logits 1d22068f8 on main 740576d70). Trees t_m2 (main) and t_i1 (main with
# i1_float32_speech_logits.patch), the Fun-CosyVoice3 unit suite on t_i1 (GPU lane on card 0), then main
# against item 1 in both card orders: seeded stream c1, stream c16 with WER and SIM, stream c64, long form
# c16.
# usage (in the container): bash queue_i1_c20261011.sh
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
    | tee "/workspace/$tree/BRANCH_MARKER.txt" >> $R/queue_i1.txt
}
git -C /workspace/sglang-omni fetch -q upstream "$MAIN"
add t_m2
add t_i1 i1_float32_speech_logits
mkdir -p $R/i1
(cd /workspace/t_i1 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_i1 \
  python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider) > $R/i1/units.log 2>&1
echo "units rc $? $(date +%T) $(tail -1 $R/i1/units.log)" >> $R/queue_i1.txt

export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1 c64:64:::1 L16:16:::1:$R/longform/meta.lst"
export SCORE_CELLS="c16"
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
echo "i1a $(date +%T) card0=t_m2 card1=t_i1" >> $R/queue_i1.txt
env $PAIR bash $IN/ab_pair.sh $R/i1/a t_m2 t_i1 37000
echo "i1b $(date +%T) card0=t_i1 card1=t_m2" >> $R/queue_i1.txt
env $PAIR bash $IN/ab_pair.sh $R/i1/b t_i1 t_m2 37000
echo "queue done $(date +%T)" >> $R/queue_i1.txt
