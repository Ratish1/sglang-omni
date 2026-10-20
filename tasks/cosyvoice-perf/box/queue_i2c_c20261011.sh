#!/usr/bin/env bash
# Item 2, arm C: #2626 without omni's 1.21 repetition penalty (i2c_no_penalty_arm: default 1.0), the rest of
# arm A kept (temperature 0.7, top k 20, top p 0.8, the masked redraw). Is the penalty still needed once the
# redraw handles runaways, and is it what shortens the speech 5 to 17 % against CosyVoice's sampling (i2)?
# t_i2a against t_i2c, both card orders, the i2 cells. Starts after queue_i2.
# usage (in the container): bash queue_i2c_c20261011.sh
set -u
R=/data/c20261011
IN=$R/in
MAIN=740576d7066e962dae60212370a5965225c20613
PATCHES=$R/patches/m740576d70_20261011
until grep -q "queue done" $R/queue_i2.txt 2>/dev/null; do sleep 20; done
git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_i2c "$MAIN"
for patch in i1_float32_speech_logits pr2626_ras i2c_no_penalty_arm; do
  git -C /workspace/t_i2c apply -3 "$PATCHES/$patch.patch"
  git -C /workspace/t_i2c add -A
done
git -C /workspace/t_i2c reset -q
echo "t_i2c on $MAIN with i1_float32_speech_logits pr2626_ras i2c_no_penalty_arm changed=$(git -C /workspace/t_i2c status --short | wc -l)" \
  | tee /workspace/t_i2c/BRANCH_MARKER.txt >> $R/queue_i2c.txt
cp -r $R/inductor_t_i1 $R/inductor_t_i2c
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1 c16r:16:::1 L16:16:::1:$R/longform/meta.lst"
export SCORE_CELLS="c16"
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
echo "i2c a $(date +%T) card0=t_i2a card1=t_i2c" >> $R/queue_i2c.txt
env $PAIR bash $IN/ab_pair.sh $R/i2c/a t_i2a t_i2c 39000
echo "i2c b $(date +%T) card0=t_i2c card1=t_i2a" >> $R/queue_i2c.txt
env $PAIR bash $IN/ab_pair.sh $R/i2c/b t_i2c t_i2a 39000
echo "queue done $(date +%T)" >> $R/queue_i2c.txt
