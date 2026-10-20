#!/usr/bin/env bash
# #2625 on H100 (its author measured Blackwell only), chained on item 1: t_i1 (main with float32 logits)
# against t_i1s (t_i1 with pr2625_staged_token_read), both card orders: seeded stream c1, stream c16 with WER
# and SIM, stream c64, long form c16. The sync census (queue_sync2) showed #2625 removes all three of the AR's
# per step .tolist() reads (5,456 each on main over 5,456 steps).
# usage (in the container): bash queue_s1_c20261011.sh
set -u
R=/data/c20261011
IN=$R/in
MAIN=740576d7066e962dae60212370a5965225c20613
PATCHES=$R/patches/m740576d70_20261011
git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_i1s "$MAIN"
for patch in i1_float32_speech_logits pr2625_staged_token_read; do
  git -C /workspace/t_i1s apply -3 "$PATCHES/$patch.patch"
  git -C /workspace/t_i1s add -A
done
git -C /workspace/t_i1s reset -q
echo "t_i1s on $MAIN with i1_float32_speech_logits pr2625_staged_token_read changed=$(git -C /workspace/t_i1s status --short | wc -l)" \
  | tee /workspace/t_i1s/BRANCH_MARKER.txt >> $R/queue_s1.txt
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1 c64:64:::1 L16:16:::1:$R/longform/meta.lst"
export SCORE_CELLS="c16"
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
mkdir -p $R/s1
echo "s1a $(date +%T) card0=t_i1 card1=t_i1s" >> $R/queue_s1.txt
env $PAIR bash $IN/ab_pair.sh $R/s1/a t_i1 t_i1s 38000
echo "s1b $(date +%T) card0=t_i1s card1=t_i1" >> $R/queue_s1.txt
env $PAIR bash $IN/ab_pair.sh $R/s1/b t_i1s t_i1 38000
echo "queue done $(date +%T)" >> $R/queue_s1.txt
