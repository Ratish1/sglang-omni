#!/usr/bin/env bash
# Component accuracy against CosyVoice (acc_reference_c20261011.py): main (t_m2, card 0) and item 1 (t_i1,
# card 1) serve a seeded stream c1 cell of 16 SeedTTS English samples with acc_probe/sitecustomize.py
# recording each request's AR inputs, raw logits per step and the vocoder's mel and samples per step, then
# each arm's records are replayed through CosyVoice's modules on its card. Starts after the i1r queue.
# usage (in the container): bash queue_acc_c20261011.sh
set -u
R=/data/c20261011
IN=$R/in
until grep -q "queue done" $R/queue_i1r.txt 2>/dev/null; do sleep 20; done
export ROOT=$R
export CELLS="acc:1:16:1234:1"
export SCORE_CELLS="none"
mkdir -p $R/acc/m2 $R/acc/i1
echo "acc serve $(date +%T)" >> $R/queue_acc.txt
env CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95 \
  EXTRA_ENV_0="PYTHONPATH=$IN/acc_probe:/workspace/t_m2 ACC_OUT=$R/acc/m2" \
  EXTRA_ENV_1="PYTHONPATH=$IN/acc_probe:/workspace/t_i1 ACC_OUT=$R/acc/i1" \
  bash $IN/ab_pair.sh $R/acc/serve t_m2 t_i1 37000
echo "acc records m2 $(ls $R/acc/m2 | wc -l) i1 $(ls $R/acc/i1 | wc -l) $(date +%T)" >> $R/queue_acc.txt
(cd /workspace/t_m2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_m2 python3 -u $IN/acc_reference_c20261011.py \
  --records $R/acc/m2 --out $R/acc/report_m2.json) > $R/acc/reference_m2.log 2>&1 &
M2=$!
(cd /workspace/t_i1 && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/t_i1 python3 -u $IN/acc_reference_c20261011.py \
  --records $R/acc/i1 --out $R/acc/report_i1.json) > $R/acc/reference_i1.log 2>&1 &
I1=$!
wait $M2
echo "reference m2 rc $? $(date +%T)" >> $R/queue_acc.txt
wait $I1
echo "reference i1 rc $? $(date +%T)" >> $R/queue_acc.txt
echo "queue done $(date +%T)" >> $R/queue_acc.txt
