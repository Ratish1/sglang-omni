#!/usr/bin/env bash
# A's race check: main (t_m, card 0) and A (t_a, card 1) at the same time, both with race_probe first on the
# server's PYTHONPATH (RACE_PROBE=1): every prefill batch's input rows compared bit for bit against rows rebuilt from
# the host ids, every new CUDA stream logged. Cells c16 and c64 over the full corpus, no scoring. Ports below the
# ephemeral range.
# usage (in the container): bash queue_race_c20261013.sh
set -u
R=/data/c20261013
IN=$R/in
OUT=$R/race
PROBE=$IN/race_probe
mkdir -p $OUT
echo "start $(date +%T)" >> $OUT/queue.txt
export ROOT=$R
export SCORE_CELLS="none"
export CELLS="c16:16:::1 c64:64:::1"
env CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95 \
  EXTRA_ENV_0="PYTHONPATH=$PROBE:/workspace/t_m RACE_PROBE=1" \
  EXTRA_ENV_1="PYTHONPATH=$PROBE:/workspace/t_a RACE_PROBE=1" \
  bash $IN/ab_pair_sync.sh $OUT t_m t_a 24000
for arm in $OUT/card0_t_m $OUT/card1_t_a; do
  echo "$arm on=$(grep -c 'race_probe on' $arm/serve.log) patched=$(grep -c 'race_probe runner patched' $arm/serve.log)" \
    "batches=$(grep -c 'race_probe batch' $arm/serve.log) mismatches=$(grep -c 'race_probe MISMATCH' $arm/serve.log)" \
    "last: $(grep 'race_probe batch' $arm/serve.log | tail -1)" >> $OUT/queue.txt
done
echo "queue done $(date +%T)" >> $OUT/queue.txt
