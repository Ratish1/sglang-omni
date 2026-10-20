#!/usr/bin/env bash
# Sync census v2 (sync_census_v2/sitecustomize.py: call sites past SGLang's showwarning wrapper) of main
# (t_m2, card 0) and main with #2625 (t_p2625, card 1), seeded stream c1 of 16 and stream c16 of 256, after
# the speech tokenizer determinism check (tok_determinism_c20261011.py, card 0). Starts after queue_sync.
# usage (in the container): bash queue_sync2_c20261011.sh
set -u
R=/data/c20261011
IN=$R/in
until grep -q "queue done" $R/queue_sync.txt 2>/dev/null; do sleep 20; done
(cd /workspace/t_m2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_m2 python3 -u $IN/tok_determinism_c20261011.py) \
  > $R/sync/tok_determinism.txt 2>&1
echo "tok determinism rc $? $(date +%T)" >> $R/queue_sync2.txt
export ROOT=$R
export CELLS="s1:1:16:1234:1 s16:16:256::1"
export SCORE_CELLS="none"
env CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95 \
  EXTRA_ENV_0="PYTHONPATH=$IN/sync_census_v2:/workspace/t_m2 CENSUS_OUT=$R/sync/census2_m2 CENSUS_PORT=37000" \
  EXTRA_ENV_1="PYTHONPATH=$IN/sync_census_v2:/workspace/t_p2625 CENSUS_OUT=$R/sync/census2_p2625 CENSUS_PORT=37200" \
  bash $IN/ab_pair.sh $R/sync/serve2 t_m2 t_p2625 37000
echo "queue done $(date +%T)" >> $R/queue_sync2.txt
