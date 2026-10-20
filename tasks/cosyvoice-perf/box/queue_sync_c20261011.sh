#!/usr/bin/env bash
# The legacy stream waits that remain, for the stream design (ARCH_AUDIT 4.8): the v2 probe of which waits on
# the legacy default stream throttle another thread's launches (card 0, idle), then a sync census in serving
# (sync_census/sitecustomize.py) of main (t_m2, card 0) and main with #2625 (t_p2625, card 1): every
# synchronizing CUDA call by thread, call site and stream, over a seeded stream c1 cell of 16 samples and a
# stream c16 cell of 256. Starts after the accuracy queue.
# usage (in the container): bash queue_sync_c20261011.sh
set -u
R=/data/c20261011
IN=$R/in
MAIN=740576d7066e962dae60212370a5965225c20613
until grep -q "queue done" $R/queue_acc.txt 2>/dev/null; do sleep 20; done
git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_p2625 "$MAIN"
git -C /workspace/t_p2625 apply -3 $R/patches/m740576d70_20261011/pr2625_staged_token_read.patch
git -C /workspace/t_p2625 add -A
git -C /workspace/t_p2625 reset -q
echo "t_p2625 on $MAIN with pr2625_staged_token_read changed=$(git -C /workspace/t_p2625 status --short | wc -l)" \
  | tee /workspace/t_p2625/BRANCH_MARKER.txt >> $R/queue_sync.txt
mkdir -p $R/sync
CUDA_VISIBLE_DEVICES=0 python3 -u $IN/legacy_stream_probe_v2_c20261011.py > $R/sync/probe_v2.txt 2>&1
echo "probe v2 rc $? $(date +%T)" >> $R/queue_sync.txt
export ROOT=$R
export CELLS="s1:1:16:1234:1 s16:16:256::1"
export SCORE_CELLS="none"
env CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95 \
  EXTRA_ENV_0="PYTHONPATH=$IN/sync_census:/workspace/t_m2 CENSUS_OUT=$R/sync/census_m2 CENSUS_PORT=37000" \
  EXTRA_ENV_1="PYTHONPATH=$IN/sync_census:/workspace/t_p2625 CENSUS_OUT=$R/sync/census_p2625 CENSUS_PORT=37200" \
  bash $IN/ab_pair.sh $R/sync/serve t_m2 t_p2625 37000
echo "queue done $(date +%T)" >> $R/queue_sync.txt
