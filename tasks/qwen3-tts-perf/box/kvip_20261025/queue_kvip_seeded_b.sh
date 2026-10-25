#!/usr/bin/env bash
# After queue_kvip_rerun.sh: the seeded c1 pair in the other card order (rings on card 2), so a vocoder-only change's
# quality is read with the codes held equal in both orders (DECISIONS 68); kv_s_c1 and kvf_s_c1 both ran rings on
# card 3. Then the CI quality metrics paired per prompt for the three seeded pairs.
set -u
D=/data/kvip1025
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/chain_20261022
M=${M:?the main tree}
P=${P:?the final tree}
L=$D/queue.log
until grep -q queue_kvip_rerun_done $L; do
  sleep 20
done
echo "queue kvip seeded b start $(date +%T) main $(git -C $M rev-parse HEAD) final $(git -C $P rev-parse HEAD)" >> $L
python3 $K/v3_20261020/pair.py --label kvf_s_c1_b --tree $M --b-tree $P --concurrency 1 --a-card 3 --b-card 2 \
  --port 4400 --seed 1234 > $D/kvf_s_c1_b.out 2>&1
echo "kvf_s_c1_b rc $? $(date +%T) $(grep 'byte identical' $D/kvf_s_c1_b.out)" >> $L
if [ -f $PAIR_ROOT/kvf_s_c1_b/a/bench/similarity_results.json ] && [ -f $PAIR_ROOT/kvf_s_c1_b/b/bench/similarity_results.json ]; then
  :
else
  python3 $HERE/similarity_excluding_caps.py kvf_s_c1_b --tree $M --card 2 > $D/similarity_kvf_s_c1_b.out 2>&1
fi
python3 $K/v3_20261020/pair_table.py kv_s_c1 kvf_s_c1 kvf_s_c1_b > $D/table_seeded.txt 2>&1
(cd $M && PYTHONPATH=$M python3 $K/d1merge_20261024/ci_metrics.py kv_s_c1 kvf_s_c1 kvf_s_c1_b > $D/ci_metrics_seeded.txt 2>&1)
echo "queue_kvip_seeded_b_done $(date +%T)" >> $L
