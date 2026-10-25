#!/usr/bin/env bash
# After queue_kvip_seeded_b.sh: scored streaming pairs at 128 concurrent requests in both card orders on the final
# head, so WER and SIM are read at a higher concurrency than c64. 128 streams outnumber the 64 codec slots, so the
# streams past them take the left-context decoder in both arms, which also exercises slot exhaustion beside the rings.
set -u
D=/data/kvip1025
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/chain_20261022
M=${M:?the main tree}
P=${P:?the final tree}
L=$D/queue.log
until grep -q queue_kvip_seeded_b_done $L; do
  sleep 20
done
echo "queue kvip c128 start $(date +%T) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
for order in "a 2 3 7000" "b 3 2 8200"; do
  set -- $order
  python3 $K/v3_20261020/pair.py --label kvf_c128_$1 --tree $M --b-tree $P --concurrency 128 --a-card $2 --b-card $3 \
    --port $4 > $D/kvf_c128_$1.out 2>&1
  echo "kvf_c128_$1 rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/kvf_c128_$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/kvf_c128_$1/b/serve.log)" >> $L
  if [ -f $PAIR_ROOT/kvf_c128_$1/a/bench/similarity_results.json ] && [ -f $PAIR_ROOT/kvf_c128_$1/b/bench/similarity_results.json ]; then
    :
  else
    python3 $HERE/similarity_excluding_caps.py kvf_c128_$1 --tree $M --card 2 > $D/similarity_kvf_c128_$1.out 2>&1
  fi
done
python3 $K/v3_20261020/pair_table.py kvf_c128_a kvf_c128_b > $D/table_kvf_c128.txt 2>&1
python3 $K/q3t_20261022/paired_by_prompt.py kvf_c128_a kvf_c128_b > $D/paired_kvf_c128.txt 2>&1
(cd $M && PYTHONPATH=$M python3 $K/d1merge_20261024/ci_metrics.py kvf_c128_a kvf_c128_b > $D/ci_metrics_kvf_c128.txt 2>&1)
echo "queue_kvip_c128_done $(date +%T)" >> $L
