#!/usr/bin/env bash
# After queue_kvip_tail.sh: the PR head's own pairs against main ab8193645, so the table a reviewer reads is its
# code. e6cbe020a adds to the measured 588200a57 only the ring sized to the widest cold decode (default serving
# unchanged) and the spread of a wide decode's discarded K/V over the trash slot's rows (outputs unchanged). Seeded
# c1 streaming, c16 and c64 both card orders, scored; L3 c16 both orders with request events. Non-streaming is not
# rerun: it never reaches the incremental decoder (seeded non-streaming byte identical on 588200a57).
set -u
D=/data/kvip1025
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/chain_20261022
M=${M:?the main tree}
P=${P:?the PR head tree}
LONG=/data/next1025/long
L=$D/queue.log
until grep -q queue_kvip_tail_done $L; do
  sleep 20
done
echo "queue kvip final start $(date +%T) main $(git -C $M rev-parse HEAD) head $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
run_pair() {
  local label=$1
  shift
  python3 $K/v3_20261020/pair.py --label $label --tree $M --b-tree $P "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
}
run_pair kvf_s_c1 --concurrency 1 --a-card 2 --b-card 3 --port 4400 --seed 1234
run_pair kvf_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 7000
run_pair kvf_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 8200
run_pair kvf_c64_a --concurrency 64 --a-card 2 --b-card 3 --port 10000
run_pair kvf_c64_b --concurrency 64 --a-card 3 --b-card 2 --port 11200
run_pair kvf_L3_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 14800 --meta $LONG/L3_c16_run.lst --events --skip-score
run_pair kvf_L3_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 17000 --meta $LONG/L3_c16_run.lst --events --skip-score
for label in kvf_s_c1 kvf_c16_a kvf_c16_b kvf_c64_a kvf_c64_b; do
  if [ -f $PAIR_ROOT/$label/a/bench/similarity_results.json ] && [ -f $PAIR_ROOT/$label/b/bench/similarity_results.json ]; then
    :
  else
    python3 $HERE/similarity_excluding_caps.py $label --tree $M --card 2 > $D/similarity_$label.out 2>&1
    echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
  fi
done
LABELS="kvf_s_c1 kvf_c16_a kvf_c16_b kvf_c64_a kvf_c64_b kvf_L3_c16_a kvf_L3_c16_b"
python3 $K/v3_20261020/pair_table.py $LABELS > $D/table_kvf.txt 2>&1
python3 $K/q3t_20261022/paired_by_prompt.py $LABELS > $D/paired_kvf.txt 2>&1
python3 $K/q3t_20261022/first_audio_tail.py kvf_c16_a kvf_c16_b kvf_c64_a kvf_c64_b > $D/first_audio_tail_kvf.txt 2>&1
(cd $M && PYTHONPATH=$M python3 $K/d1merge_20261024/ci_metrics.py kvf_s_c1 kvf_c16_a kvf_c16_b kvf_c64_a kvf_c64_b \
  > $D/ci_metrics_kvf.txt 2>&1)
echo "queue_kvip_final_done $(date +%T)" >> $L
