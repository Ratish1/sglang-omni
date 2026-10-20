#!/usr/bin/env bash
# V4 (perf/qwen3-tts-flat-codec-arena a6214172d, stacked on V5 a962e3db6: each codec arena slot one flat row, one
# gather and one scatter per decode) against V5: the touched units, a seeded c1 pair that must be byte identical (data
# movement only), then c16 and c64 pairs in both card orders with each arm's SM Issue, full corpus, WER and SIM. Waits
# for queue_v5.sh.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
V5=/workspace/v5_a962e3db6
V4=/workspace/v4_a6214172d
L=$D/queue.log
until grep -q queue_v5_done $L; do sleep 30; done
echo "queue v4 start $(date +%T) v5 $(git -C $V5 rev-parse HEAD:sglang_omni) v4 $(git -C $V4 rev-parse HEAD:sglang_omni)" >> $L
(cd $V4 && PYTHONPATH=$V4 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider -rfE \
  > $D/units/v4.log 2>&1)
echo "units v4 rc $? $(date +%T)" >> $L
run_pair() {
  python3 $IN/pair.py --label $1 --tree $V5 --b-tree $V4 --concurrency $2 --a-card $3 --b-card $4 --port $5 ${6:-} \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) $(grep -h 'byte identical' $D/$1.out) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $D/pairs/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $D/pairs/$1/b/serve.log)" >> $L
}
run_pair v4_c1_seed 1 0 1 14000 "--seed 1234"
run_pair v4_c16_a 16 0 1 17000 --gpu-metrics
run_pair v4_c16_b 16 1 0 19000 --gpu-metrics
run_pair v4_c64_a 64 0 1 21000 --gpu-metrics
run_pair v4_c64_b 64 1 0 27000 --gpu-metrics
python3 $IN/pair_table.py v4_c1_seed v4_c16_a v4_c16_b v4_c64_a v4_c64_b > $D/table_v4.txt 2>&1
echo "queue_v4_done $(date +%T)" >> $L
