#!/usr/bin/env bash
# V5 (perf/qwen3-tts-codebook-tables a962e3db6, stacked on V3 6f6039ba6: the vocoder's codebook tables computed once
# at load) against V3: the touched units, a seeded c1 pair that must be byte identical (no arithmetic changes), then
# c16 and c64 pairs in both card orders with each arm's SM Issue, full corpus, WER and SIM. Waits for queue_v3.sh.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
V3=/workspace/v3_6f6039ba6
V5=/workspace/v5_a962e3db6
L=$D/queue.log
until grep -q queue_v3_done $L; do sleep 30; done
echo "queue v5 start $(date +%T) v3 $(git -C $V3 rev-parse HEAD:sglang_omni) v5 $(git -C $V5 rev-parse HEAD:sglang_omni)" >> $L
(cd $V5 && PYTHONPATH=$V5 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider -rfE \
  > $D/units/v5.log 2>&1)
echo "units v5 rc $? $(date +%T)" >> $L
run_pair() {
  python3 $IN/pair.py --label $1 --tree $V3 --b-tree $V5 --concurrency $2 --a-card $3 --b-card $4 --port $5 ${6:-} \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) $(grep -h 'byte identical' $D/$1.out) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $D/pairs/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $D/pairs/$1/b/serve.log)" >> $L
}
run_pair v5_c1_seed 1 0 1 7000 "--seed 1234"
run_pair v5_c16_a 16 0 1 9000 --gpu-metrics
run_pair v5_c16_b 16 1 0 11000 --gpu-metrics
run_pair v5_c64_a 64 0 1 22000 --gpu-metrics
run_pair v5_c64_b 64 1 0 30000 --gpu-metrics
python3 $IN/pair_table.py v5_c1_seed v5_c16_a v5_c16_b v5_c64_a v5_c64_b > $D/table_v5.txt 2>&1
echo "queue_v5_done $(date +%T)" >> $L
