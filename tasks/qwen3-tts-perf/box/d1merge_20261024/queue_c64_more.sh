#!/usr/bin/env bash
# After queue_refresh.sh: #3102's SeedTTS EN c64 pairs read paired latency +7.35 and +7.15 ms in both card orders
# (RTF +0.7 and +0.3 %), against a c64 A/A floor of +8.85 ms from another lease. Two more pairs main against #3102 and
# main against main in both card orders on this lease, streaming, unseeded, no scoring; then the paired latencies.
set -u
D=/data/d1merge
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
M=${M:?the main tree}
P1=${P1:?the 3102 tree}
L=$D/queue.log
until grep -q queue_refresh_done $L; do sleep 30; done
echo "queue c64 more start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
run_pair() {
  local label=$1 b_tree=$2
  shift 2
  python3 $IN/pair.py --label $label --tree $M --b-tree $b_tree --concurrency 64 --skip-score "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b" >> $L
}
run_pair rp_c64_c $P1 --a-card 2 --b-card 3 --port 4400
run_pair rp_c64_d $P1 --a-card 3 --b-card 2 --port 7000
run_pair aa_c64_a $M --a-card 2 --b-card 3 --port 8200
run_pair aa_c64_b $M --a-card 3 --b-card 2 --port 10000
LABELS="rp_c64_a rp_c64_b rp_c64_c rp_c64_d aa_c64_a aa_c64_b"
for label in $LABELS; do
  python3 $IN/pair_table.py $label >> $D/table_c64_more.txt 2>&1
done
python3 $READ/paired_by_prompt.py $LABELS > $D/paired_c64_more.txt 2>&1
echo "queue_c64_more_done $(date +%T)" >> $L
