#!/usr/bin/env bash
# After queue_aa_c16.sh: D8's c64 first audio after the opening wave read p99 +6.5 and +6.8 ms in both card orders,
# inside this node's same-code band (-7.9 to +3.4 ms) but one-signed, and D8's speaker mel now runs on a saturated
# device where main ran it on the CPU. The discriminating observable is each request's preprocessing time under c64
# load: main against D8 at c64 in both orders with request events, unscored, then each pair's first-audio chain.
set -u
D=/data/next1024
export PAIR_ROOT=$D/pairs
K=/workspace/t_marks_2f039db26/tasks/qwen3-tts-perf/box
CHAIN=$K/profile_20261022/first_audio_chain.py
M=/workspace/main_67476823a
P8=/workspace/d8_ccdcf38c5
L=$D/queue.log
until grep -q queue_aa_c16_done $L; do sleep 30; done
echo "queue d8 c64 events start $(date +%T)" >> $L
for spec in "d8ev_c64_a 2 3 2000" "d8ev_c64_b 3 2 3200"; do
  set -- $spec
  python3 $K/v3_20261020/pair.py --label $1 --tree $M --b-tree $P8 --concurrency 64 --a-card $2 --b-card $3 --port $4 \
    --events --skip-score > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T)" >> $L
  python3 $CHAIN $PAIR_ROOT/$1/a/events $PAIR_ROOT/$1/b/events > $D/chain_$1.txt 2>&1
done
python3 $K/v3_20261020/pair_table.py d8ev_c64_a d8ev_c64_b > $D/table_d8ev_c64.txt 2>&1
python3 $K/q3t_20261022/first_audio_tail.py d8ev_c64_a d8ev_c64_b > $D/first_audio_tail_d8ev.txt 2>&1
echo "queue_d8_c64_events_done $(date +%T)" >> $L
