#!/usr/bin/env bash
# After queue_d1_mid.sh: #3102 read +0.9 % and +0.8 % RTF at SeedTTS EN c64 (paired latency +5.6 and +12.7 ms) where
# its references never reach the cap; the planner's per-plan reach call is the suspect. reach_call_bench.py times the
# call on the trained decoder; then c64 in both card orders again with request events, and each arm's first-audio chain.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
CHAIN=$T/tasks/qwen3-tts-perf/box/profile_20261022/first_audio_chain.py
M=${M:?the main tree}
P1=${P1:?the D1 tree}
L=$D/queue.log
until grep -q queue_d1_mid_done $L; do sleep 20; done
echo "queue d1 c64 events start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $P1 && PYTHONPATH=$P1 python3 $HERE/reach_call_bench.py --card 3 > $D/reach_call_bench.txt 2>&1)
echo "reach call bench rc $? $(grep 'a call' $D/reach_call_bench.txt | tr '\n' ' ') $(date +%T)" >> $L
for run in "p1e_c64_a 2 3 20600" "p1e_c64_b 3 2 21800"; do
  set -- $run
  python3 $IN/pair.py --label $1 --tree $M --b-tree $P1 --concurrency 64 --a-card $2 --b-card $3 --port $4 --events \
    > $D/$1.out 2>&1
  rc=$?
  failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log)
  failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)
  echo "$1 rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b" >> $L
  python3 $CHAIN $PAIR_ROOT/$1/a/events $PAIR_ROOT/$1/b/events > $D/chain_$1.txt 2>&1
done
python3 $IN/pair_table.py p1e_c64_a p1e_c64_b > $D/table_p1e.txt 2>&1
python3 $READ/paired_by_prompt.py p1e_c64_a p1e_c64_b > $D/paired_p1e.txt 2>&1
echo "queue_d1_c64_events_done $(date +%T)" >> $L
