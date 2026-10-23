#!/usr/bin/env bash
# #3102 with the first-chunk reach computed once per emit count (check/d1-reach-once): the per-plan module walk cost
# 52 us a call and showed in c64's dispatch-to-launch gap (+0.26 and +0.58 ms at the median). qwen3_tts units (a
# failure runs nothing else), reach_call_bench.py, then SeedTTS EN c64 against main in both card orders with request
# events and each arm's first-audio chain.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
CHAIN=$T/tasks/qwen3-tts-perf/box/profile_20261022/first_audio_chain.py
M=${M:?the main tree}
PC=${PC:?the check tree}
L=$D/queue.log
echo "queue d1 reach once start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) check $(git -C $PC rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $PC && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$PC timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider -rfEs \
  > $D/units/d1_reach_once.log 2>&1)
UNITS=$?
echo "units d1 reach once rc $UNITS $(grep -E 'passed|failed' $D/units/d1_reach_once.log | tail -n 1) $(date +%T)" >> $L
if [ $UNITS -ne 0 ]; then
  echo "stopped: d1 reach once units failed" >> $L
  echo "queue_d1_reach_once_done $(date +%T)" >> $L
  exit 1
else
  :
fi
(cd $PC && PYTHONPATH=$PC python3 $HERE/reach_call_bench.py --card 3 > $D/reach_call_bench_once.txt 2>&1)
echo "reach call bench once rc $? $(grep 'a call' $D/reach_call_bench_once.txt | tr '\n' ' ') $(date +%T)" >> $L
for run in "p1r_c64_a 2 3 23000" "p1r_c64_b 3 2 24200"; do
  set -- $run
  python3 $IN/pair.py --label $1 --tree $M --b-tree $PC --concurrency 64 --a-card $2 --b-card $3 --port $4 --events \
    > $D/$1.out 2>&1
  rc=$?
  failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log)
  failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)
  echo "$1 rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b" >> $L
  python3 $CHAIN $PAIR_ROOT/$1/a/events $PAIR_ROOT/$1/b/events > $D/chain_$1.txt 2>&1
done
python3 $IN/pair_table.py p1r_c64_a p1r_c64_b > $D/table_p1r.txt 2>&1
python3 $READ/paired_by_prompt.py p1r_c64_a p1r_c64_b > $D/paired_p1r.txt 2>&1
echo "queue_d1_reach_once_done $(date +%T)" >> $L
