#!/usr/bin/env bash
# After queue_d1.sh: D8 measured again on main plus D1, the stage its L1 c16 regression sat in (a long reference's first
# vocoder decode) now decoding from its reach. D8's two commits on D1's branch against D1, L1 c16 and L3 c16 in both
# card orders with request events and each pair's first-audio chain; D8's SeedTTS EN, cap and non-streaming readings
# stand (D8 and D1 share no file). Runs only when D1's units and gates passed.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
CHAIN=$T/tasks/qwen3-tts-perf/box/profile_20261022/first_audio_chain.py
LONG=/data/prof_20261022/formal/meta_w2/long
P1=${P1:?the D1 tree}
P81=${P81:?the D8 on D1 tree}
L=$D/queue.log
until grep -q queue_d1_done $L; do sleep 20; done
if sed -n '/queue d1 start/,$p' $L | grep -q "^stopped"; then
  echo "queue d8 on d1 skipped: the d1 queue stopped" >> $L
  echo "queue_d8_on_d1_done $(date +%T)" >> $L
  exit 1
else
  :
fi
echo "queue d8 on d1 start $(date +%T) d1 $(git -C $P1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d8 on d1 $(git -C $P81 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
for run in "d8d1_L1_c16_a L1 2 3 19400" "d8d1_L1_c16_b L1 3 2 20600" "d8d1_L3_c16_a L3 2 3 21800" "d8d1_L3_c16_b L3 3 2 23000"; do
  set -- $run
  python3 $IN/pair.py --label $1 --tree $P1 --b-tree $P81 --concurrency 16 --a-card $3 --b-card $4 --port $5 \
    --meta $LONG/$2_c16_run.lst --events > $D/$1.out 2>&1
  rc=$?
  failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log)
  failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)
  echo "$1 rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b" >> $L
  python3 $CHAIN $PAIR_ROOT/$1/a/events $PAIR_ROOT/$1/b/events > $D/chain_$1.txt 2>&1
done
LABELS="d8d1_L1_c16_a d8d1_L1_c16_b d8d1_L3_c16_a d8d1_L3_c16_b"
python3 $IN/pair_table.py $LABELS > $D/table_d8d1.txt 2>&1
python3 $READ/paired_by_prompt.py $LABELS > $D/paired_d8d1.txt 2>&1
echo "queue_d8_on_d1_done $(date +%T)" >> $L
