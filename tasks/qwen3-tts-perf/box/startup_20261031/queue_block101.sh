#!/usr/bin/env bash
# Block 101 of 11-02: the serving stress harness on Qwen3-TTS, baseline against an arm at the same time. Long streaming
# requests (4,096 characters, 2,048 frames) at concurrency 64 in both card orders and concurrency 1, then the mixed
# production schedule once. First read: does each path serve this load, and where does it stand against SeedTTS.
# usage: D=... T=<kit tree> BASE=<tree> ARM=<tree> HARNESS=<tree> [BLOCK=block101] [CARD_A=2] [CARD_B=3] [PORT=26000] bash queue_block101.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the base tree}
ARM=${ARM:?the tree under test}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block101}
CARD_A=${CARD_A:-2}
CARD_B=${CARD_B:-3}
PORT=${PORT:-26000}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
EX=$HARNESS/benchmarks/tts_serving/examples
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in BASE ARM HARNESS; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/block96/cache/inductor TRITON_CACHE_DIR=$D/block96/cache/triton
one() {
  local label=$1 tree=$2 spec=$3 card=$4 port=$5 barrier=$6
  (cd $K && timeout 3600 python3 stress_serving_run.py --label $label --serve-tree $tree --harness-tree $HARNESS --spec $spec \
    --root $W --card $card --port $port --barrier $barrier > $W/$label.txt 2>&1)
  echo "$label card $card exit $? $(date +%T)" >> $L
  grep -E "^[a-z_0-9-]+: [0-9]+ of [0-9]+ ok|overall passed|harness exit|Tracebacks" $W/$label.txt | cut -c1-210 | sed "s/^/  $label /" >> $L
}
port=$PORT
pair() {
  local name=$1 spec=$2 first=$3 second=$4
  one base_$name $BASE $spec $first $port $W/barrier_$name &
  one arm_$name $ARM $spec $second $((port + 400)) $W/barrier_$name &
  wait
  port=$((port + 800))
}
pair long_c64_o0 $EX/stress_qwen3_tts_long_c64.json $CARD_A $CARD_B
pair long_c64_o1 $EX/stress_qwen3_tts_long_c64.json $CARD_B $CARD_A
pair long_c1_o0 $EX/stress_qwen3_tts_long_c1.json $CARD_A $CARD_B
pair mixed_o0 $EX/stress_qwen3_tts.json $CARD_A $CARD_B
echo "queue $BLOCK done $(date +%T)" >> $L
