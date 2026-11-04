#!/usr/bin/env bash
# Block 124 of 11-04: the head against main on 0.5.23, the PR's tables. Non-streaming c64 and c16, streaming c64 and first
# chunks beside long requests, all both orders. Seeded c1 streaming must stay byte identical; non-streaming gets WER and SIM on
# the same codes, 1.7B and 0.6B, since a whole request's decode differs from main's on both. 0.6B non-streaming c64 and c16.
# usage: D=... T=<kit tree> MAIN=<tree> ARM=<tree> HARNESS=<tree> [BLOCK=block124] [PORT=2050] bash queue_block124.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
ARM=${ARM:?the head}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block124}
PORT=${PORT:-2050}
K=$T/tasks/qwen3-tts-perf/box
S=$K/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in MAIN ARM; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
port=$PORT
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5 concurrency=$6 mode=$7 model=$8
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency $concurrency $mode --model $model --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency p95|TTFC mean|TTFC p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-210); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null) left context $(grep -c left-context $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
pair() {
  local name=$1 concurrency=$2 mode=$3 model=$4
  run_arm main_${name}_o0 $MAIN 0 $port $W/barrier_${name}_o0 $concurrency "$mode" $model &
  run_arm head_${name}_o0 $ARM 1 $((port + 400)) $W/barrier_${name}_o0 $concurrency "$mode" $model &
  wait
  port=$((port + 800))
  run_arm main_${name}_o1 $MAIN 1 $port $W/barrier_${name}_o1 $concurrency "$mode" $model &
  run_arm head_${name}_o1 $ARM 0 $((port + 400)) $W/barrier_${name}_o1 $concurrency "$mode" $model &
  wait
  port=$((port + 800))
}
LARGE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
SMALL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
pair n64 64 --no-stream $LARGE
pair n16 16 --no-stream $LARGE
pair s64 64 "" $LARGE
(cd $S && timeout 1500 python3 mixed_long.py --a main=$MAIN --b head=$ARM --harness $HARNESS --root $W/mixed --port $port > $W/mixed.txt 2>&1)
echo "mixed rc $? $(date +%T)" >> $L
grep -E ' card [01]: stream done|^  [a-z]+: [0-9]+ of |Traceback|Error' $W/mixed.txt | cut -c1-290 >> $L
port=$((port + 800))
(cd $S && timeout 3000 python3 whole_seeded_pair.py --base $MAIN --arm $ARM --root $W/seeded --port $port > $W/seeded.txt 2>&1)
echo "seeded rc $? $(date +%T)" >> $L
grep -E 'byte identical|SIM over|WER over|Traceback' $W/seeded.txt | cut -c1-230 >> $L
port=$((port + 800))
pair small_n64 64 --no-stream $SMALL
pair small_n16 16 --no-stream $SMALL
(cd $S && BOX_TTS_MODEL=$SMALL timeout 3000 python3 whole_seeded_pair.py --base $MAIN --arm $ARM --root $W/small_seeded --port $port --no-streaming > $W/small_seeded.txt 2>&1)
echo "small seeded rc $? $(date +%T)" >> $L
grep -E 'byte identical|SIM over|WER over|Traceback|model ' $W/small_seeded.txt | cut -c1-230 >> $L
echo "queue $BLOCK done $(date +%T)" >> $L
