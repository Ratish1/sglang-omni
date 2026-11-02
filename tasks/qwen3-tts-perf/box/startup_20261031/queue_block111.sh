#!/usr/bin/env bash
# Block 111 of 11-02: how much of c64 is the preprocessing pool's hold on the stage's interpreter lock? The default layout
# against preprocessing in its own process, same tree, same time, both card orders, SeedTTS EN streaming at 64 and 1. This
# is an upper bound read, not a fix: the split also moves the speaker and reference encoders to a second CUDA context, which
# the vocoder split showed is expensive. A gain says how much a device-free split of tokenizing and audio loading could reach.
# usage: D=... T=<kit tree> TREE=<tree> [BLOCK=block111] [CARD_A=0] [CARD_B=1] [PORT=48000] bash queue_block111.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
TREE=${TREE:?the tree}
BLOCK=${BLOCK:-block111}
CARD_A=${CARD_A:-0}
CARD_B=${CARD_B:-1}
port=${PORT:-48000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "tree $TREE $(git -C $TREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $TREE diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/block96/cache/inductor TRITON_CACHE_DIR=$D/block96/cache/triton
run_arm() {
  local label=$1 serve_args=$2 card=$3 port=$4 conc=$5 barrier=$6
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py \
    --label $label --tree $TREE --card $card --port $port --concurrency $conc --serve-args "$serve_args" --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T); $(grep -hE 'RTF mean|Throughput \(req/s\)|TTFC mean|TTFC p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-150); tracebacks $(grep -c Traceback $W/arms/$label/serve.log)" >> $L
}
pair() {
  local conc=$1
  run_arm one_c${conc}_o0 "" $CARD_A $port $conc $W/barrier_c${conc}_o0 &
  run_arm split_c${conc}_o0 "--preprocessing.process preprocessing" $CARD_B $((port + 400)) $conc $W/barrier_c${conc}_o0 &
  wait
  port=$((port + 800))
  run_arm one_c${conc}_o1 "" $CARD_B $port $conc $W/barrier_c${conc}_o1 &
  run_arm split_c${conc}_o1 "--preprocessing.process preprocessing" $CARD_A $((port + 400)) $conc $W/barrier_c${conc}_o1 &
  wait
  port=$((port + 800))
}
pair 64
echo "queue $BLOCK done $(date +%T)" >> $L
