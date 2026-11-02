#!/usr/bin/env bash
# Block 108 of 11-02: does CPython's switch interval bound the engine at c64? One tree, an arm per interval, same time, both
# card orders, SeedTTS EN streaming: 5 ms (the default) against 1 ms, then against 0.2 ms, then c1 at the best. Decision rule,
# written before the read: a default change needs at least 3 % req/s at c64 in both orders with first chunk no worse and c1
# level, because a shorter interval trades lock hand-offs for throughput on every model in the process.
# usage: D=... T=<kit tree> TREE=<tree> [BLOCK=block108] [CARD_A=0] [CARD_B=1] [PORT=40000] bash queue_block108.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
TREE=${TREE:?the tree}
BLOCK=${BLOCK:-block108}
CARD_A=${CARD_A:-0}
CARD_B=${CARD_B:-1}
port=${PORT:-40000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "tree $TREE $(git -C $TREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $TREE diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/block96/cache/inductor TRITON_CACHE_DIR=$D/block96/cache/triton
run_arm() {
  local label=$1 interval=$2 card=$3 port=$4 conc=$5 barrier=$6
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms SGLANG_OMNI_SWITCH_INTERVAL_S=$interval timeout 2400 python3 serve_arm.py \
    --label $label --tree $TREE --card $card --port $port --concurrency $conc --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T); $(grep -hE 'RTF mean|Throughput \(req/s\)|TTFC mean|TTFC p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-150); tracebacks $(grep -c Traceback $W/arms/$label/serve.log)" >> $L
}
pair() {
  local name=$1 interval=$2 conc=$3
  run_arm d5ms_${name}_o0 0.005 $CARD_A $port $conc $W/barrier_${name}_o0 &
  run_arm s${name}_o0 $interval $CARD_B $((port + 400)) $conc $W/barrier_${name}_o0 &
  wait
  port=$((port + 800))
  run_arm d5ms_${name}_o1 0.005 $CARD_B $port $conc $W/barrier_${name}_o1 &
  run_arm s${name}_o1 $interval $CARD_A $((port + 400)) $conc $W/barrier_${name}_o1 &
  wait
  port=$((port + 800))
}
pair 1ms_c64 0.001 64
pair 200us_c64 0.0002 64
pair 1ms_c1 0.001 1
echo "queue $BLOCK done $(date +%T)" >> $L
