#!/usr/bin/env bash
# Block 140 of 11-05: what block 139 left. Streaming c64's second order, lost to a server's internal port landing on its partner's,
# with the partners far apart. VoiceDesign seeded at concurrency 1 and scored, so both arms decode the same talker codes.
# usage: D=... T=<kit tree> MAIN=<tree> ARM=<tree> [BLOCK=block140] bash queue_block140.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
ARM=${ARM:?the tree under test}
BLOCK=${BLOCK:-block140}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
LARGE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
DESIGN=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
DESIGN_ARGS="--task-type VoiceDesign --no-ref-audio --instructions=A_calm_adult_female_voice_speaking_clearly"
export CARD_CPUS="0=0-31,64-95;1=32-63,96-127"
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
mkdir -p $W
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD) main $(git -C $MAIN rev-parse HEAD) arm $(git -C $ARM rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
report() {
  local label=$1 card=$2 rc=$3
  echo "$label card $card exit $rc $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency mean|Latency p95|TTFC mean|TTFC p95|WER \(corpus' $W/arms/$label/bench.log $W/$label.out 2>/dev/null | sort -u | tr -s ' ' | tr '\n' ';' | cut -c1-260); memory $(cut -c1-60 $W/arms/$label/memory.txt 2>/dev/null); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
stream_arm() {
  local label=$1 tree=$2 card=$3 port=$4
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py --label $label --tree $tree --card $card --port $port --concurrency 64 --model $LARGE --barrier $W/barrier_s64 --parties 2 > $W/$label.out 2>&1)
  report $label $card $?
}
design_arm() {
  local label=$1 tree=$2 card=$3 port=$4
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 3000 python3 serve_arm.py --label $label --tree $tree --card $card --port $port --concurrency 1 --no-stream --seed 1234 --score --barrier $W/barrier_design --parties 2 --model $DESIGN "--bench-args=$DESIGN_ARGS" > $W/$label.out 2>&1)
  report $label $card $?
}
stream_arm main_s64_o1 $MAIN 1 28000 &
stream_arm head_s64_o1 $ARM 0 29400 &
wait
design_arm main_design_c1 $MAIN 0 25000 &
design_arm head_design_c1 $ARM 1 26400 &
wait
echo "queue $BLOCK done $(date +%T)" >> $L
