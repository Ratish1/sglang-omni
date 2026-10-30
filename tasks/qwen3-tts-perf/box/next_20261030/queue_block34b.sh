#!/usr/bin/env bash
# Block 34b of 10-30: block 34 rerun with two settings fixed. Block 34's split layout took the cookbook's fractions
# (talker 0.85, vocoder 0.10), measured on an H200; on an H100 the talker's 0.85 left the vocoder process 10.7 GiB and
# main's vocoder ran out of memory in 33 of 1,088 requests. The vocoder's fraction does not cap it, the talker's static
# fraction decides what is left, so here the talker declares 0.78 with its engine's static fraction 0.78 (the
# cookbook's split frontend keeps the two equal) and the vocoder 0.17. And a stage worker's listening port (random,
# 32768 to 60999) took a scorer's port + 400, so every port is below 32768, the pair's arms 200 apart, each pair 1000
# from the next.
# The talker priority branch with the default-stream hand-off (BR) against main 466da750d, SeedTTS EN 1.7B Base,
# serve_arm.py at its barrier, scored on each arm's card: the vocoder in its own process, non-streaming and streaming
# c16 in both card orders; then the one-process default, non-streaming c64 and c16, streaming c16 and c64, both orders.
# usage: D=... T=... BR=<commit> bash queue_block34b.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BR=${BR:?the branch commit}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
MAIN=/workspace/n30_main
BRANCH=/workspace/n30_prio_$BR
W=$D/block34b
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs $W/barrier
echo "queue block34b start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')branch $(git -C $BRANCH rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

arm() {
  local barrier=$1 label=$2 tree=$3 card=$4 port=$5
  shift 5
  python3 $ARM --label $label --tree $tree --card $card --port $port --barrier $W/barrier/$barrier "--serve-args=$SERVE_ARGS" "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? $(grep -h -E 'Completed requests|Failed requests|RTF mean|WER \(corpus|Max token hits|Throughput|TTFC mean|TTFC p95|Max underrun p95' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) errors $(grep -c -E 'device-side assert|CUDA error|Traceback|cannot pickle|out of memory' $ARM_ROOT/$label/serve.log) $(date +%T)" >> $L
}

pair() {
  local name=$1
  arm $name $2 &
  local a=$!
  arm $name $3 &
  local b=$!
  wait $a $b
}

N16="--concurrency 16 --no-stream --score"
S16="--concurrency 16 --score"
N64="--concurrency 64 --no-stream --score"
S64="--concurrency 64 --score"
SERVE_ARGS="--vocoder.process vocoder --tts_engine.gpu_memory_fraction 0.78 --tts_engine.engine.mem_fraction_static 0.78 --vocoder.gpu_memory_fraction 0.17"
pair sp_n16_a "sp_n16_a_main $MAIN 0 20000 $N16" "sp_n16_a_prio $BRANCH 1 20200 $N16"
pair sp_n16_b "sp_n16_b_prio $BRANCH 0 21000 $N16" "sp_n16_b_main $MAIN 1 21200 $N16"
pair sp_s16_a "sp_s16_a_main $MAIN 0 22000 $S16" "sp_s16_a_prio $BRANCH 1 22200 $S16"
pair sp_s16_b "sp_s16_b_prio $BRANCH 0 23000 $S16" "sp_s16_b_main $MAIN 1 23200 $S16"
echo "split layout done $(date +%T)" >> $L
SERVE_ARGS=""
pair op_n64_a "op_n64_a_main $MAIN 0 24000 $N64" "op_n64_a_prio $BRANCH 1 24200 $N64"
pair op_n64_b "op_n64_b_prio $BRANCH 0 25000 $N64" "op_n64_b_main $MAIN 1 25200 $N64"
pair op_n16_a "op_n16_a_main $MAIN 0 26000 $N16" "op_n16_a_prio $BRANCH 1 26200 $N16"
pair op_n16_b "op_n16_b_prio $BRANCH 0 27000 $N16" "op_n16_b_main $MAIN 1 27200 $N16"
pair op_s16_a "op_s16_a_main $MAIN 0 28000 $S16" "op_s16_a_prio $BRANCH 1 28200 $S16"
pair op_s16_b "op_s16_b_prio $BRANCH 0 29000 $S16" "op_s16_b_main $MAIN 1 29200 $S16"
pair op_s64_a "op_s64_a_main $MAIN 0 30000 $S64" "op_s64_a_prio $BRANCH 1 30200 $S64"
pair op_s64_b "op_s64_b_prio $BRANCH 0 31000 $S64" "op_s64_b_main $MAIN 1 31200 $S64"
echo "queue block34b done $(date +%T)" >> $L
