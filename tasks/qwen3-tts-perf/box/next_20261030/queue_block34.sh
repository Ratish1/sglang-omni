#!/usr/bin/env bash
# Block 34 of 10-30: the talker priority branch with the default-stream hand-off (BR) against main 466da750d, the
# standard SeedTTS EN A/B on 1.7B Base with serve_arm.py at its barrier, scored on each arm's card. First the layout the
# hand-off is for, the vocoder in its own process (the cookbook's --vocoder.process vocoder with its fractions),
# non-streaming and streaming c16 in both card orders; then the one-process default, where the hand-off replaced the
# vocoder's event wait: non-streaming c64 (the gain's load) and c16, streaming c16, both orders.
# usage: D=... T=... BR=<commit> bash queue_block34.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BR=${BR:?the branch commit}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
MAIN=/workspace/n30_main
BRANCH=/workspace/n30_prio_$BR
W=$D/block34
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs $W/barrier
echo "queue block34 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')branch $(git -C $BRANCH rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

arm() {
  local barrier=$1 label=$2 tree=$3 card=$4 port=$5
  shift 5
  python3 $ARM --label $label --tree $tree --card $card --port $port --barrier $W/barrier/$barrier "--serve-args=$SERVE_ARGS" "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? $(grep -h -E 'Completed requests|Failed requests|RTF mean|WER \(corpus|Max token hits|Throughput|TTFC mean|TTFC p95|Max underrun p95' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) errors $(grep -c -E 'device-side assert|CUDA error|Traceback|cannot pickle' $ARM_ROOT/$label/serve.log) $(date +%T)" >> $L
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
SERVE_ARGS="--vocoder.process vocoder --tts_engine.gpu_memory_fraction 0.85 --vocoder.gpu_memory_fraction 0.10"
pair sp_n16_a "sp_n16_a_main $MAIN 0 40000 $N16" "sp_n16_a_prio $BRANCH 1 41000 $N16"
pair sp_n16_b "sp_n16_b_prio $BRANCH 0 42000 $N16" "sp_n16_b_main $MAIN 1 43000 $N16"
pair sp_s16_a "sp_s16_a_main $MAIN 0 44000 $S16" "sp_s16_a_prio $BRANCH 1 45000 $S16"
pair sp_s16_b "sp_s16_b_prio $BRANCH 0 46000 $S16" "sp_s16_b_main $MAIN 1 47000 $S16"
echo "split layout done $(date +%T)" >> $L
SERVE_ARGS=""
pair op_n64_a "op_n64_a_main $MAIN 0 48000 $N64" "op_n64_a_prio $BRANCH 1 49000 $N64"
pair op_n64_b "op_n64_b_prio $BRANCH 0 50000 $N64" "op_n64_b_main $MAIN 1 51000 $N64"
pair op_n16_a "op_n16_a_main $MAIN 0 52000 $N16" "op_n16_a_prio $BRANCH 1 53000 $N16"
pair op_n16_b "op_n16_b_prio $BRANCH 0 54000 $N16" "op_n16_b_main $MAIN 1 55000 $N16"
pair op_s16_a "op_s16_a_main $MAIN 0 56000 $S16" "op_s16_a_prio $BRANCH 1 57000 $S16"
pair op_s16_b "op_s16_b_prio $BRANCH 0 58000 $S16" "op_s16_b_main $MAIN 1 59000 $S16"
echo "queue block34 done $(date +%T)" >> $L
