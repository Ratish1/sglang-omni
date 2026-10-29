#!/usr/bin/env bash
# Block 15b of 10-29: block 15's served pairs again, each pair's arms held at a barrier until both servers are healthy
# and again until both benches end (serve_arm.py --barrier). In block 15 the branch's cold compile booted 60 s after
# main, so main benched during the branch's compile and the branch benched during main's scoring; those pairs are void.
# The branch's tree is /workspace/n29_compile (92025c654), its inductor cache warmed by block 15's first boot.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
W=$D/block15b
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs $W/barrier
echo "queue block15b start $(date +%T) kit $(git -C $T rev-parse HEAD) compile $(git -C /workspace/n29_compile rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L

arm() {
  local barrier=$1 label=$2 tree=$3 card=$4 port=$5
  shift 5
  python3 $ARM --label $label --tree /workspace/n29_$tree --card $card --port $port --barrier $W/barrier/$barrier "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? startup $(cat $ARM_ROOT/$label/startup.txt 2> /dev/null) $(grep -h -E 'RTF mean|WER \(corpus|Max token hits|Throughput' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) $(grep -h 'timed out' $W/logs/arm_$label.log) $(date +%T)" >> $L
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
N1="--concurrency 1 --no-stream --score"
pair c16n_a "c16n_a_main main 0 11000 $N16" "c16n_a_compile compile 1 11500 $N16"
pair c16n_b "c16n_b_compile compile 0 12000 $N16" "c16n_b_main main 1 12500 $N16"
pair c16s_a "c16s_a_main main 0 13000 $S16" "c16s_a_compile compile 1 13500 $S16"
pair c16s_b "c16s_b_compile compile 0 14000 $S16" "c16s_b_main main 1 14500 $S16"
pair c1n_a "c1n_a_main main 0 17000 $N1" "c1n_a_compile compile 1 17500 $N1"
pair c1n_b "c1n_b_compile compile 0 18000 $N1" "c1n_b_main main 1 18500 $N1"
echo "queue block15b done $(date +%T)" >> $L
