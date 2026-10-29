#!/usr/bin/env bash
# Block 17 of 10-29, after block 16 (WAIT_PID): the talker's loop on the top priority stream, against main (STATE 10-29
# item 2, option b). test/qwen3-tts-talker-top-priority 70e7a8783: main 08cecee99 with Qwen3-TTS's scheduler on the
# existing should_run_top_priority_stream (#2907's mechanism), so the talker's kernels sit above the reference encode
# (default priority) and the vocoder's decode (second highest). SeedTTS EN 1.7B Base, non-streaming c16 and streaming
# c16, both card orders, pairs held at serve_arm.py's barrier, scored; streaming arms also report first chunk and
# underruns.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
REPO=/workspace/sglang-omni
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
W=$D/block17
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs $W/barrier
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
git -C $REPO fetch -q origin test/qwen3-tts-talker-top-priority
rm -rf /workspace/n29_prio
git -C $REPO worktree prune
git -C $REPO worktree add -q --detach /workspace/n29_prio 70e7a8783
echo "queue block17 start $(date +%T) kit $(git -C $T rev-parse HEAD) prio $(git -C /workspace/n29_prio rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L

arm() {
  local barrier=$1 label=$2 tree=$3 card=$4 port=$5
  shift 5
  python3 $ARM --label $label --tree /workspace/n29_$tree --card $card --port $port --barrier $W/barrier/$barrier "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? $(grep -h -E 'RTF mean|WER \(corpus|Max token hits|Throughput|TTFC mean|TTFC p95|Max underrun p95' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) $(grep -h 'timed out' $W/logs/arm_$label.log) $(date +%T)" >> $L
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
pair c16n_a "c16n_a_main main 0 11000 $N16" "c16n_a_prio prio 1 11500 $N16"
pair c16n_b "c16n_b_prio prio 0 12000 $N16" "c16n_b_main main 1 12500 $N16"
pair c16s_a "c16s_a_main main 0 13000 $S16" "c16s_a_prio prio 1 13500 $S16"
pair c16s_b "c16s_b_prio prio 0 14000 $S16" "c16s_b_main main 1 14500 $S16"
echo "queue block17 done $(date +%T)" >> $L
