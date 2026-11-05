#!/usr/bin/env bash
# Block 141 of 11-05: speaker similarity and WER on the final head against main, so the PR's quality tables come from the head and
# not through a chain of byte identities. Seeded concurrency 1, non-streaming, all 1,088 prompts: 1.7B seed 1234, 0.6B three seeds.
# usage: D=... T=<kit tree> MAIN=<tree> ARM=<tree> [BLOCK=block141] [PORT=20000] bash queue_block141.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
ARM=${ARM:?the tree under test}
BLOCK=${BLOCK:-block141}
PORT=${PORT:-20000}
S=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
SMALL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
mkdir -p $W
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD) main $(git -C $MAIN rev-parse HEAD) arm $(git -C $ARM rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
port=$PORT
read_pair() {
  local name=$1 seed=$2 model=$3
  (cd $S && BOX_TTS_MODEL=$model timeout 3000 python3 whole_seeded_pair.py --base $MAIN --arm $ARM --root $W/$name --port $port --seed $seed --no-streaming > $W/$name.txt 2>&1)
  echo "$name rc $? $(date +%T)" >> $L
  grep -E 'byte identical|SIM over|WER over|Traceback' $W/$name.txt | cut -c1-230 >> $L
  port=$((port + 1500))
}
read_pair large_1234 1234 Qwen/Qwen3-TTS-12Hz-1.7B-Base
read_pair small_1234 1234 $SMALL
read_pair small_2345 2345 $SMALL
read_pair small_3456 3456 $SMALL
echo "queue $BLOCK done $(date +%T)" >> $L
