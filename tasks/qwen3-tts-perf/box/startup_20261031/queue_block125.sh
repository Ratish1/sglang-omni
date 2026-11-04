#!/usr/bin/env bash
# Block 125 of 11-04: the rest of the head's read against main. 0.6B seeds 2345 and 3456, because seed 1234's SIM interval
# (-0.034 to -0.001) just excludes zero and a worse lean gets new seeds until bounded. CustomVoice and VoiceDesign by block 94's
# driver, which that path's changed rounding needs.
# usage: D=... T=<kit tree> MAIN=<tree> ARM=<tree> [BLOCK=block125] [PORT=2050] bash queue_block125.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
ARM=${ARM:?the head}
BLOCK=${BLOCK:-block125}
PORT=${PORT:-2050}
S=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ') head $(git -C $ARM rev-parse HEAD HEAD:sglang_omni | tr '\n' ' '); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
port=$PORT
for seed in 2345 3456; do
  (cd $S && BOX_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base timeout 3000 python3 whole_seeded_pair.py --base $MAIN --arm $ARM --root $W/small_seed_$seed --port $port --seed $seed --no-streaming > $W/small_seed_$seed.txt 2>&1)
  echo "small seed $seed rc $? $(date +%T)" >> $L
  grep -E 'byte identical|SIM over|WER over|Traceback' $W/small_seed_$seed.txt | cut -c1-230 >> $L
  port=$((port + 800))
done
D=$W T=$T PR=$MAIN WHOLE=$ARM bash $S/queue_block94.sh > $W/voice.out 2>&1
echo "voice rc $? $(date +%T)" >> $L
cut -c1-300 $W/block94/queue.log >> $L
echo "queue $BLOCK done $(date +%T)" >> $L
