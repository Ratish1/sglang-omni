#!/usr/bin/env bash
# Block 66 of 11-01: startup on main as it now stands (#3403, #3404, #3405 merged) against Fun-CosyVoice3 on the same
# card and day. Sets up CosyVoice3 as its cookbook does (CosyVoice and Matcha-TTS at the pinned commits on PYTHONPATH,
# the public checkpoint downloaded without a token), then one boot at a time on card 0, each model at its cookbook's
# default command: Qwen3-TTS 1.7B Base cold into a fresh cache and three warm boots, then CosyVoice3 the same. Every
# boot's log is kept for the per-family split.
# usage: D=... T=<kit tree> MAIN=<tree> bash queue_block66.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block66
L=$W/queue.log
C=$D/CosyVoice
mkdir -p $W/logs
echo "tree MAIN $MAIN $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $MAIN diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
if [ ! -d $C/.git ]; then
  git clone -q --recursive https://github.com/FunAudioLLM/CosyVoice.git $C >> $W/setup.log 2>&1
else
  :
fi
git -C $C checkout -q 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc >> $W/setup.log 2>&1
git -C $C submodule update -q --init --recursive >> $W/setup.log 2>&1
git -C $C/third_party/Matcha-TTS checkout -q dd9105b34bf2be2230f4aa1e4769fb586a3c824e >> $W/setup.log 2>&1
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512'))" >> $W/setup.log 2>&1
echo "setup: CosyVoice $(git -C $C rev-parse --short HEAD) Matcha $(git -C $C/third_party/Matcha-TTS rev-parse --short HEAD); $(tail -n 1 $W/setup.log)" >> $L
echo "queue block66 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

boot() {
  local label=$1 model=$2 extra_path=$3 port=$4
  local began=$(date +%s.%N)
  (cd $MAIN && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$MAIN$extra_path TORCHINDUCTOR_CACHE_DIR=$W/cache_${label%%_*}/inductor TRITON_CACHE_DIR=$W/cache_${label%%_*}/triton \
    setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port \
    --model-path $model" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 1
  local pgid=$(cat $W/logs/serve_$label.pid)
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null; then
      echo "$label died; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 1
    fi
  done
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}
port=24000
QWEN=Qwen/Qwen3-TTS-12Hz-1.7B-Base
COSY=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYPATH=":$C:$C/third_party/Matcha-TTS"
for label in qwen_cold qwen_warm_1 qwen_warm_2 qwen_warm_3; do
  boot $label $QWEN "" $port
  port=$((port + 200))
done
for label in cosy_cold cosy_warm_1 cosy_warm_2 cosy_warm_3; do
  boot $label $COSY "$COSYPATH" $port
  port=$((port + 200))
done
echo "queue block66 done $(date +%T)" >> $L
