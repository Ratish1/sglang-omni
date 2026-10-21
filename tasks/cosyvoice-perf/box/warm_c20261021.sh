#!/usr/bin/env bash
# One server of a tree on one card, 64 streaming c16 requests, then stopped by its process group: fills the tree's
# Inductor cache (Triton picks included) before any pair boots both arms from it, and serves as the smoke test.
# 10-21 copy of warm_c20261020.sh under /data/c20261021.
# PAIR_HOME names the run home (default /data/c20261021), for the 0.5.22 container's /data/c20261021b.
# usage (in the container): bash warm_c20261021.sh <tree> <card> <cpus> <port> <out>
TREE=$1 CARD=$2 CPUS=$3 PORT=$4 R=$5
D=${PAIR_HOME:-/data/c20261021}
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
mkdir -p $R
cd /workspace/$TREE
echo "start $(date +%T) tree $TREE head $(git rev-parse --short HEAD) card $CARD port $PORT" > $R/progress.txt
setsid env CUDA_VISIBLE_DEVICES=$CARD SGLANG_OMNI_STRICT_PORT=1 TORCHINDUCTOR_CACHE_DIR=$D/inductor_$TREE \
  PYTHONPATH=/workspace/$TREE:$COSYVOICE taskset -c $CPUS \
  python3 -m sglang_omni.cli serve --model-path $MODEL --host 127.0.0.1 --port $PORT > $R/serve.log 2>&1 < /dev/null &
SERVE_PID=$!
echo "serve pgid $SERVE_PID" >> $R/progress.txt
for i in $(seq 1 360); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null && break
  kill -0 $SERVE_PID 2> /dev/null || { echo "server died $(date +%T)" >> $R/progress.txt; exit 1; }
  sleep 5
done
echo "healthy $(date +%T)" >> $R/progress.txt
taskset -c $CPUS env CUDA_VISIBLE_DEVICES= python3 -m benchmarks.eval.benchmark_tts_seedtts --model $MODEL --port $PORT \
  --use-existing-server --generate-only --stream --lang en --warmup 1 --max-concurrency 16 --max-samples 64 \
  --output-dir $R/bench > $R/bench.log 2>&1
echo "bench rc $? $(date +%T)" >> $R/progress.txt
kill -TERM -- -$SERVE_PID
sleep 20
kill -KILL -- -$SERVE_PID 2> /dev/null
echo "done $(date +%T)" >> $R/progress.txt
