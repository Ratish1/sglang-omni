#!/usr/bin/env bash
# Nsight Systems pass over a bare Qwen3-TTS tree: the whole serve under nsys profile (CUDA
# with graph nodes, GPU metrics ad10x), one seed-tts stream benchmark as the load. The
# window is the benchmark's timed requests, cut from bench.log afterwards, so the means
# exclude startup, capture and warmup. Only the serve is sent TERM; nsys finalizes itself.
# usage: run_nsys_boot.sh <tree> <out dir> <card> <port> <concurrency> <max samples> [meta]
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4 CONC=$5 SAMPLES=$6 META=${7:-zhaochenyang20/seed-tts-eval-arrow}
S=$(cd "$(dirname "$0")" && pwd)
PY=/workspace/sglang-omni/.venv/bin/python
MODEL=/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base
URL=http://127.0.0.1:$PORT
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$OUT"
cd "$TREE" || exit 1
git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
md5sum "$0" "$S/nsys_metrics.py" > "$OUT/md5.txt"
nsys --version > "$OUT/nsys_version.txt" 2>&1
nvidia-smi > "$OUT/gpus_before.txt"
echo "start $(date +%T)" > "$OUT/progress.txt"

CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE nsys profile -o "$OUT/serve" --force-overwrite=true \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --gpu-metrics-devices=cuda-visible --gpu-metrics-set=ad10x --gpu-metrics-frequency=2000 \
  $PY -u -m sglang_omni.cli serve --model-path $MODEL --port $PORT > "$OUT/serve.log" 2>&1 &
NSYS_PID=$!

healthy=0
for _ in $(seq 180); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health)" = 200 ]; then healthy=1; break; fi
  kill -0 $NSYS_PID 2>/dev/null || break
  sleep 5
done
if [ $healthy = 1 ]; then
  echo "healthy $(date +%T)" >> "$OUT/progress.txt"
  PYTHONPATH=$TREE $PY -m benchmarks.eval.benchmark_tts_seedtts --model $MODEL --lang en --meta "$META" \
    --use-existing-server --host 127.0.0.1 --port $PORT --warmup 1 --stream --generate-only \
    --concurrency "$CONC" --max-samples "$SAMPLES" --output-dir "$OUT/bench" > "$OUT/bench.log" 2>&1
  echo "bench rc $? $(date +%T)" >> "$OUT/progress.txt"
else
  echo "server not healthy" > "$OUT/FAILED"
fi

# note(ratish): nsys carries the serve command line too; the serve is the first python
SERVE_PID=$(for p in $(pgrep -f "sglang_omni.cli serve.*--port $PORT"); do
  if [[ "$(cat /proc/$p/comm)" == python* ]]; then echo $p; fi; done | head -1)
echo "serve pid $SERVE_PID" >> "$OUT/progress.txt"
kill -TERM "$SERVE_PID"
for _ in $(seq 120); do
  kill -0 $NSYS_PID 2>/dev/null || break
  sleep 5
done
kill -0 $NSYS_PID 2>/dev/null && echo "nsys still running after 10 min" >> "$OUT/progress.txt"
echo "nsys ended $(date +%T)" >> "$OUT/progress.txt"
nvidia-smi > "$OUT/gpus_after.txt"
[ -f "$OUT/FAILED" ] && exit 1

nsys export --type sqlite --force-overwrite=true -o "$OUT/serve.sqlite" "$OUT/serve.nsys-rep" > "$OUT/export.log" 2>&1
$PY "$S/nsys_metrics.py" "$OUT/serve.sqlite" --bench-log "$OUT/bench.log" > "$OUT/metrics.txt" 2>&1
echo "done $(date +%T)" >> "$OUT/progress.txt"
