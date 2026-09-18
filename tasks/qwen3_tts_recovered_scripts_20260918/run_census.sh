#!/usr/bin/env bash
# A/B census for #2126 (non-blocking copies) on moss, GPU 4, one card for every boot.
set -uo pipefail
ROOT=/workspace/sglang-omni
PY=$ROOT/.venv/bin/python
OUT=$ROOT/.tmp/nbc-4090
GPU=4
PORT=31004
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
META=zhaochenyang20/seed-tts-eval-arrow
BENCH=benchmarks/eval/benchmark_tts_seedtts.py
mkdir -p $OUT

wait_ready() {
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
    kill -0 $1 2>/dev/null || { echo "SERVER DIED"; return 1; }
    sleep 5
  done
  echo "TIMEOUT waiting for server"; return 1
}

wait_gpu_free() {
  for i in $(seq 1 60); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    [ "$u" -lt 500 ] && return 0
    sleep 3
  done
}

boot() {  # $1=arm $2=label $3=with_stack(0|1)
  arm=$1; label=$2; stack=$3
  wt=$ROOT/.tmp/wt/$arm
  d=$OUT/$label
  rm -rf $d; mkdir -p $d
  cd $wt

  git rev-parse HEAD > $d/head.txt
  git diff --stat >> $d/head.txt
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $d/gpus_before.txt

  CUDA_VISIBLE_DEVICES=$GPU $PY - > $d/gate.txt 2>&1 <<'PY'
import inspect, sglang_omni
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
print("import", sglang_omni.__file__)
print("cleanup", "pin_memory=pin_memory" in inspect.getsource(Qwen3TTSTalker.prepare_decode_buffers))
print("early_ids", "only host read" in inspect.getsource(Qwen3TTSModelRunner._collect_codes))
PY
  cat $d/gate.txt

  export SGLANG_TORCH_PROFILER_DIR=$d/profiles
  export SGLANG_TORCH_PROFILER_WITH_STACK=$stack
  mkdir -p $d/profiles
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY -m sglang_omni.cli serve \
      --config examples/configs/qwen3_tts_1_7b.yaml --port $PORT > $d/serve.log 2>&1 &
  SRV=$!
  echo $SRV > $d/server.pid
  wait_ready $SRV || { kill -9 $SRV 2>/dev/null; wait_gpu_free; return 1; }
  echo "[$label] server up"
}

shutdown() {
  d=$1
  kill -TERM $(cat $d/server.pid) 2>/dev/null
  sleep 20
  kill -9 $(cat $d/server.pid) 2>/dev/null
  wait_gpu_free
}

bench() {  # $1=outdir $2=extra flags...
  o=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU $PY $BENCH \
    --base-url http://127.0.0.1:$PORT --model $MODEL --meta $META \
    --ref-format references --generate-only \
    --output-dir $o "$@" 2>&1 | tail -30
}

census_boot() {  # $1=arm $2=label
  arm=$1; label=$2
  boot $arm $label 0 || return 1
  d=$OUT/$label
  # streaming: pass1 warms, pass2 measured with events
  bench $d/s1 --stream --response-format pcm --warmup 1 --concurrency 16 > $d/s1.log 2>&1
  curl -sf -X POST "http://127.0.0.1:$PORT/start_request_profile" \
       -H 'content-type: application/json' \
       -d "{\"run_id\":\"$label\",\"event_dir\":\"$d/events\"}" > $d/prof_start.json
  bench $d/s2 --stream --response-format pcm --warmup 1 --concurrency 16 > $d/s2.log 2>&1
  curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" \
       -H 'content-type: application/json' -d "{\"run_id\":\"$label\"}" > $d/prof_stop.json
  # non-streaming
  bench $d/n1 --response-format wav --warmup 1 --concurrency 16 > $d/n1.log 2>&1
  shutdown $d
  echo "[$label] done"
}

for pair in A1:A B1:B A2:A B2:B; do
  label=${pair%%:*}; arm=${pair##*:}
  echo "================= $label (arm $arm) ================="
  census_boot $arm $label
done
echo "ALL CENSUS BOOTS DONE"
