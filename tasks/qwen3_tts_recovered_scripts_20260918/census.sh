#!/usr/bin/env bash
set -uo pipefail
ROOT=/workspace/sglang-omni
PY=$ROOT/.venv/bin/python
OUT=$ROOT/.tmp/nbc-4090
SRV_GPU=4; CLI_GPU=5; PORT=31004; CONC=12
MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
CFG=examples/configs/qwen3_tts_0_6b.yaml
META=zhaochenyang20/seed-tts-eval-arrow
mkdir -p $OUT

wait_gpu_free(){ for i in $(seq 1 80); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $SRV_GPU); [ "$u" -lt 500 ] && return 0; sleep 3; done; }

bench(){ o=$1; shift; CUDA_VISIBLE_DEVICES=$CLI_GPU timeout 2400 $PY -m benchmarks.eval.benchmark_tts_seedtts \
  --base-url http://127.0.0.1:$PORT --model $MODEL --meta $META --ref-format references \
  --generate-only --warmup 1 --concurrency $CONC --output-dir $o "$@"; }

run_boot(){
  arm=$1; label=$2; stack=$3
  wt=$ROOT/.tmp/wt/$arm; d=$OUT/$label; rm -rf $d; mkdir -p $d; cd $wt
  git rev-parse HEAD > $d/head.txt; git diff --stat >> $d/head.txt
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $d/gpus_before.txt
  CUDA_VISIBLE_DEVICES=$SRV_GPU $PY - > $d/gate.txt 2>&1 <<'PY'
import inspect, sglang_omni
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
print("import", sglang_omni.__file__)
print("cleanup", "pin_memory=pin_memory" in inspect.getsource(Qwen3TTSTalker.prepare_decode_buffers))
print("early_ids", "only host read" in inspect.getsource(Qwen3TTSModelRunner._collect_codes))
PY
  echo "[$label] gate: $(tr '\n' ' ' < $d/gate.txt)"
  export SGLANG_TORCH_PROFILER_DIR=$d/profiles SGLANG_TORCH_PROFILER_WITH_STACK=$stack
  mkdir -p $d/profiles
  CUDA_VISIBLE_DEVICES=$SRV_GPU nohup $PY -m sglang_omni.cli serve --config $CFG --port $PORT > $d/serve.log 2>&1 &
  echo $! > $d/server.pid
  for i in $(seq 1 120); do curl -sf -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 5; done
  curl -sf -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "[$label] SERVER FAILED"; cat $d/serve.log | tail -20; return 1; }
  echo "[$label] up"

  if [ "$stack" = "1" ]; then
    bench $d/warm --stream --response-format pcm --max-samples 96 > $d/warm.log 2>&1
    curl -sf -X POST http://127.0.0.1:$PORT/start_profile -H 'content-type: application/json' \
      -d "{\"run_id\":\"$label\",\"enable_torch\":true}" > $d/prof_start.json
    bench $d/p --stream --response-format pcm --max-samples 96 > $d/p.log 2>&1
    curl -sf -X POST http://127.0.0.1:$PORT/stop_profile -H 'content-type: application/json' -d "{\"run_id\":\"$label\"}" > $d/prof_stop.json
    sleep 45
  else
    bench $d/s1 --stream --response-format pcm > $d/s1.log 2>&1
    curl -sf -X POST http://127.0.0.1:$PORT/start_request_profile -H 'content-type: application/json' \
      -d "{\"run_id\":\"$label\",\"event_dir\":\"$d/events\"}" > $d/prof_start.json
    bench $d/s2 --stream --response-format pcm > $d/s2.log 2>&1
    curl -sf -X POST http://127.0.0.1:$PORT/stop_profile -H 'content-type: application/json' -d "{\"run_id\":\"$label\"}" > $d/prof_stop.json
    bench $d/n1 --response-format wav > $d/n1.log 2>&1
  fi
  kill -TERM $(cat $d/server.pid) 2>/dev/null; sleep 20; kill -9 $(cat $d/server.pid) 2>/dev/null
  wait_gpu_free
  echo "[$label] done"
}

for p in A1:A:0 B1:B:0 A2:A:0 B2:B:0 Ap:A:1 Bp:B:1; do
  l=${p%%:*}; r=${p#*:}; arm=${r%%:*}; st=${r##*:}
  echo "=============== $l (arm $arm, stack=$st) ==============="
  run_boot $arm $l $st
done
echo "CENSUS COMPLETE"
