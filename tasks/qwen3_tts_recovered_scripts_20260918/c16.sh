#!/usr/bin/env bash
set -uo pipefail
ROOT=/workspace/sglang-omni; PY=$ROOT/.venv/bin/python
OUT=$ROOT/.tmp/cpu16; SRV=4; CLI=5; PORT=31004; CONC=16
MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base; CFG=examples/configs/qwen3_tts_0_6b.yaml
META=zhaochenyang20/seed-tts-eval-arrow
mkdir -p $OUT
bench(){ o=$1; shift; CUDA_VISIBLE_DEVICES=$CLI timeout 3000 $PY -m benchmarks.eval.benchmark_tts_seedtts \
  --base-url http://127.0.0.1:$PORT --model $MODEL --meta $META --ref-format references \
  --generate-only --warmup 1 --concurrency $CONC --output-dir $o "$@"; }
run(){
  arm=$1; label=$2; wt=$ROOT/.tmp/wt/$arm; d=$OUT/$label
  rm -rf $d; mkdir -p $d; cd $wt
  git rev-parse HEAD > $d/head.txt; git diff --stat >> $d/head.txt
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $d/gpus_before.txt
  CUDA_VISIBLE_DEVICES=$SRV $PY - > $d/gate.txt 2>&1 <<'PY'
import inspect, sglang_omni
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.models.qwen3_tts import request_builders as rb
print("import", sglang_omni.__file__)
print("cpu_drop", "stays on the device" in inspect.getsource(rb.apply_sglang_qwen3_tts_result))
print("restage", "pin_memory=pin_memory" in inspect.getsource(Qwen3TTSTalker.prepare_decode_buffers))
PY
  echo "[$label] $(grep -v torchada $d/gate.txt | tr '\n' ' ')"
  CUDA_VISIBLE_DEVICES=$SRV nohup $PY -m sglang_omni.cli serve --config $CFG --port $PORT > $d/serve.log 2>&1 &
  echo $! > $d/server.pid
  for i in $(seq 1 90); do curl -sf -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 5; done
  curl -sf -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "[$label] FAILED TO BOOT"; tail -15 $d/serve.log; return 1; }
  echo "[$label] up $(date +%T)"
  bench $d/s1 --stream --response-format pcm > $d/s1.log 2>&1
  curl -sf -X POST http://127.0.0.1:$PORT/start_request_profile -H 'content-type: application/json' \
    -d "{\"run_id\":\"$label\",\"event_dir\":\"$d/events\"}" > /dev/null
  bench $d/s2 --stream --response-format pcm > $d/s2.log 2>&1
  curl -sf -X POST http://127.0.0.1:$PORT/stop_profile -H 'content-type: application/json' -d "{\"run_id\":\"$label\"}" > /dev/null
  bench $d/n1 --response-format wav > $d/n1.log 2>&1
  kill -TERM $(cat $d/server.pid) 2>/dev/null; sleep 20
  pkill -9 -f "sglang_omni.cli serve" 2>/dev/null; pkill -9 -f spawn_main 2>/dev/null; sleep 8
  echo "[$label] done $(date +%T)  gpu4=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 4)"
}
for p in A1:A C1:C A2:A C2:C; do run ${p##*:} ${p%%:*}; done
echo "C16 CENSUS COMPLETE $(date +%T)"
