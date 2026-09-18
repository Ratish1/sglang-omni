#!/usr/bin/env bash
# Non-streaming c16, A = upstream/main, B = PR head. Interleaved, one pass per boot.
set -uo pipefail
ROOT=/workspace/sglang-omni; PY=$ROOT/.venv/bin/python
OUT=$ROOT/.tmp/ns16; SRV=0; CLI=4; PORT=31000; CONC=16
MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base; CFG=examples/configs/qwen3_tts_0_6b.yaml
META=zhaochenyang20/seed-tts-eval-arrow
# note(ratish): the model and corpus are cached; going offline keeps a flaky
# HF proxy from killing a boot mid census.
export HF_HUB_OFFLINE=1
mkdir -p $OUT
run(){
  arm=$1; label=$2; wt=$ROOT/.tmp/wt/$arm; d=$OUT/$label
  rm -rf $d; mkdir -p $d
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $SRV)
  [ "$used" -gt 500 ] && { echo "[$label] REFUSE gpu$SRV busy ${used}MiB"; return 2; }
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $d/gpus_before.txt
  ( cd $wt
    CUDA_VISIBLE_DEVICES=$SRV $PY - > $d/gate.txt 2>&1 <<'PROBE'
import inspect, sglang_omni
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.models.qwen3_tts import request_builders as rb
print("import", sglang_omni.__file__)
print("cpu_drop", "vocoder decodes these" in inspect.getsource(rb.apply_sglang_qwen3_tts_result))
print("restage", "is_pin_memory_available" in inspect.getsource(Qwen3TTSTalker.prepare_decode_buffers))
PROBE
    CUDA_VISIBLE_DEVICES=$SRV nohup $PY -m sglang_omni.cli serve --config $CFG --port $PORT > $d/serve.log 2>&1 &
    echo $! > $d/server.pid )
  echo "[$label] gate: $(grep -v torchada $d/gate.txt | tr '\n' ' ')"
  pid=$(cat $d/server.pid)
  for i in $(seq 1 90); do
    kill -0 $pid 2>/dev/null || { echo "[$label] DIED: $(grep -i OutOfMemory $d/serve.log | tail -1 | cut -c1-110)"; return 1; }
    curl -sf -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break
    sleep 5
  done
  curl -sf -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "[$label] never ready"; return 1; }
  echo "[$label] up $(date +%T)"
  CUDA_VISIBLE_DEVICES=$CLI timeout 3000 $PY -m benchmarks.eval.benchmark_tts_seedtts \
    --base-url http://127.0.0.1:$PORT --use-existing-server --model $MODEL --meta $META \
    --ref-format references --generate-only --warmup 1 --concurrency $CONC \
    --response-format wav --output-dir $d/n1 > $d/n1.log 2>&1
  kill -TERM $pid 2>/dev/null; sleep 18
  pkill -9 -f "sglang_omni.cli serve" 2>/dev/null; pkill -9 -f spawn_main 2>/dev/null; sleep 6
  echo "[$label] done $(date +%T)"
}
for p in A1:nsA B1:nsB A2:nsA B2:nsB; do run ${p##*:} ${p%%:*}; done
echo "NS16 COMPLETE $(date +%T)"
