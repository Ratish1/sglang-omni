#!/usr/bin/env bash
# boot_server <worktree> <outdir> <srv_gpu> <port>
# Claims a card only if it is free at launch time, then waits on BOTH liveness
# and readiness so a dead process is reported in seconds, not after a timeout.
boot_server() {
  local wt=$1 d=$2 gpu=$3 port=$4
  local PY=/workspace/sglang-omni/.venv/bin/python
  mkdir -p "$d"

  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
  if [ "$used" -gt 500 ]; then
    echo "REFUSING: gpu $gpu has ${used} MiB in use (neighbour). Census:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4,5,6,7
    return 2
  fi
  echo "gpu $gpu free (${used} MiB) at $(date +%T); launching"

  # gate probe MUST run from the same cwd the server will use, or it reports
  # the main checkout's markers while the server runs the worktree's code.
  ( cd "$wt" && CUDA_VISIBLE_DEVICES=$gpu $PY - 2>&1 | grep -v torchada ) > "$d/gate.txt" <<'PROBE'
import inspect, sglang_omni
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.models.qwen3_tts import request_builders as rb
print("import  ", sglang_omni.__file__)
print("cpu_drop", "stays on the device" in inspect.getsource(rb.apply_sglang_qwen3_tts_result))
print("restage ", "pin_memory=pin_memory" in inspect.getsource(Qwen3TTSTalker.prepare_decode_buffers))
PROBE
  echo "gate: $(tr '\n' ' ' < "$d/gate.txt")"

  ( cd "$wt" && CUDA_VISIBLE_DEVICES=$gpu nohup $PY -m sglang_omni.cli serve \
      --config examples/configs/qwen3_tts_0_6b.yaml --port "$port" > "$d/serve.log" 2>&1 & \
    echo $! > "$d/server.pid" )
  local pid; pid=$(cat "$d/server.pid")

  local i
  for i in $(seq 1 90); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "SERVER DIED after $((i*5))s. cause:"
      grep -iE "OutOfMemoryError|RuntimeError|Error:" "$d/serve.log" | tail -3 | cut -c1-200
      return 1
    fi
    if curl -sf -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "READY after $((i*5))s at $(date +%T)"
      return 0
    fi
    sleep 5
  done
  echo "TIMEOUT after 450s; last log line:"; tail -1 "$d/serve.log" | cut -c1-160
  return 1
}
