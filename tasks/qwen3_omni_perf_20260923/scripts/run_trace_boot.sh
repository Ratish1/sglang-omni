#!/usr/bin/env bash
# Torch profiler traces of every stage process of one colocated Qwen3-Omni serve over a
# window of requests, through the server's own /start_profile and /stop_profile (the
# omni-gpu-deep-dive server capture path). MODE formal serves the shipped h200 config;
# MODE mapping turns every CUDA graph off and records python stacks, so each kernel maps
# to its launch line. WARM requests of the arm run first so every shape bucket is warm,
# then WINDOW requests run inside the profiled window. One trace per process lands as
# <out>/<stage>_rank0.trace.json.gz once its background gzip is done. EXTRA_SERVE_ARGS is appended.
# usage: run_trace_boot.sh <tree> <out dir> <card> <port> <formal|mapping> <concurrency> <arm>
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4 MODE=$5 CONC=$6 ARM=$7
S=$(cd "$(dirname "$0")" && pwd)
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
SERVE_ARGS="--config examples/configs/qwen3_omni_colocated_h200.yaml --colocate --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768 ${EXTRA_SERVE_ARGS:-}"
STACK=0
case $MODE in
  mapping)
    SERVE_ARGS="$SERVE_ARGS --thinker.engine.disable_cuda_graph true --thinker.engine.cuda_graph_backend_prefill disabled --talker_ar.engine.disable_cuda_graph true --talker_ar.engine.cuda_graph_backend_prefill disabled --code2wav.factory.enable_cuda_graph false --audio_encoder.factory.enable_layer_cuda_graph false"
    STACK=1 ;;
  formal) ;;
  *) echo "mode must be formal or mapping"; exit 1 ;;
esac
PIN=${PIN_CPUS:+numactl --physcpubind=$PIN_CPUS --membind=${PIN_NODE:-0}}
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"
cd "$TREE" || exit 1

echo "start $(date +%T) card $CARD mode $MODE conc $CONC arm $ARM warm ${WARM:-32} window ${WINDOW:-48}" > "$OUT/progress.txt"
git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
echo "$SERVE_ARGS" > "$OUT/serve_args.txt"
md5sum "$0" "$S/run_bench.py" > "$OUT/md5.txt"

setsid bash -c "echo \$\$ > $OUT/server.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE \
  SGLANG_TORCH_PROFILER_WITH_STACK=$STACK SGLANG_TORCH_PROFILER_RECORD_SHAPES=1 \
  $PIN python3 -u -m sglang_omni.cli serve --model-path $MODEL $SERVE_ARGS --host 127.0.0.1 --port $PORT" > "$OUT/serve.log" 2>&1 &

healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health)" = 200 ]; then healthy=1; break; fi
  ps -o stat= -g "$(cat "$OUT/server.pgid" 2>/dev/null)" 2>/dev/null | grep -qv '^Z' || break
  sleep 5
done
echo "healthy=$healthy $(date +%T)" >> "$OUT/progress.txt"
if [ $healthy = 1 ]; then
  CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE $PIN python3 "$S/run_bench.py" gen --arm "$ARM" --port "$PORT" \
    --concurrency "$CONC" --out "$OUT/warm" --max-samples "${WARM:-32}" > "$OUT/gen_warm.log" 2>&1
  echo "warm rc $? $(date +%T)" >> "$OUT/progress.txt"
  curl -s -X POST $URL/start_profile -H 'Content-Type: application/json' \
    -d "{\"run_id\": \"trace\", \"trace_path_template\": \"$OUT/{stage}\", \"enable_torch\": true}" >> "$OUT/progress.txt"
  CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE $PIN python3 "$S/run_bench.py" gen --arm "$ARM" --port "$PORT" \
    --concurrency "$CONC" --out "$OUT/window" --max-samples "${WINDOW:-48}" > "$OUT/gen_window.log" 2>&1
  echo "window rc $? $(date +%T)" >> "$OUT/progress.txt"
  curl -s -X POST $URL/stop_profile -H 'Content-Type: application/json' -d '{}' >> "$OUT/progress.txt"
  # a trace is complete when gzip has removed its .trace.json; processes export one after
  # another, so wait until no .trace.json is left and the .gz set held for a minute
  last="" still=0
  for _ in $(seq 360); do
    sleep 5
    now=$(ls "$OUT"/*.trace.json.gz 2>/dev/null | tr '\n' ' ')
    if [ -n "$now" ] && [ "$now" = "$last" ] && ! ls "$OUT"/*.trace.json >/dev/null 2>&1; then
      still=$((still + 1))
    else
      still=0
    fi
    last=$now
    [ $still -ge 12 ] && break
  done
  echo "traces $(ls "$OUT"/*.trace.json.gz 2>/dev/null | xargs -n1 basename | tr '\n' ' ') $(date +%T)" >> "$OUT/progress.txt"
else
  echo "server not healthy" > "$OUT/FAILED"
fi
kill -TERM -- -"$(cat "$OUT/server.pgid")" 2>/dev/null
sleep 15
kill -0 -- -"$(cat "$OUT/server.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$OUT/server.pgid")"
find "$OUT" -name '*.wav' -delete
echo "done $(date +%T)" >> "$OUT/progress.txt"
touch "$OUT/DONE"
