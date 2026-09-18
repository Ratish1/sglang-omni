#!/usr/bin/env bash
# Nsight Systems pass over a Qwen3-TTS tree: the whole server under nsys profile, with
# collection opening after DELAY seconds for DURATION seconds while bs 16 decode load runs
# continuously from the moment the server is healthy. CUDA kernels (graph nodes
# included), OS runtime, and GPU metrics (ad10x) on the card. Summary reports and a sqlite
# export are written on the box. The C6 probe showed nsys profile collects kernels and
# metrics in this container; launch/start sessions collected neither.
# usage: run_nsys_boot.sh <tree> <out dir> <card> <port> [delay_s] [duration_s]
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4 DELAY=${5:-240} DURATION=${6:-6}
S=/workspace/sglang-omni/.tmp/omni_step_profiling/scripts
MODEL=/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base
URL=http://127.0.0.1:$PORT
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$OUT"
cd "$TREE" || exit 1
md5sum "$0" "$S/profile_workloads.py" > "$OUT/md5.txt"
echo "start $(date +%T) delay $DELAY duration $DURATION" > "$OUT/progress.txt"

setsid bash -c "echo \$\$ > $OUT/server.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE \
  nsys profile -o $OUT/nsys_decode_b16 --force-overwrite=true \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --gpu-metrics-devices=$CARD --gpu-metrics-set=ad10x \
  --delay=$DELAY --duration=$DURATION \
  python -u -m sglang_omni.cli serve --model-path $MODEL --port $PORT" > "$OUT/serve.log" 2>&1 &
NSYS_PID=$!

healthy=0
for _ in $(seq 120); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health)" = 200 ]; then healthy=1; break; fi
  kill -0 $NSYS_PID 2>/dev/null || break
  sleep 5
done
if [ $healthy = 0 ]; then
  echo "server not healthy" > "$OUT/FAILED"
  kill -TERM -- -"$(cat "$OUT/server.pgid")" 2>/dev/null
  exit 1
fi
echo "healthy $(date +%T)" >> "$OUT/progress.txt"

rounds=0
while kill -0 $NSYS_PID 2>/dev/null && [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health)" = 200 ]; do
  PYTHONPATH=$TREE python "$S/profile_workloads.py" --url $URL --model $MODEL --out "$OUT" \
    --kind decode --batch 16 --steps 40 --label nsys_load --no-capture \
    >> "$OUT/driver_nsys_load.jsonl" 2>> "$OUT/driver_nsys_load.err"
  rounds=$((rounds + 1))
done
echo "load rounds $rounds, nsys ended $(date +%T)" >> "$OUT/progress.txt"

wait $NSYS_PID 2>/dev/null
kill -TERM -- -"$(cat "$OUT/server.pgid")" 2>/dev/null
sleep 10
kill -0 -- -"$(cat "$OUT/server.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$OUT/server.pgid")"
sleep 5
nvidia-smi > "$OUT/gpus_after.txt"

REP="$OUT/nsys_decode_b16.nsys-rep"
nsys stats --force-export=true --format csv --output "$OUT/stats" \
  --report cuda_gpu_kern_sum,cuda_api_sum,osrt_sum,cuda_gpu_mem_time_sum "$REP" > "$OUT/stats.log" 2>&1
nsys export --type sqlite --force-overwrite=true -o "$OUT/nsys_decode_b16.sqlite" "$REP" > "$OUT/export.log" 2>&1
python "$S/nsys_metrics.py" "$OUT/nsys_decode_b16.sqlite" > "$OUT/metrics.txt" 2>&1
echo "done $(date +%T)" >> "$OUT/progress.txt"
