#!/usr/bin/env bash
# Nsight Systems pass over a Qwen3-TTS tree: C6 probe (GPU metrics in this container),
# then a 5 s window of steady decode bs 16 with CUDA graph nodes, OS runtime and GPU
# metrics, then the summary reports and a sqlite export, all on the box.
# usage: run_nsys_boot.sh <tree> <out dir> <card> <port>
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4
S=/workspace/sglang-omni/.tmp/omni_step_profiling/scripts
MODEL=/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base
URL=http://127.0.0.1:$PORT
SESSION=omni$PORT
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$OUT"
cd "$TREE" || exit 1
md5sum "$0" "$S/profile_workloads.py" > "$OUT/md5.txt"

CUDA_VISIBLE_DEVICES=$CARD nsys profile -o "$OUT/c6_probe" --force-overwrite=true \
  --gpu-metrics-devices="$CARD" --gpu-metrics-set=ad10x \
  python -c "import torch; x = torch.randn(4096, 4096, device='cuda'); [x @ x for _ in range(200)]; torch.cuda.synchronize()" \
  > "$OUT/c6_probe.log" 2>&1
echo "c6 probe rc=$?" > "$OUT/progress.txt"
nsys export --type sqlite --force-overwrite=true -o "$OUT/c6_probe.sqlite" "$OUT/c6_probe.nsys-rep" > "$OUT/c6_export.log" 2>&1
python - "$OUT/c6_probe.sqlite" > "$OUT/c6_metrics.txt" 2>&1 <<'PY'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
print("GPU_METRICS table:", "GPU_METRICS" in tables)
if "GPU_METRICS" in tables:
    print("rows:", db.execute("select count(*) from GPU_METRICS").fetchone()[0])
    names = [r[0] for r in db.execute("select distinct metricName from TARGET_INFO_GPU_METRICS")]
    print("metrics:", names)
PY

# note(ratish): sampling and context-switch switches belong to nsys start, not launch.
setsid bash -c "echo \$\$ > $OUT/server.pgid; exec nsys launch --session-new=$SESSION \
  --trace=cuda,nvtx,osrt,cudnn,cublas --cuda-graph-trace=node \
  env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE python -u -m sglang_omni.cli serve \
  --model-path $MODEL --port $PORT" > "$OUT/serve.log" 2>&1 &
LAUNCH_PID=$!

teardown() {
  nsys shutdown --session="$SESSION" > "$OUT/nsys_shutdown.log" 2>&1
  kill -TERM -- -"$(cat "$OUT/server.pgid")" 2>/dev/null
  sleep 10
  kill -0 -- -"$(cat "$OUT/server.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$OUT/server.pgid")"
  sleep 5
  nvidia-smi > "$OUT/gpus_after.txt"
}

healthy=0
for _ in $(seq 120); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health)" = 200 ]; then healthy=1; break; fi
  kill -0 $LAUNCH_PID 2>/dev/null || break
  sleep 10
done
if [ $healthy = 0 ]; then
  echo "server not healthy after 20 min" > "$OUT/FAILED"
  teardown
  exit 1
fi
echo "healthy $(date +%T)" >> "$OUT/progress.txt"

PYTHONPATH=$TREE python "$S/profile_workloads.py" --url $URL --model $MODEL --out "$OUT" \
  --kind decode --batch 16 --steps 40 --label nsys_window --no-capture \
  > "$OUT/driver_nsys_window.json" 2> "$OUT/driver_nsys_window.err" &
DRIVER_PID=$!
sleep 20
nsys start --session="$SESSION" -o "$OUT/nsys_decode_b16" --force-overwrite=true \
  --sample=none --cpuctxsw=none \
  --gpu-metrics-devices="$CARD" --gpu-metrics-set=ad10x > "$OUT/nsys_start.log" 2>&1
START_RC=$?
if [ $START_RC != 0 ]; then
  echo "nsys start with gpu metrics rc=$START_RC, retrying without" >> "$OUT/progress.txt"
  nsys start --session="$SESSION" -o "$OUT/nsys_decode_b16" --force-overwrite=true \
    --sample=none --cpuctxsw=none >> "$OUT/nsys_start.log" 2>&1
fi
sleep 5
nsys stop --session="$SESSION" > "$OUT/nsys_stop.log" 2>&1
echo "window captured $(date +%T)" >> "$OUT/progress.txt"
wait $DRIVER_PID
teardown

REP="$OUT/nsys_decode_b16.nsys-rep"
nsys stats --force-export=true --format csv --output "$OUT/stats" \
  --report cuda_gpu_kern_sum,cuda_api_sum,osrt_sum,nvtx_sum,cuda_gpu_mem_time_sum "$REP" > "$OUT/stats.log" 2>&1
nsys export --type sqlite --force-overwrite=true -o "$OUT/nsys_decode_b16.sqlite" "$REP" > "$OUT/export.log" 2>&1
echo "done $(date +%T)" >> "$OUT/progress.txt"
