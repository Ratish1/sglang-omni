#!/usr/bin/env bash
# Boot one tree to healthy, record startup seconds, the card's memory every second, the
# Dynamo recompile log and the capture and pool lines of serve.log, then stop the server.
# usage: run_boot_probe.sh <tree> <out dir> <card> <port> [serve args...]
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4
shift 4
EXTRA="$*"
PY=python3
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
mkdir -p "$OUT"
cd "$TREE" || exit 1
git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
nvidia-smi -i "$CARD" --query-gpu=timestamp,memory.used --format=csv,noheader -l 1 > "$OUT/mem.csv" 2>&1 &
MEM_PID=$!
#the box is shared; a foreign pid on the card voids the probe
UUID=$(nvidia-smi -i "$CARD" --query-gpu=uuid --format=csv,noheader)
(while true; do nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$UUID" | sed "s/^/$(date +%T) /"; sleep 2; done) > "$OUT/apps.csv" 2>&1 &
APPS_PID=$!
START=$(date +%s)
#PROBE_PATH adds a directory (a sitecustomize probe) behind the tree
setsid bash -c "echo \$\$ > $OUT/server.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE${PROBE_PATH:+:$PROBE_PATH} \
  TORCH_LOGS=recompiles $PY -u -m sglang_omni.cli serve --model-path $MODEL --port $PORT $EXTRA" > "$OUT/serve.log" 2>&1 &
healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  kill -0 -- -"$(cat "$OUT/server.pgid" 2>/dev/null)" 2>/dev/null || break
  sleep 5
done
echo "healthy=$healthy startup_s=$(( $(date +%s) - START ))" > "$OUT/result.txt"
kill -TERM -- -"$(cat "$OUT/server.pgid")" 2>/dev/null
sleep 12
kill -0 -- -"$(cat "$OUT/server.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$OUT/server.pgid")"
kill "$MEM_PID" "$APPS_PID" 2>/dev/null
echo "pids seen on the card: $(awk '{print $3}' "$OUT/apps.csv" | sort -u | tr '\n' ' ')" >> "$OUT/result.txt"
sort -t, -k2 -n "$OUT/mem.csv" | tail -1 >> "$OUT/result.txt"
grep -E "fused SnakeBeta|incremental Codec graph|Codec graphs captured|max_total_num_tokens|KV Cache is allocated|avail mem|Recompil|recompile|capture disabled|Traceback|Error" "$OUT/serve.log" | cut -c1-400 > "$OUT/lines.txt"
echo done >> "$OUT/result.txt"
