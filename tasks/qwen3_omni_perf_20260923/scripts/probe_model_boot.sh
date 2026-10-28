#!/usr/bin/env bash
# One boot of any model with the IPC payload probe on: the server (its own process group, the
# probe on its PYTHONPATH, OMNI_IPC_PROBE=<out>/ipc), one client command, stop by the process
# group, then ipc_probe_report.py and a per-process summary of every direct CUDA IPC payload:
# calls, header bytes, the CPU tensors' own bytes and the bytes of the storages they keep.
# The health wait allows 60 minutes, as the first boot downloads the checkpoint.
# usage: probe_model_boot.sh <tree> <out dir> <card> <port> <model> "<serve args>" "<client command>"
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4 MODEL=$5 SERVE_ARGS=$6 CLIENT=$7
S=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/ipc"
OUT=$(cd "$OUT" && pwd)
cd "$TREE" || exit 1
echo "start $(date +%T) card $CARD model $MODEL" > "$OUT/progress.txt"
git -C "$TREE" rev-parse HEAD HEAD:sglang_omni > "$OUT/head.txt"
echo "$SERVE_ARGS" > "$OUT/serve_args.txt"
echo "$CLIENT" > "$OUT/client.txt"
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  echo "port $PORT already in use" > "$OUT/FAILED"
  exit 1
fi
setsid bash -c "echo \$\$ > $OUT/serve.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD OMNI_IPC_PROBE=$OUT/ipc PYTHONPATH=$S/ipc_payload_probe:$TREE \
  python3 -u -m sglang_omni.cli serve --model-path $MODEL $SERVE_ARGS --host 127.0.0.1 --port $PORT" > "$OUT/serve.log" 2>&1 &
began=$(date +%s) healthy=0
for _ in $(seq 720); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  ps -o stat= -g "$(cat "$OUT/serve.pgid" 2>/dev/null)" 2>/dev/null | grep -qv '^Z' || break
  sleep 5
done
echo "healthy=$healthy startup_s $(( $(date +%s) - began )) $(date +%T)" >> "$OUT/progress.txt"
if [ $healthy = 1 ]; then
  echo "client start $(date +%T)" >> "$OUT/progress.txt"
  (cd "$TREE" && CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE bash -c "$CLIENT") > "$OUT/client.log" 2>&1
  echo "client rc $? $(date +%T)" >> "$OUT/progress.txt"
else
  echo "server not healthy" > "$OUT/FAILED"
fi
kill -TERM -- -"$(cat "$OUT/serve.pgid")" 2>/dev/null
sleep 15
kill -0 -- -"$(cat "$OUT/serve.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$OUT/serve.pgid")"
for _ in $(seq 60); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD")" -lt 100 ] && break
  sleep 2
done
python3 "$S/ipc_probe_report.py" "$OUT/ipc" > "$OUT/ipc_report.txt" 2>&1
python3 - "$OUT/ipc" > "$OUT/ipc_summary.txt" 2>&1 <<'PY'
import collections, glob, json, os, sys
rows = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
for path in glob.glob(os.path.join(sys.argv[1], "ipc_*.jsonl")):
    for line in open(path):
        record = json.loads(line)
        if record.get("kind") != "serialize":
            continue
        row = rows[record.get("process", "?")]
        row[0] += 1
        row[1] += record.get("header_bytes", 0)
        row[2] += record.get("cpu_tensor_bytes", 0)
        row[3] += record.get("cpu_storage_bytes", 0)
        row[4] = max(row[4], record.get("cpu_storage_bytes", 0) - record.get("cpu_tensor_bytes", 0))
print("process calls header_bytes cpu_tensor_bytes cpu_storage_bytes max_excess_per_call")
for process, (calls, header, own, storage, excess) in sorted(rows.items()):
    print(process, calls, header, own, storage, excess)
if not rows:
    print("no direct CUDA IPC payload serializations")
PY
echo "done $(date +%T)" >> "$OUT/progress.txt"
touch "$OUT/DONE"
