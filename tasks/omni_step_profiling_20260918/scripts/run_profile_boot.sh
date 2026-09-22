#!/usr/bin/env bash
# One profiling boot of a Qwen3-TTS tree: server, GPU memory log, the fixed capture list,
# the step ledger on every trace, teardown. Same list for every arm.
# usage: run_profile_boot.sh <tree> <out dir> <card> <port>
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4
S=$(cd "$(dirname "$0")" && pwd)
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"
cd "$TREE" || exit 1

git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
git -C "$TREE" diff --stat > "$OUT/tree_stat.txt"
PYTHONPATH=$TREE python3 -c "import sglang_omni; print(sglang_omni.__file__)" > "$OUT/import_path.txt"
md5sum "$S/profile_workloads.py" "$S/step_ledger.py" "$0" > "$OUT/md5.txt"
nvidia-smi > "$OUT/gpus_before.txt"
nvidia-smi -i "$CARD" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$OUT/mem.csv" 2>&1 &
MEM_PID=$!

setsid bash -c "echo \$\$ > $OUT/server.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE \
  python3 -u -m sglang_omni.cli serve --model-path $MODEL --port $PORT" > "$OUT/serve.log" 2>&1 &

teardown() {
  kill -TERM -- -"$(cat "$OUT/server.pgid")" 2>/dev/null
  sleep 10
  kill -0 -- -"$(cat "$OUT/server.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$OUT/server.pgid")"
  kill "$MEM_PID" 2>/dev/null
  sleep 5
  nvidia-smi > "$OUT/gpus_after.txt"
}

healthy=0
for _ in $(seq 90); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health)" = 200 ]; then healthy=1; break; fi
  sleep 10
done
if [ $healthy = 0 ]; then
  echo "server not healthy after 15 min" > "$OUT/FAILED"
  teardown
  exit 1
fi
echo "healthy $(date +%T)" > "$OUT/progress.txt"

capture() {
  local name=$1; shift
  echo "start $name $(date +%T)" >> "$OUT/progress.txt"
  PYTHONPATH=$TREE python3 "$S/profile_workloads.py" --url $URL --model $MODEL --out "$OUT" "$@" \
    > "$OUT/driver_$name.json" 2> "$OUT/driver_$name.err"
  echo "end $name rc=$? $(date +%T)" >> "$OUT/progress.txt"
}

capture uncaptured_decode_b16_1 --kind decode --batch 16 --steps 40 --label uncaptured1 --no-capture
capture uncaptured_decode_b1 --kind decode --batch 1 --steps 40 --label uncaptured_b1 --no-capture
capture formal_decode_b1 --kind decode --batch 1 --steps 40 --label formal
capture formal_decode_b2 --kind decode --batch 2 --steps 40 --label formal
capture formal_decode_b4 --kind decode --batch 4 --steps 40 --label formal
capture formal_decode_b8 --kind decode --batch 8 --steps 40 --label formal
capture formal_decode_b16 --kind decode --batch 16 --steps 40 --label formal
capture uncaptured_decode_b16_2 --kind decode --batch 16 --steps 40 --label uncaptured2 --no-capture
capture formal_prefill_b1 --kind prefill --batch 1 --steps 10 --label formal
capture formal_prefill_b8 --kind prefill --batch 8 --steps 8 --label formal
capture mapping_decode_b16 --kind decode --batch 16 --steps 20 --label mapping --with-stack
capture mapping_prefill_b1 --kind prefill --batch 1 --steps 10 --label mapping --with-stack

grep -m1 "Torch profiler armed" "$OUT/serve.log" > "$OUT/armed_marker.txt"
teardown

for trace in "$OUT"/formal/*/b*/*.trace.json.gz "$OUT"/mapping/*/b*/*.trace.json.gz; do
  [ -e "$trace" ] || continue
  python3 "$S/step_ledger.py" "$trace" --top 30 > "$(dirname "$trace")/ledger.txt" 2>&1
done
echo "done $(date +%T)" >> "$OUT/progress.txt"
