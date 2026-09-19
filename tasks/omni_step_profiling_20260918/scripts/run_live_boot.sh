#!/usr/bin/env bash
# One profiling boot of a Qwen3-TTS tree with the profiler: the degenerate captures
# (prefill-only, decode-only), then windows under real load. A live cell is one benchmark
# run in the background with a torch window armed 15 s after its timed requests start:
#   events pass   request event recorder only (no torch), the unprofiled client numbers
#   formal pass   a window of scheduler forwards, no stack (timing)
#   roles pass    a short with-stack window (names the threads; never timed)
# Cells: seed-tts stream c16, long-form stream c16, seed-tts stream c1.
# usage: run_live_boot.sh <tree> <out dir> <card> <port>
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4
S=$(cd "$(dirname "$0")" && pwd)
PY=/workspace/sglang-omni/.venv/bin/python
MODEL=/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base
META=zhaochenyang20/seed-tts-eval-arrow
URL=http://127.0.0.1:$PORT
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$OUT"
cd "$TREE" || exit 1

git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
PYTHONPATH=$TREE $PY -c "import sglang_omni; print(sglang_omni.__file__)" > "$OUT/import_path.txt"
md5sum "$S"/*.py "$0" > "$OUT/md5.txt"
nvidia-smi > "$OUT/gpus_before.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> "$OUT/gpus_before.txt"
nvidia-smi dmon -s pucvm -d 1 > "$OUT/dmon_all_cards.log" 2>&1 &
DMON_PID=$!
PYTHONPATH=$TREE $PY "$S/make_longform_meta.py" --out "$OUT/longform" > "$OUT/longform.log" 2>&1

setsid bash -c "echo \$\$ > $OUT/server.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE \
  $PY -u -m sglang_omni.cli serve --model-path $MODEL --port $PORT" > "$OUT/serve.log" 2>&1 &

teardown() {
  kill -TERM -- -"$(cat "$OUT/server.pgid")" 2>/dev/null
  sleep 10
  kill -0 -- -"$(cat "$OUT/server.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$OUT/server.pgid")"
  kill "$DMON_PID" 2>/dev/null
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
  PYTHONPATH=$TREE $PY "$S/profile_workloads.py" --url $URL --model $MODEL --out "$OUT" "$@" \
    > "$OUT/driver_$name.json" 2> "$OUT/driver_$name.err"
  echo "end $name rc=$? $(date +%T)" >> "$OUT/progress.txt"
}

bench() {
  local dir=$1; shift
  PYTHONPATH=$TREE $PY -m benchmarks.eval.benchmark_tts_seedtts --model $MODEL --lang en \
    --use-existing-server --host 127.0.0.1 --port $PORT --warmup 1 --stream --generate-only \
    --output-dir "$dir/bench" "$@" > "$dir/bench.log" 2>&1
}

# live <cell> <formal steps> <bench args...>
live() {
  local cell=$1 steps=$2; shift 2
  local d=$OUT/live/$cell
  mkdir -p "$d/events_pass" "$d/formal_pass" "$d/roles_pass"
  echo "start live $cell events $(date +%T)" >> "$OUT/progress.txt"
  curl -s -X POST $URL/start_request_profile -H 'content-type: application/json' \
    -d "{\"run_id\":\"$cell\",\"event_dir\":\"$d/events_pass/events\"}" > "$d/events_pass/start.json"
  bench "$d/events_pass" "$@"
  curl -s -X POST $URL/stop_request_profile -H 'content-type: application/json' \
    -d "{\"run_id\":\"$cell\"}" > "$d/events_pass/stop.json"
  PYTHONPATH=$TREE $PY -m sglang_omni.profiler "$d/events_pass/events" --format table \
    > "$d/events_pass/report.txt" 2> "$d/events_pass/report.err"
  for pass in formal roles; do
    echo "start live $cell $pass $(date +%T)" >> "$OUT/progress.txt"
    bench "$d/${pass}_pass" "$@" &
    local bench_pid=$!
    until grep -q "Benchmarking" "$d/${pass}_pass/bench.log" 2>/dev/null; do
      kill -0 $bench_pid 2>/dev/null || break
      sleep 1
    done
    sleep 15
    if [ $pass = formal ]; then
      $PY "$S/live_window.py" --url $URL --out "$d/formal_pass" --label window --steps "$steps" \
        > "$d/formal_pass/window.out" 2> "$d/formal_pass/window.err"
    else
      $PY "$S/live_window.py" --url $URL --out "$d/roles_pass" --label window --steps 60 --with-stack \
        > "$d/roles_pass/window.out" 2> "$d/roles_pass/window.err"
    fi
    wait $bench_pid
    echo "end live $cell $pass rc=$? $(date +%T)" >> "$OUT/progress.txt"
  done
}

capture formal_prefill_b1 --kind prefill --batch 1 --steps 10 --label formal
capture formal_prefill_b16 --kind prefill --batch 16 --steps 16 --label formal
capture formal_decode_b1 --kind decode --batch 1 --steps 40 --label formal
capture formal_decode_b16 --kind decode --batch 16 --steps 40 --label formal
capture uncaptured_decode_b16 --kind decode --batch 16 --steps 40 --label uncaptured --no-capture

live seedtts_c16 600 --meta $META --concurrency 16
live longform_c16 400 --meta "$OUT/longform/meta.lst" --concurrency 16
live seedtts_c1 300 --meta $META --concurrency 1 --max-samples 120

grep -m1 "Torch profiler armed" "$OUT/serve.log" > "$OUT/armed_marker.txt"
teardown

for trace in "$OUT"/formal/*/b*/*.trace.json.gz; do
  [ -e "$trace" ] || continue
  $PY "$S/step_ledger.py" "$trace" --top 30 > "$(dirname "$trace")/ledger.txt" 2>&1
done
for cell in "$OUT"/live/*; do
  formal=$(ls "$cell"/formal_pass/window/*.trace.json.gz 2>/dev/null | head -1)
  roles=$(ls "$cell"/roles_pass/window/*.trace.json.gz 2>/dev/null | head -1)
  [ -n "$formal" ] || continue
  $PY "$S/window_budget.py" "$formal" ${roles:+--roles "$roles"} > "$cell/budget.txt" 2>&1
done
echo "done $(date +%T)" >> "$OUT/progress.txt"
