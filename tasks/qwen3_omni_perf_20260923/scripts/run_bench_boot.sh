#!/usr/bin/env bash
# One benchmark boot of a Qwen3-Omni tree on one card. Provenance, the colocated server,
# every arm of ARMS one after another (run_bench.py gen), stop by the server's own process
# group; then, with SCORE=1, a Qwen3-ASR server on the same card and run_bench.py score
# for every arm that has speech. pids seen on the card are logged every 2 s; a second pid
# while the omni server runs voids the boot.
# MAX_SAMPLES=N runs the first N samples of every arm (identity smokes); unset is the full corpus.
# minicpmo serves MiniCPM-o-4.5 with the Omni CI speech worker args (tests/test_model/conftest.py).
# SERVER_PYTHONPATH is prepended for the omni server only (memdiag/ loads the memory diagnostics).
# usage: run_bench_boot.sh <tree> <out dir> <card> <port> <bf16|fp8|minicpmo> <concurrency> "<arm> <arm> ..."
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4 DTYPE=$5 CONC=$6 ARMS=$7
S=$(cd "$(dirname "$0")" && pwd)
QWEN_ARGS="--colocate --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
case $DTYPE in
  bf16) MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
    SERVE_ARGS="--config examples/configs/qwen3_omni_colocated_h100_bf16.yaml $QWEN_ARGS" ;;
  fp8) MODEL=marksverdhei/Qwen3-Omni-30B-A3B-FP8
    SERVE_ARGS="--config examples/configs/qwen3_omni_colocated_h100_fp8.yaml $QWEN_ARGS" ;;
  minicpmo) MODEL=openbmb/MiniCPM-o-4_5
    SERVE_ARGS="--thinker.factory.max_seq_len 8192 --thinker.engine.mem_fraction_static 0.55 --talker.engine.mem_fraction_static 0.15" ;;
  *) echo "dtype must be bf16, fp8 or minicpmo"; exit 1 ;;
esac
SERVE_ARGS="$SERVE_ARGS ${EXTRA_SERVE_ARGS:-}"
ASR_MODEL=Qwen/Qwen3-ASR-1.7B
# PIN_CPUS (e.g. 0-15,64-79) runs the server, the benchmark client and the scorer on those
# cores with memory on NUMA node 0, so the two arms of a pair get identical, disjoint CPU
PIN=${PIN_CPUS:+numactl --physcpubind=$PIN_CPUS --membind=0}
mkdir -p "$OUT"
cd "$TREE" || exit 1

echo "start $(date +%T) card $CARD dtype $DTYPE conc $CONC arms: $ARMS" > "$OUT/progress.txt"
git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
git -C "$TREE" status --short > "$OUT/tree_status.txt"
CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE python3 -c "import sglang_omni, sglang; print(sglang_omni.__file__, sglang.__version__)" > "$OUT/import_path.txt" 2>&1
md5sum "$S/run_bench.py" "$0" > "$OUT/md5.txt"
echo "$SERVE_ARGS" > "$OUT/serve_args.txt"
echo "cpu pin: ${PIN:-none}" >> "$OUT/progress.txt"
nvidia-smi > "$OUT/gpus_before.txt"
nvidia-smi dmon -i "$CARD" -s pucvm -d 1 > "$OUT/dmon.log" 2>&1 &
DMON_PID=$!
uuid=$(nvidia-smi -i "$CARD" --query-gpu=uuid --format=csv,noheader)
(while true; do nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$uuid" | sed "s/^/$(date +%T) /"; sleep 2; done) > "$OUT/apps.csv" 2>&1 &
APPS_PID=$!
(while true; do echo "$(date +%T) $(cat /proc/loadavg) | $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"; sleep 30; done) > "$OUT/host_load.txt" 2>&1 &
LOAD_PID=$!

# serve <label> <model> <port> <args...>: starts a server in its own process group
serve() {
  local label=$1 model=$2 port=$3; shift 3
  setsid bash -c "echo \$\$ > $OUT/$label.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=${SERVER_PYTHONPATH:+$SERVER_PYTHONPATH:}$TREE \
    $PIN python3 -u -m sglang_omni.cli serve --model-path $model $* --host 127.0.0.1 --port $port" > "$OUT/$label.log" 2>&1 &
  local began=$(date +%s) healthy=0
  for _ in $(seq 360); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; then healthy=1; break; fi
    # the container's pid 1 never reaps, so a dead server group is all zombies
    ps -o stat= -g "$(cat "$OUT/$label.pgid" 2>/dev/null)" 2>/dev/null | grep -qv '^Z' || break
    sleep 5
  done
  echo "$label healthy=$healthy startup_s $(( $(date +%s) - began )) $(date +%T)" >> "$OUT/progress.txt"
  [ $healthy = 1 ]
}

stop() {
  local label=$1
  kill -TERM -- -"$(cat "$OUT/$label.pgid")" 2>/dev/null
  sleep 15
  kill -0 -- -"$(cat "$OUT/$label.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$OUT/$label.pgid")"
  for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD")" -lt 100 ] && break
    sleep 2
  done
  echo "$label stopped, card at $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$CARD") $(date +%T)" >> "$OUT/progress.txt"
}

if serve serve "$MODEL" "$PORT" $SERVE_ARGS; then
  # EVENTS=1 records request events (no torch trace) over the timed arms into $OUT/events
  [ "${EVENTS:-0}" = 1 ] && curl -s -X POST "http://127.0.0.1:$PORT/start_request_profile" \
    -H 'Content-Type: application/json' -d "{\"run_id\": \"bench\", \"event_dir\": \"$OUT/events\"}" >> "$OUT/progress.txt"
  for arm in $ARMS; do
    echo "gen $arm start $(date +%T)" >> "$OUT/progress.txt"
    CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE $PIN python3 "$S/run_bench.py" gen --arm "$arm" --port "$PORT" --concurrency "$CONC" --out "$OUT" ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES} \
      > "$OUT/gen_$arm.log" 2>&1
    echo "gen $arm rc $? $(date +%T)" >> "$OUT/progress.txt"
  done
  if [ "${EVENTS:-0}" = 1 ]; then
    curl -s -X POST "http://127.0.0.1:$PORT/stop_request_profile" -H 'Content-Type: application/json' -d '{}' >> "$OUT/progress.txt"
    python3 "$S/events_first_audio.py" "$OUT/events" > "$OUT/first_audio.txt" 2>&1
  fi
else
  echo "server not healthy" > "$OUT/FAILED"
fi
stop serve
echo "pids on the card: $(awk '{print $3}' "$OUT/apps.csv" | sort -u | tr '\n' ' ')" >> "$OUT/progress.txt"

SPEECH=""
for arm in $ARMS; do
  case $arm in seedtts_en|*_talker) [ -d "$OUT/$arm" ] && SPEECH="$SPEECH $arm" ;; esac
done
if [ "${SCORE:-0}" = 1 ] && [ -n "$SPEECH" ] && [ ! -f "$OUT/FAILED" ]; then
  if serve asr "$ASR_MODEL" $((PORT + 100)); then
    for arm in $SPEECH; do
      echo "score $arm start $(date +%T)" >> "$OUT/progress.txt"
      CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE $PIN python3 "$S/run_bench.py" score --arm "$arm" --asr-port $((PORT + 100)) --out "$OUT" \
        > "$OUT/score_$arm.log" 2>&1
      echo "score $arm rc $? $(date +%T)" >> "$OUT/progress.txt"
    done
  fi
  stop asr
  # the similarity model needs the card the ASR server held
  if [ -d "$OUT/seedtts_en" ]; then
    CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE $PIN python3 "$S/run_bench.py" sim --arm seedtts_en --out "$OUT" \
      > "$OUT/sim_seedtts_en.log" 2>&1
    echo "sim seedtts_en rc $? $(date +%T)" >> "$OUT/progress.txt"
  fi
  find "$OUT" -name '*.wav' -path '*/audio/*' -delete
fi
kill "$DMON_PID" "$APPS_PID" "$LOAD_PID" 2>/dev/null
echo "done $(date +%T)" >> "$OUT/progress.txt"
touch "$OUT/DONE"
