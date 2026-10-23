#!/usr/bin/env bash
# run_probe_boot.sh for the Qwen3-Omni speech default instead of the colocated profile: the
# thinker on card 0, the talker on card 1, code2wav on card 0 in a process of its own. The
# container's two cards are both used and CUDA_VISIBLE_DEVICES is not set. A cell refuses to
# boot while another process holds memory on either card, since a profile needs the cards to
# itself. Same probe, nsys flags, warm pass, window and census as run_probe_boot.sh, then the
# talker readers (talker_step_classes.py, talker_gap_split.py, chunk_gap_origin.py) on the
# same report. NSYS_ARGS=none serves the same cell clean (no probe, no profiler).
# EXTRA_SERVE_ARGS is appended to the serve args (for example the talker's running cap).
# usage: run_probe_boot_speech.sh <tree> <out dir> <port> <concurrency> <arm>
set -u
TREE=$1 OUT=$2 PORT=$3 CONC=$4 ARM=$5
S=$(cd "$(dirname "$0")" && pwd)
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
NSYS_ARGS=${NSYS_ARGS:---trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=node --gpuctxsw=true --sample=none --cpuctxsw=none}
if [ "$NSYS_ARGS" = none ]; then
  WRAP="" PROBE=""
else
  WRAP="nsys profile -o $OUT/serve --force-overwrite=true $NSYS_ARGS"
  PROBE="OMNI_PIPE_NVTX=1 OMNI_PIPE_LINES=${LINES:-}"
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$TREE" || exit 1
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then
  echo "port $PORT already serves" > "$OUT/FAILED"
  exit 1
fi
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 > 1024 {print $1 ":" $2 "MiB"}')
if [ -n "$busy" ]; then
  echo "cards busy before boot: $busy" > "$OUT/FAILED"
  nvidia-smi > "$OUT/gpus_before.txt"
  exit 1
fi

echo "start $(date +%T) speech default conc $CONC arm $ARM samples ${MAX_SAMPLES:-128} extra '${SERVE_ARGS}'" > "$OUT/progress.txt"
git -C "$TREE" rev-parse HEAD HEAD:sglang_omni > "$OUT/head.txt"
git -C "$TREE" status --short > "$OUT/tree_status.txt"
md5sum "$0" "$S/run_bench.py" "$S/omni_census.py" "$S/omni_pipeline_nvtx/sitecustomize.py" > "$OUT/md5.txt"
nsys --version > "$OUT/nsys_version.txt" 2>&1
echo "$SERVE_ARGS | $NSYS_ARGS | $PROBE" > "$OUT/serve_args.txt"
nvidia-smi > "$OUT/gpus_before.txt"
cat /proc/loadavg > "$OUT/loadavg_before.txt"
(while true; do cat /proc/loadavg; sleep 30; done) > "$OUT/loadavg.log" 2>&1 &
LOADS=$!
nvidia-smi dmon -s pucm -d 1 > "$OUT/dmon.log" 2>&1 &
DMON=$!
(while true; do nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | sed "s/^/$(date +%T) /"; sleep 10; done) > "$OUT/apps.csv" 2>&1 &
APPS=$!

env PYTHONPATH=$TREE${PROBE:+:$S/omni_pipeline_nvtx} $PROBE \
  setsid $WRAP python3 -u -m sglang_omni.cli serve --model-path $MODEL $SERVE_ARGS --host 127.0.0.1 --port $PORT \
  > "$OUT/serve.log" 2>&1 &
WRAP_PID=$!
echo "pgid $WRAP_PID" >> "$OUT/progress.txt"

healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  kill -0 $WRAP_PID 2>/dev/null || break
  sleep 5
done
if [ $healthy = 1 ]; then
  echo "healthy $(date +%T)" >> "$OUT/progress.txt"
  PYTHONPATH=$TREE python3 "$S/run_bench.py" gen --arm "$ARM" --port "$PORT" \
    --concurrency "$CONC" --out "$OUT/warm" --max-samples "${WARM_SAMPLES:-32}" > "$OUT/warm_$ARM.log" 2>&1
  echo "warm rc $? $(date +%T)" >> "$OUT/progress.txt"
  date +%s.%N > "$OUT/window.txt"
  PYTHONPATH=$TREE python3 "$S/run_bench.py" gen --arm "$ARM" --port "$PORT" \
    --concurrency "$CONC" --out "$OUT" --max-samples "${MAX_SAMPLES:-128}" \
    --skip-samples "${WARM_SAMPLES:-32}" > "$OUT/gen_$ARM.log" 2>&1
  echo "gen rc $? $(date +%T)" >> "$OUT/progress.txt"
  date +%s.%N >> "$OUT/window.txt"
else
  echo "server not healthy" > "$OUT/FAILED"
fi

descendants() {
  local child
  for child in $(pgrep -P "$1"); do echo "$child"; descendants "$child"; done
}
SERVE_PID=$(for p in $WRAP_PID $(descendants "$WRAP_PID"); do
  if [[ "$(cat /proc/$p/comm 2>/dev/null)" == python* ]] && tr '\0' ' ' < /proc/$p/cmdline | grep -qE -- "--port $PORT( |$)"; then echo $p; fi
done | head -1)
echo "serve pid $SERVE_PID" >> "$OUT/progress.txt"
[ -n "$SERVE_PID" ] && kill -TERM "$SERVE_PID"
for _ in $(seq 120); do
  kill -0 $WRAP_PID 2>/dev/null || break
  sleep 5
done
if kill -0 $WRAP_PID 2>/dev/null || { [ -n "$SERVE_PID" ] && kill -0 "$SERVE_PID" 2>/dev/null; }; then
  echo "serve or wrapper still running after 10 min; cell void" >> "$OUT/progress.txt"
  echo "serve not stopped" > "$OUT/FAILED"
fi
echo "serve ended $(date +%T)" >> "$OUT/progress.txt"
kill $LOADS $DMON $APPS 2>/dev/null
nvidia-smi > "$OUT/gpus_after.txt"
cat /proc/loadavg > "$OUT/loadavg_after.txt"
grep -h "omni_pipeline_nvtx patch" "$OUT/serve.log" > "$OUT/probe_patch_errors.txt" 2>/dev/null
find "$OUT" -name '*.wav' -delete
[ -f "$OUT/FAILED" ] && exit 1

if [ "$NSYS_ARGS" != none ]; then
  (
    nsys export --type sqlite --force-overwrite=true -o "$OUT/serve.sqlite" "$OUT/serve.nsys-rep" > "$OUT/export.log" 2>&1
    echo "exported $(date +%T)" >> "$OUT/progress.txt"
    cd "$S" || exit 1
    python3 omni_census.py "$OUT/serve.sqlite" --window "$OUT/window.txt" > "$OUT/census.txt" 2>&1
    python3 talker_step_classes.py "$OUT/serve.sqlite" > "$OUT/talker_step_classes.txt" 2>&1
    python3 talker_gap_split.py "$OUT/serve.sqlite" --window "$OUT/window.txt" > "$OUT/talker_gap_split.txt" 2>&1
    python3 chunk_gap_origin.py "$OUT/serve.sqlite" --window "$OUT/window.txt" > "$OUT/chunk_gap_origin.txt" 2>&1
    echo "census done $(date +%T)" >> "$OUT/progress.txt"
    touch "$OUT/CENSUS_DONE"
  ) &
fi
echo "done $(date +%T)" >> "$OUT/progress.txt"
touch "$OUT/DONE"
