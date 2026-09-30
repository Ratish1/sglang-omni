#!/usr/bin/env bash
# Probed Nsight Systems pass over one colocated Qwen3-Omni server on any tree: the
# omni_pipeline_nvtx probe (sitecustomize, no runtime changes), node-traced graphs, GPU
# context switches, the GIL and OS runtime, one run_bench.py arm as the load, then
# omni_census.py over the generation window. NSYS_ARGS=none serves the same tree with the
# probe off and no profiler: the clean boot of the same cell (K10). LINES=capture adds the
# op labels at graph capture. WARM_SAMPLES (default 32) run before the window; MAX_SAMPLES
# (default 128) bound the report. Only the serve
# is sent TERM; nsys finalizes by itself. Export and census run here, in the container.
# usage: run_probe_boot.sh <tree> <out dir> <card> <port> <h200|bf16|fp8> <concurrency> <arm>
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4 DTYPE=$5 CONC=$6 ARM=$7
S=$(cd "$(dirname "$0")" && pwd)
case $DTYPE in
  bf16) MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct CONFIG=examples/configs/qwen3_omni_colocated_h100_bf16.yaml ;;
  fp8) MODEL=marksverdhei/Qwen3-Omni-30B-A3B-FP8 CONFIG=examples/configs/qwen3_omni_colocated_h100_fp8.yaml ;;
  h200) MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct CONFIG=examples/configs/qwen3_omni_colocated_h200.yaml ;;
  *) echo "dtype must be bf16, fp8 or h200"; exit 1 ;;
esac
SERVE_ARGS="--config $CONFIG --colocate --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768 ${EXTRA_SERVE_ARGS:-}"
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
# a server already answering on the port would be benchmarked under this cell's name
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then
  echo "port $PORT already serves" > "$OUT/FAILED"
  exit 1
fi

echo "start $(date +%T) card $CARD dtype $DTYPE conc $CONC arm $ARM samples ${MAX_SAMPLES:-128}" > "$OUT/progress.txt"
git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
git -C "$TREE" status --short > "$OUT/tree_status.txt"
md5sum "$0" "$S/run_bench.py" "$S/omni_census.py" "$S/omni_pipeline_nvtx/sitecustomize.py" > "$OUT/md5.txt"
nsys --version > "$OUT/nsys_version.txt" 2>&1
echo "$SERVE_ARGS | $NSYS_ARGS | $PROBE" > "$OUT/serve_args.txt"
nvidia-smi > "$OUT/gpus_before.txt"
cat /proc/loadavg > "$OUT/loadavg_before.txt"
(while true; do cat /proc/loadavg; sleep 30; done) > "$OUT/loadavg.log" 2>&1 &
LOADS=$!
nvidia-smi dmon -i "$CARD" -s pucm -d 1 > "$OUT/dmon.log" 2>&1 &
DMON=$!

env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE${PROBE:+:$S/omni_pipeline_nvtx} $PROBE \
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
  # the warm pass runs the arm's first samples at the same concurrency outside the window,
  # so one-time costs (lazy compiles, first shapes) land before it; the measured pass skips
  # them, so no measured prompt hits the thinker's radix cache from the warm pass
  CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE python3 "$S/run_bench.py" gen --arm "$ARM" --port "$PORT" \
    --concurrency "$CONC" --out "$OUT/warm" --max-samples "${WARM_SAMPLES:-32}" > "$OUT/warm_$ARM.log" 2>&1
  echo "warm rc $? $(date +%T)" >> "$OUT/progress.txt"
  date +%s.%N > "$OUT/window.txt"
  CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE python3 "$S/run_bench.py" gen --arm "$ARM" --port "$PORT" \
    --concurrency "$CONC" --out "$OUT" --max-samples "${MAX_SAMPLES:-128}" \
    --skip-samples "${WARM_SAMPLES:-32}" > "$OUT/gen_$ARM.log" 2>&1
  echo "gen rc $? $(date +%T)" >> "$OUT/progress.txt"
  date +%s.%N >> "$OUT/window.txt"
else
  echo "server not healthy" > "$OUT/FAILED"
fi

# the serve is the python process on our port among the wrapper's descendants (nsys starts
# it in a session of its own, so it is not in the wrapper's process group); nsys, when
# present, finalizes after it exits
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
kill $LOADS $DMON 2>/dev/null
nvidia-smi > "$OUT/gpus_after.txt"
cat /proc/loadavg > "$OUT/loadavg_after.txt"
find "$OUT" -name '*.wav' -delete
[ -f "$OUT/FAILED" ] && exit 1

if [ "$NSYS_ARGS" != none ]; then
  nsys export --type sqlite --force-overwrite=true -o "$OUT/serve.sqlite" "$OUT/serve.nsys-rep" > "$OUT/export.log" 2>&1
  echo "exported $(date +%T)" >> "$OUT/progress.txt"
  (cd "$S" && python3 omni_census.py "$OUT/serve.sqlite" --window "$OUT/window.txt") > "$OUT/census.txt" 2>&1
fi
echo "done $(date +%T)" >> "$OUT/progress.txt"
touch "$OUT/DONE"
