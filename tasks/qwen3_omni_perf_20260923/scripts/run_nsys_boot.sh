#!/usr/bin/env bash
# Nsight Systems pass over one colocated Qwen3-Omni server: the whole serve under nsys
# profile with the pipeline NVTX annotations on (SGLANG_OMNI_PIPELINE_NVTX=1, from the
# prof branch), one run_bench.py arm as the load. The generation window's wall clock is
# written to window.txt so the readers cut startup, graph capture and warmup away.
# MAX_SAMPLES (default 64) bounds the report size. Only the serve is sent TERM; nsys
# finalizes by itself. Export and readers run here, in the container.
# usage: run_nsys_boot.sh <tree> <out dir> <card> <port> <bf16|fp8> <concurrency> <arm>
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
# GPU metrics and CPU sampling need a privileged container; NSYS_ARGS replaces the flags
NSYS_ARGS=${NSYS_ARGS:---trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none}
PIN=${PIN_CPUS:+numactl --physcpubind=$PIN_CPUS --membind=0}
mkdir -p "$OUT"
cd "$TREE" || exit 1

echo "start $(date +%T) card $CARD dtype $DTYPE conc $CONC arm $ARM samples ${MAX_SAMPLES:-64}" > "$OUT/progress.txt"
git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
md5sum "$0" "$S/run_bench.py" "$S/nsys_stage_ledger.py" > "$OUT/md5.txt"
nsys --version > "$OUT/nsys_version.txt" 2>&1
echo "$SERVE_ARGS | $NSYS_ARGS" > "$OUT/serve_args.txt"
nvidia-smi > "$OUT/gpus_before.txt"
cat /proc/loadavg > "$OUT/loadavg_before.txt"

env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE SGLANG_OMNI_PIPELINE_NVTX=1 \
  $PIN nsys profile -o "$OUT/serve" --force-overwrite=true $NSYS_ARGS \
  python3 -u -m sglang_omni.cli serve --model-path $MODEL $SERVE_ARGS --host 127.0.0.1 --port $PORT \
  > "$OUT/serve.log" 2>&1 &
NSYS_PID=$!

healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  kill -0 $NSYS_PID 2>/dev/null || break
  sleep 5
done
if [ $healthy = 1 ]; then
  echo "healthy $(date +%T)" >> "$OUT/progress.txt"
  date +%s.%N > "$OUT/window.txt"
  CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE $PIN python3 "$S/run_bench.py" gen --arm "$ARM" --port "$PORT" \
    --concurrency "$CONC" --out "$OUT" --max-samples "${MAX_SAMPLES:-64}" > "$OUT/gen_$ARM.log" 2>&1
  echo "gen rc $? $(date +%T)" >> "$OUT/progress.txt"
  date +%s.%N >> "$OUT/window.txt"
else
  echo "server not healthy" > "$OUT/FAILED"
fi

# nsys carries the serve command line too; the serve is the python process on our port
SERVE_PID=$(for p in $(pgrep -f "sglang_omni.cli serve.*--port $PORT"); do
  if [[ "$(cat /proc/$p/comm)" == python* ]]; then echo $p; fi; done | head -1)
echo "serve pid $SERVE_PID" >> "$OUT/progress.txt"
kill -TERM "$SERVE_PID"
for _ in $(seq 120); do
  kill -0 $NSYS_PID 2>/dev/null || break
  sleep 5
done
kill -0 $NSYS_PID 2>/dev/null && echo "nsys still running after 10 min" >> "$OUT/progress.txt"
echo "nsys ended $(date +%T)" >> "$OUT/progress.txt"
nvidia-smi > "$OUT/gpus_after.txt"
cat /proc/loadavg > "$OUT/loadavg_after.txt"
find "$OUT" -name '*.wav' -delete
[ -f "$OUT/FAILED" ] && exit 1

nsys export --type sqlite --force-overwrite=true -o "$OUT/serve.sqlite" "$OUT/serve.nsys-rep" > "$OUT/export.log" 2>&1
python3 "$S/nsys_stage_ledger.py" "$OUT/serve.sqlite" --window "$OUT/window.txt" > "$OUT/stage_ledger.txt" 2>&1
echo "done $(date +%T)" >> "$OUT/progress.txt"
touch "$OUT/DONE"
