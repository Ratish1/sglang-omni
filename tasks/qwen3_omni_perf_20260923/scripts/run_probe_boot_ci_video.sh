#!/usr/bin/env bash
# One boot of a Qwen3-Omni CI video talker stage as CI serves it, cold, for design 13's profile:
# examples/run_qwen3_omni_speech_server.py with the conftest arguments of the stage, the
# omni_pipeline_nvtx probe under nsys (NSYS_ARGS=none serves the same cell clean), CI's samples
# at CI's concurrency as the one measured pass (no warm pass: CI's stage starts cold), then the
# census over that window. The container sees the stage's two cards as 0 and 1.
#   videomme_talker   stage 8: bf16 disaggregated, thinker and encoders on 0, talker and code2wav on 1, 20 samples
#   videoamme_talker  stage 10: FP8, thinker TP2 on 0 and 1, talker and code2wav on 1, 10 samples
# PROBE_PATH puts another sitecustomize directory on the server's PYTHONPATH (with NSYS_ARGS=none,
# e.g. ipc_payload_probe with OMNI_IPC_PROBE=<dir>).
# usage: run_probe_boot_ci_video.sh <tree> <out dir> <port> <videomme_talker|videoamme_talker>
set -u
TREE=$1 OUT=$2 PORT=$3 ARM=$4
S=$(cd "$(dirname "$0")" && pwd)
case $ARM in
  videomme_talker) MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct SAMPLES=20
    EXTRA="--gpu-thinker 0 --gpu-image-encoder 0 --gpu-audio-encoder 0 --gpu-talker 1 --gpu-code2wav 1 --thinker-mem-fraction-static 0.82 --talker-mem-fraction-static 0.40" ;;
  videoamme_talker) MODEL=marksverdhei/Qwen3-Omni-30B-A3B-FP8 SAMPLES=10
    EXTRA="--thinker-tp-size 2 --gpu-thinker-tp 0,1 --gpu-talker 1 --gpu-code2wav 1 --thinker-mem-fraction-static 0.40 --talker-mem-fraction-static 0.21" ;;
  *) echo "arm must be videomme_talker or videoamme_talker"; exit 1 ;;
esac
NSYS_ARGS=${NSYS_ARGS:---trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=node --gpuctxsw=true --sample=none --cpuctxsw=none}
if [ "$NSYS_ARGS" = none ]; then
  WRAP="" PROBE=""
else
  WRAP="nsys profile -o $OUT/serve --force-overwrite=true $NSYS_ARGS"
  PROBE="OMNI_PIPE_NVTX=1"
fi
PIN=${PIN_CPUS:+numactl --physcpubind=$PIN_CPUS --membind=${PIN_NODE:-0}}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$TREE" || exit 1
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 > 1024 {print $1 ":" $2 "MiB"}')
if [ -n "$busy" ]; then
  echo "cards busy before boot: $busy" > "$OUT/FAILED"
  exit 1
else
  :
fi

echo "start $(date +%T) $ARM samples $SAMPLES nsys '${NSYS_ARGS}'" > "$OUT/progress.txt"
git -C "$TREE" rev-parse HEAD HEAD:sglang_omni > "$OUT/head.txt"
md5sum "$0" "$S/run_bench.py" "$S/omni_census.py" "$S/omni_pipeline_nvtx/sitecustomize.py" ${PROBE_PATH:+"$PROBE_PATH/sitecustomize.py"} > "$OUT/md5.txt"
echo "$MODEL | $EXTRA | $NSYS_ARGS | $PROBE" > "$OUT/serve_args.txt"
nvidia-smi > "$OUT/gpus_before.txt"
(while true; do cat /proc/loadavg; sleep 30; done) > "$OUT/loadavg.log" 2>&1 &
LOADS=$!

env PYTHONPATH=$TREE${PROBE:+:$S/omni_pipeline_nvtx}${PROBE_PATH:+:$PROBE_PATH} $PROBE \
  setsid $PIN $WRAP python3 -u examples/run_qwen3_omni_speech_server.py --model-path $MODEL --port $PORT \
  --model-name qwen3-omni --thinker-max-seq-len 32768 $EXTRA > "$OUT/serve.log" 2>&1 &
WRAP_PID=$!
echo "pgid $WRAP_PID" >> "$OUT/progress.txt"

healthy=0
for _ in $(seq 240); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  kill -0 $WRAP_PID 2>/dev/null || break
  sleep 5
done
if [ $healthy = 1 ]; then
  echo "healthy $(date +%T)" >> "$OUT/progress.txt"
  date +%s.%N > "$OUT/window.txt"
  PYTHONPATH=$TREE $PIN python3 "$S/run_bench.py" gen --arm "$ARM" --port "$PORT" \
    --concurrency 16 --out "$OUT" --max-samples "$SAMPLES" > "$OUT/gen_$ARM.log" 2>&1
  echo "gen $ARM rc $? $(date +%T)" >> "$OUT/progress.txt"
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
if [ -n "$SERVE_PID" ]; then
  kill -TERM "$SERVE_PID"
else
  :
fi
for _ in $(seq 120); do
  kill -0 $WRAP_PID 2>/dev/null || break
  sleep 5
done
if kill -0 $WRAP_PID 2>/dev/null; then
  echo "serve or wrapper still running after 10 min; cell void" >> "$OUT/progress.txt"
  echo "serve not stopped" > "$OUT/FAILED"
else
  :
fi
echo "serve ended $(date +%T)" >> "$OUT/progress.txt"
kill $LOADS 2>/dev/null
grep -h "omni_pipeline_nvtx patch" "$OUT/serve.log" > "$OUT/probe_patch_errors.txt" 2>/dev/null
find "$OUT" -name '*.wav' -delete
if [ -f "$OUT/FAILED" ]; then
  exit 1
else
  :
fi
if [ "$NSYS_ARGS" != none ]; then
  nsys export --type sqlite --force-overwrite=true -o "$OUT/serve.sqlite" "$OUT/serve.nsys-rep" > "$OUT/export.log" 2>&1
  echo "exported $(date +%T)" >> "$OUT/progress.txt"
  (cd "$S" && python3 omni_census.py "$OUT/serve.sqlite" --window "$OUT/window.txt" > "$OUT/census.txt" 2>&1)
  echo "census done $(date +%T)" >> "$OUT/progress.txt"
else
  :
fi
echo "done $(date +%T)" >> "$OUT/progress.txt"
touch "$OUT/DONE"
