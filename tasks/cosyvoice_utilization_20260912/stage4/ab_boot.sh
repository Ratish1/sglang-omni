#!/usr/bin/env bash
# One A/B arm boot: the server from the arm's tree, pinned to the card's cores, then the
# repo's SeedTTS benchmark against it, warmup 1: by default a seeded streaming c1 pass over the first
# 64 English samples (identity and c1 speed), then an unseeded c16 pass over the whole
# English split. The client code is the main tree's for every arm. Scoring is score_arm.sh.
#
# usage: ab_boot.sh <tree> <out dir> <card> <port> <cpus> [inductor cache dir]
#   EXTRA_ENV  extra server environment, e.g. the GPU core dump variables
#   CELLS      "name:concurrency:samples:seed:stream[:meta] ...", stream 1 or 0, meta a local
#              meta.lst in place of the SeedTTS split
#              (default "c1:1:64:1234:1 c16:16:::1"; b16:16:::0 adds buffered c16)
#   SERVE_ARGS extra serve arguments
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4 CPUS=$5 CACHE=${6:-}
CELLS=${CELLS:-"c1:1:64:1234:1 c16:16:::1"}
CLIENT=/workspace/sglang-omni
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
mkdir -p "$OUT"
git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
cp "$TREE/BRANCH_MARKER.txt" "$OUT/" 2>/dev/null
(cd "$TREE" && PYTHONPATH="$TREE" python3 -c "import sglang_omni; print(sglang_omni.__file__)") > "$OUT/import_path.txt"
grep -q "^$TREE/" "$OUT/import_path.txt" || { echo "server would import $(cat "$OUT/import_path.txt")" > "$OUT/FAILED"; exit 1; }
nvidia-smi > "$OUT/gpus_before.txt"
echo "start $(date +%T) tree=$TREE card=$CARD port=$PORT cpus=$CPUS cache=${CACHE:-default}" > "$OUT/progress.txt"
(while true; do echo "$(date +%s) $(cat /proc/loadavg)"; sleep 30; done) > "$OUT/loadavg.log" 2>&1 &
LOADS=$!
nvidia-smi dmon -i "$CARD" -s pucm -d 1 > "$OUT/dmon.log" 2>&1 &
DMON=$!
SERVE_ENV=(CUDA_VISIBLE_DEVICES=$CARD SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$TREE ${EXTRA_ENV:-})
if [ -n "$CACHE" ]; then
  SERVE_ENV+=(TORCHINDUCTOR_CACHE_DIR=$CACHE)
else
  :
fi
echo "taskset -c $CPUS env ${SERVE_ENV[*]} python3 -u -m sglang_omni.cli serve --model-path $MODEL --host 127.0.0.1 --port $PORT ${SERVE_ARGS:-}" > "$OUT/serve_cmd.txt"
(cd "$TREE" && taskset -c "$CPUS" env "${SERVE_ENV[@]}" python3 -u -m sglang_omni.cli serve \
  --model-path $MODEL --host 127.0.0.1 --port $PORT ${SERVE_ARGS:-} > "$OUT/serve.log" 2>&1) &
LAUNCH=$!

healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  kill -0 $LAUNCH 2>/dev/null || break
  sleep 5
done
if [ $healthy = 1 ]; then
  echo "healthy $(date +%T)" >> "$OUT/progress.txt"
  for cell in $CELLS; do
    IFS=: read -r name concurrency samples seed stream meta <<< "$cell"
    if [ "$stream" = 1 ]; then STREAM=--stream; else STREAM=; fi
    (cd "$CLIENT" && taskset -c "$CPUS" env CUDA_VISIBLE_DEVICES= python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --model $MODEL --port $PORT --lang en --max-concurrency "$concurrency" --warmup 1 \
      --use-existing-server --generate-only $STREAM ${samples:+--max-samples $samples} ${seed:+--seed $seed} \
      ${meta:+--meta $meta} --output-dir "$OUT/$name") > "$OUT/$name.log" 2>&1
    echo "cell $name rc $? $(date +%T)" >> "$OUT/progress.txt"
  done
else
  echo "server not healthy" > "$OUT/FAILED"
fi

SERVE_PID=$(for p in $(pgrep -f "sglang_omni.cli serve.*--port $PORT"); do
  if [[ "$(cat /proc/$p/comm)" == python* ]]; then echo $p; fi; done | head -1)
echo "serve pid $SERVE_PID" >> "$OUT/progress.txt"
[ -n "$SERVE_PID" ] && kill -TERM "$SERVE_PID"
for _ in $(seq 180); do
  kill -0 $LAUNCH 2>/dev/null || break
  sleep 5
done
kill $LOADS $DMON 2>/dev/null
nvidia-smi > "$OUT/gpus_after.txt"
echo "done $(date +%T)" >> "$OUT/progress.txt"
